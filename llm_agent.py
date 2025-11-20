import google.generativeai as genai
import numpy as np
import json
import os
from google.generativeai.types import generation_types
import google.generativeai.protos as protos
from google.protobuf import struct_pb2, json_format
from proto.marshal.collections.maps import MapComposite
from typing import Dict, Any, List, Optional

from collective_memory import CollectiveMemory
from digital_twin import DigitalTwin
import e2_api_tool
from config import NUM_TIME_STEPS
import network_simulator


# Import the parsing function
from negotiation_parser import parse_agent_message


from config import SLA_LATENCY_THRESHOLD_MS

# Configure the genai library
try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Could not configure Google Generative AI: {e}")

class LLMAgent:
    def __init__(self, name: str, role: str, model_name: str = 'gemini-2.5-flash-preview-09-2025', tools: List[Any] = None,
                 collective_memory_tool: Optional[CollectiveMemory] = None, digital_twin_instance: Optional[DigitalTwin] = None,
                 debiased_memory_prompt_enabled: bool = False): # New parameter
        self.name = name
        self.role = role
        
        all_tools = []
        if tools:
            all_tools.extend(tools)

        self.model = genai.GenerativeModel(model_name=model_name, tools=all_tools)
        self.chat_session = self.model.start_chat(history=[]) # Initialize chat session here once
        self.negotiation_goal = ""
        self.last_proposed_config = None
        self.collective_memory_tool = collective_memory_tool
        self.digital_twin = digital_twin_instance # Store the DigitalTwin instance
        self.current_metrics_for_query = None
        self.max_llm_response_retries = 5 # Defined here
        self.debiased_memory_prompt_enabled = debiased_memory_prompt_enabled # Store the flag

    def _sanitize_text_for_prompt(self, text: str) -> str:
        """Sanitizes text to ensure it's safe for inclusion in LLM prompt parts.
        Ensures the returned string is never empty if the input was not empty,
        to prevent 'contents.parts must not be empty' errors."""
        if not text:
            return ""
        
        # Remove non-printable ASCII characters (except common whitespace)
        # and ensure it's valid UTF-8.
        cleaned_text = "".join(char for char in text if 31 < ord(char) < 127 or char in "\n\r\t ")
        
        # If all characters were filtered out but the original text was not empty,
        # return a single space to avoid the 'contents.parts must not be empty' error.
        if not cleaned_text and text:
            return " "
            
        return cleaned_text.encode('utf-8', 'ignore').decode('utf-8')

    def _extract_text_from_response(self, response: Any) -> str:
        """Extracts text content from a model response, returning a default if unparseable."""
        if not response or not response.candidates:
            # Return a default message instead of raising ValueError
            return "Model did not provide a valid text response. Please try again."
        
        text_parts = []
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                text_parts.append(part.text)
            elif hasattr(part, 'function_call') and part.function_call:
                # If a function call is encountered when text is expected, return a message
                print(f"Warning: Model generated an unexpected function call when text was expected. Part: {part}")
                return "Model generated an unexpected tool call instead of a text response. Please rephrase your query."
        
        extracted_text = " ".join(text_parts).strip()
        if not extracted_text:
            # Return a default message instead of raising ValueError
            return "Model response was empty or unparseable. Please try again."
        return extracted_text


    def _call_tool_if_needed(self, response: Any):
        tool_outputs = []
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    tool_call = part.function_call
                    function_name = tool_call.name
                    args = tool_call.args

                    def convert_to_python_native(obj):
                        if isinstance(obj, MapComposite):
                            return {k.strip("'\""): convert_to_python_native(v) for k, v in obj.items()}
                        elif isinstance(obj, (struct_pb2.Struct, struct_pb2.ListValue)):
                            return json.loads(json_format.MessageToJson(obj))
                        elif isinstance(obj, np.bool_):
                            return bool(obj)
                        elif isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, (list, tuple)):
                            return [convert_to_python_native(elem) for elem in obj]
                        elif isinstance(obj, dict):
                            return {k: convert_to_python_native(v) for k, v in obj.items()}
                        else:
                            return obj

                    if function_name == "query_memory" and self.collective_memory_tool:
                        print(f"[{self.name}] Calling Collective Memory tool: {function_name} with args: {args}")
                        try:
                            query_context_from_llm = convert_to_python_native(args)

                            actual_query_context = {}
                            if "query_context" in query_context_from_llm and isinstance(query_context_from_llm["query_context"], dict):
                                actual_query_context = query_context_from_llm["query_context"]
                                if "current_time_step" not in actual_query_context and "current_time_step" in query_context_from_llm:
                                    actual_query_context["current_time_step"] = query_context_from_llm["current_time_step"]
                            else:
                                actual_query_context = query_context_from_llm

                            if "current_time_step" not in actual_query_context and self.current_metrics_for_query:
                                actual_query_context["current_time_step"] = self.current_metrics_for_query.get("current_time_step", NUM_TIME_STEPS)
                            
                            # Ensure current_trial_number is passed to query_memory
                            if "current_trial_number" not in actual_query_context and hasattr(self, 'current_trial_num'):
                                actual_query_context["current_trial_number"] = self.current_trial_num

                            # Add current latency to query context for better failure retrieval
                            if "latency_ms" not in actual_query_context and self.current_metrics_for_query:
                                actual_query_context["latency_ms"] = self.current_metrics_for_query.get("latency_ms")

                            # Add current traffic to query context for better retrieval
                            if "current_traffic_arrival_rate_bps" not in actual_query_context and self.current_metrics_for_query:
                                actual_query_context["current_traffic_arrival_rate_bps"] = self.current_metrics_for_query.get("current_traffic_arrival_rate_bps")


                            result_data = self.collective_memory_tool.query_memory(query_context=actual_query_context)
                            
                            # Ensure result_data is a dict, even if empty or None
                            if not isinstance(result_data, dict):
                                result_data = {}
                            
                            print(f"[{self.name}] Collective Memory Query Result: {json.dumps(result_data, indent=2)}")

                            tool_outputs.append({"function_name": function_name, "result": convert_to_python_native(result_data)})
                        except Exception as e:
                            print(f"[{self.name}] Error calling Collective Memory tool {function_name}: {e}")
                            tool_outputs.append({"function_name": function_name, "error": f"Error: {e}"})
                    elif hasattr(e2_api_tool, function_name):
                        tool_func = getattr(e2_api_tool, function_name)
                        try:
                            result_data = tool_func(**args)
                            parsed_result = convert_to_python_native(result_data) # Ensure parsed_result is always assigned

                            if not isinstance(parsed_result, dict):
                                print(f"[{self.name}] Warning: Tool result for {function_name} is not a dict: {parsed_result}. Converting to empty dict.") # Corrected line
                                parsed_result = {}
                            
                            # Ensure parsed_result is never an empty dictionary
                            if not parsed_result:
                                parsed_result = {"status": "success", "message": "Tool executed, but returned no specific data."}

                            tool_outputs.append({"function_name": function_name, "result": parsed_result})
                        except Exception as e:
                            print(f"[{self.name}] Error calling E2 API tool {function_name}: {e}")
                            tool_outputs.append({"function_name": function_name, "error": f"Error: {e}"})
                    else:
                        print(f"[{self.name}] Unknown tool requested: {function_name}")
                        tool_outputs.append({"function_name": function_name, "error": f"Unknown tool: {function_name}"})
        return tool_outputs

    def make_negotiation_move(self, opposing_agent_message: str, current_metrics: Dict[str, Any], iteration: int, max_iterations: int, negotiation_ended: bool = False, current_trial_num: int = 0) -> str:
        
        self.current_metrics_for_query = current_metrics
        self.current_trial_num = current_trial_num # Store current trial number

        if negotiation_ended:
            return "The negotiation is complete. No further action is needed."

        # Ensure traffic_level_category and current_traffic_bps are always defined
        current_traffic_bps = current_metrics.get("current_traffic_arrival_rate_bps", 0)
        if current_traffic_bps < network_simulator._TRAFFIC_BASE_SLICE * 0.8:
            traffic_level_category = "low"
        elif current_traffic_bps > network_simulator._TRAFFIC_BASE_SLICE * 1.2:
            traffic_level_category = "high"
        else:
            traffic_level_category = "medium"

        memory_summary = ""
        memory_insights = {} # Initialize memory_insights here
        # Force memory invocation if collective_memory_tool is available
        if self.collective_memory_tool:
            # Define the args dictionary for the function call
            function_call_args = {
                "query_context": {
                    "traffic_level_category": traffic_level_category, # Pass category
                    "current_traffic_arrival_rate_bps": current_traffic_bps, # Pass exact value
                    "agent_role": self.role.split(' ')[0].lower(),
                    "event_type": "strategy_guidance",
                    "current_time_step": current_metrics.get("current_time_step", NUM_TIME_STEPS),
                    "current_trial_number": current_trial_num, # Pass current trial number
                    "keywords": ["successful agreement", "latency", "energy", "compromise", "failure", "unresolved", "violation", "high latency", "low latency", "energy saving"], # Added more keywords
                    "latency_ms": current_metrics.get("latency_ms") # Add current latency to query context
                }
            }

            # Create the FunctionCall protobuf object
            function_call_proto = protos.FunctionCall(
                name="query_memory",
                args=function_call_args
            )

            # Create the Part protobuf object
            part_proto = protos.Part(
                function_call=function_call_proto
            )

            # Dynamically create the MockContent class
            MockContent = type('MockContent', (object,), {
                'parts': [part_proto]
            })

            # Dynamically create the MockCandidate class
            MockCandidate = type('MockCandidate', (object,), {
                'content': MockContent() # Instantiate MockContent
            })

            # Dynamically create the MockResponse class and instantiate it
            mock_response_for_tool_call = type('MockResponse', (object,), {
                'candidates': [MockCandidate()] # Instantiate MockCandidate
            })()
            
            tool_outputs = self._call_tool_if_needed(mock_response_for_tool_call)
            
            # Process tool outputs to create memory_summary
            for output_item in tool_outputs:
                if output_item.get("function_name") == "query_memory" and "result" in output_item:
                    memory_insights = output_item["result"]
                    break
            
            if memory_insights:
                memory_summary_parts = ["\n\n**Insights from Collective Memory:**"]
                
                inferred_config = memory_insights.get("inferred_successful_config")
                if inferred_config:
                    memory_summary_parts.append(f"\nInferred BEST successful configuration (consider proposing this or similar):")
                    memory_summary_parts.append(f"- RAN BW: {inferred_config['ran_bandwidth_mhz']:.1f} MHz, Edge CPU: {inferred_config['edge_cpu_frequency_ghz']:.1f} GHz")
                
                patterns_to_avoid = memory_insights.get("patterns_to_avoid")
                if patterns_to_avoid:
                    memory_summary_parts.append("\nPast failed negotiation patterns (actively avoid these to prevent SLA violations and improve efficiency):")
                    for fp in patterns_to_avoid:
                        last_ran = fp.get('last_ran_proposal_mhz')
                        last_edge = fp.get('last_edge_proposal_ghz')
                        reason = fp.get('reason', 'Unknown reason')
                        sla_violation_info = " (SLA VIOLATION)" if fp.get('sla_violation_occurred') else ""
                        
                        # Include context, especially traffic level and exact values
                        failed_context = fp.get('context', {})
                        failed_traffic_level_category = failed_context.get('traffic_level_category', 'unknown')
                        failed_traffic_bps = failed_context.get('current_traffic_arrival_rate_bps', 'N/A')
                        failed_trial_num = failed_context.get('trial_number', 'N/A')

                        memory_summary_parts.append(f"- Failed in Trial {failed_trial_num} due to '{reason}'{sla_violation_info}. This occurred under '{failed_traffic_level_category}' traffic ({failed_traffic_bps} bps). Last RAN proposed: {last_ran}. Last Edge proposed: {last_edge}.")
                
                if not inferred_config and not patterns_to_avoid:
                    memory_summary_parts.append("No highly relevant past strategies found to distill concrete suggestions.")
                
                # Emphasize learning and reasoning over direct application
                memory_summary_parts.append("\n**Crucially, use these insights to inform your reasoning and adapt your proposal for the current situation.** Focus on *why* past strategies succeeded or failed, and *how* to adjust to avoid conflicts and meet SLAs. Do not simply copy past actions; instead, learn and apply that learning intelligently. **Prioritize avoiding negotiation conflicts.**")
                memory_summary = "\n".join(memory_summary_parts)
            else:
                memory_summary = "\n\n**Collective Memory:** No relevant past strategies found yet. Proceed with your best judgment."

        # Sanitize opposing_agent_message and memory_summary before adding to prompt parts
        sanitized_opposing_agent_message = self._sanitize_text_for_prompt(opposing_agent_message)
        sanitized_memory_summary = self._sanitize_text_for_prompt(memory_summary)

        # Initial prompt for the agent to generate a proposal or accept/reject
        initial_agent_prompt_parts = [
            {"text": f"Current network metrics for the single cross-domain slice: {json.dumps(current_metrics, indent=2)}\n"},
            {"text": f"You are the {self.role}. Your goal is to {self.negotiation_goal}.\n"},
            {"text": f"The opposing agent ({'EDGE_AGENT' if self.name == 'RAN_AGENT' else 'RAN_AGENT'}) said: '{sanitized_opposing_agent_message}'\n"},
            {"text": f"This is negotiation iteration {iteration + 1} out of {max_iterations}. Formulate your next move (PROPOSE_ACTION, ACCEPT_AGREEMENT, NO_AGREEMENT_POSSIBLE). Provide specific values for ran_bandwidth_mhz and edge_cpu_frequency_ghz if proposing/accepting. Explain your reasoning. Try to reach an agreement within the given iterations. Remember to use only numerical values for bandwidth and CPU, and use double quotes for JSON keys. **Your response MUST strictly adhere to one of these formats: PROPOSE_ACTION: {{...}}, ACCEPT_AGREEMENT: {{...}}, or NO_AGREEMENT_POSSIBLE. Do NOT include any direct tool calls or other text outside these specified JSON structures in your final negotiation message. When providing the 'reason' in your JSON, ensure the entire string value for 'reason' is valid JSON. This means any double quotes *within* your descriptive text must be escaped with a backslash (e.g., `\\\"This is an \\\\\\\"example\\\\\\\" text.\\\"` or `\\\"Configuration: \\\\\\\"RAN BW 39.2 MHz, Edge CPU 40.0 GHz.\\\\\\\"\\\"`). Do NOT attempt to embed JSON or dictionary-like structures within the 'reason' string.**"}
        ]

        if self.last_proposed_config:
            initial_agent_prompt_parts.append({"text": f"Your last proposed configuration was: RAN BW = {self.last_proposed_config.get('ran_bandwidth_mhz')} MHz, Edge CPU = {self.last_proposed_config.get('edge_cpu_frequency_ghz')} GHz."})

        # Only append sanitized_memory_summary if it's not empty
        if sanitized_memory_summary:
            initial_agent_prompt_parts.append({"text": sanitized_memory_summary})

        # Send the initial prompt for this negotiation turn
        response = None
        # Initialize parsed_move to a default state to prevent UnboundLocalError
        parsed_move = {"intent": "PARSING_FAILED", "parameters": {"reason": "Initial LLM call failed or response unparseable."}}
        try:
            response = self.chat_session.send_message(initial_agent_prompt_parts)
        except Exception as e:
            print(f"[{self.name}] Initial LLM call failed: {e}. Declaring NO_AGREEMENT_POSSIBLE for this negotiation turn.")
            # If initial LLM call fails, set final_negotiation_message and return immediately
            return "NO_AGREEMENT_POSSIBLE: An internal error occurred during initial LLM call."

        final_negotiation_message = None
        for attempt in range(self.max_llm_response_retries):
            try:
                # Step 1: Process any tool calls in the current response
                tool_outputs = self._call_tool_if_needed(response)
                if tool_outputs:
                    # If there were tool calls, send their results back to the model
                    tool_response_parts = []
                    for output_item in tool_outputs:
                        if "error" in output_item:
                            tool_response_parts.append(
                                protos.Part(
                                    function_response=protos.FunctionResponse(
                                        name=output_item["function_name"],
                                        response=struct_pb2.Struct(
                                            fields={
                                                "error": struct_pb2.Value(string_value=output_item["error"])
                                            }
                                        )
                                    )
                                )
                            )
                        else:
                            parsed_result = output_item["result"]
                            if not isinstance(parsed_result, dict):
                                print(f"[{self.name}] Warning: Tool result for {output_item['function_name']} is not a dict: {parsed_result}. Converting to empty dict.") # Corrected line
                                parsed_result = {}
                            
                            # Ensure parsed_result is never an empty dictionary
                            if not parsed_result:
                                parsed_result = {"status": "success", "message": "Tool executed, but returned no specific data."}

                            tool_response_parts.append(
                                protos.Part(
                                    function_response=protos.FunctionResponse(
                                        name=output_item["function_name"],
                                        response=json_format.ParseDict(parsed_result, struct_pb2.Struct())
                                    )
                                )
                            )
                    response = self.chat_session.send_message(tool_response_parts)
                    continue # Go to the next iteration to get the model's *new* response (which should now be text or another tool call)

                # Step 2: If no tool calls in the response, it should be a text message
                negotiation_message = self._extract_text_from_response(response)
                
            
                # Call the refactored parsing function.
                parsed_move = parse_agent_message(negotiation_message)
    


                is_late_iteration_fallback_attempt = False
                if self.collective_memory_tool and iteration >= max_iterations - 2: # Last two iterations
                    is_late_iteration_fallback_attempt = True


                if parsed_move["intent"] == "PROPOSE_ACTION":
                    proposed_ran_bw = parsed_move["parameters"].get("ran_bandwidth_mhz")
                    proposed_edge_cpu = parsed_move["parameters"].get("edge_cpu_frequency_ghz")

                    if proposed_ran_bw is None or proposed_edge_cpu is None:
                        raise ValueError("PROPOSE_ACTION missing ran_bandwidth_mhz or edge_cpu_frequency_ghz.")

                    # --- Digital Twin Test ---
                    if self.digital_twin:
                        self.digital_twin.reset_to_current_state(current_metrics)
                        predicted_metrics = self.digital_twin.simulate_step_for_prediction(
                            proposed_ran_bandwidth_mhz=proposed_ran_bw,
                            proposed_edge_cpu_frequency_ghz=proposed_edge_cpu
                        )
                        predicted_latency = predicted_metrics["predicted_latency_ms"]
                        predicted_energy = predicted_metrics["predicted_energy_watts"]
                        predicted_cpu_conflict = predicted_metrics["predicted_cpu_allocation_conflict_count"]

                        # Check SLA and CPU conflicts
                        if predicted_latency > SLA_LATENCY_THRESHOLD_MS or predicted_cpu_conflict > current_metrics["cpu_allocation_conflict_count"]:
                            print(f"[{self.name}] Digital Twin test failed for proposal (BW: {proposed_ran_bw:.1f}, CPU: {proposed_edge_cpu:.1f}). Predicted Latency: {predicted_latency:.2f}ms (SLA: {SLA_LATENCY_THRESHOLD_MS}ms), Predicted CPU Conflicts: {predicted_cpu_conflict}.")
                            
                            # If late iteration and DT failed, try memory-guided fallback
                            if is_late_iteration_fallback_attempt and memory_insights.get("inferred_successful_config"):
                                inferred_config = memory_insights["inferred_successful_config"]
                                print(f"[{self.name}] Late iteration DT failure. Attempting memory-guided fallback: {inferred_config}")
                                
                                # Use inferred config as the new proposal
                                proposed_ran_bw_fallback = inferred_config["ran_bandwidth_mhz"]
                                proposed_edge_cpu_fallback = inferred_config["edge_cpu_frequency_ghz"]

                                # Re-run DT test with fallback proposal
                                predicted_metrics_fallback = self.digital_twin.simulate_step_for_prediction(
                                    proposed_ran_bandwidth_mhz=proposed_ran_bw_fallback,
                                    proposed_edge_cpu_frequency_ghz=proposed_edge_cpu_fallback
                                )
                                predicted_latency_fallback = predicted_metrics_fallback["predicted_latency_ms"]
                                predicted_energy_fallback = predicted_metrics_fallback["predicted_energy_watts"]
                                predicted_cpu_conflict_fallback = predicted_metrics_fallback["predicted_cpu_allocation_conflict_count"]

                                if predicted_latency_fallback <= SLA_LATENCY_THRESHOLD_MS and predicted_cpu_conflict_fallback <= current_metrics["cpu_allocation_conflict_count"]:
                                    # Memory-guided fallback passed DT test
                                    parsed_move["parameters"]["ran_bandwidth_mhz"] = proposed_ran_bw_fallback
                                    parsed_move["parameters"]["edge_cpu_frequency_ghz"] = proposed_edge_cpu_fallback
                                    parsed_move["parameters"]["reason"] = (
                                        f"Original proposal failed DT. This is a memory-guided fallback proposal to ensure SLA compliance in late iteration. "
                                        f"Predicted Latency: {predicted_latency_fallback:.2f} ms, Predicted Energy: {predicted_energy_fallback:.2f} W."
                                    )
                                    parsed_move["parameters"]["reason"] = parsed_move["parameters"]["reason"].replace('"', '\\"')
                                    final_negotiation_message = f"PROPOSE_ACTION: {json.dumps(parsed_move['parameters'])}"
                                    break # Break out of internal loop
                                else:
                                    # Memory-guided fallback also failed DT test
                                    print(f"[{self.name}] Memory-guided fallback also failed DT. Declaring NO_AGREEMENT_POSSIBLE.")
                                    final_negotiation_message = "NO_AGREEMENT_POSSIBLE: Failed to find a valid proposal, even with memory-guided fallback, in late iteration."
                                    break # Break out of internal loop
                            else:
                                # Not late iteration or no memory insights for fallback, guide agent to adjust
                                latency_violation_severity = predicted_latency - SLA_LATENCY_THRESHOLD_MS
                                adjustment_hint = ""
                                if predicted_latency > SLA_LATENCY_THRESHOLD_MS:
                                    if latency_violation_severity < 1.0: # Small violation (<1ms over SLA)
                                        adjustment_hint += "The latency violation is slight. You should make a **small, incremental adjustment** to your proposal to just meet the SLA. "
                                    elif latency_violation_severity < 5.0: # Moderate violation (1-5ms over SLA)
                                        adjustment_hint += "The latency violation is moderate. You need to make a **more significant adjustment** to your proposal to ensure SLA compliance. "
                                    else: # Large violation (>5ms over SLA)
                                        adjustment_hint += "The latency violation is severe. You **MUST make a substantial increase** to bring latency down. This is a critical priority. "

                                    if self.role == "RAN Energy Saving Agent":
                                        adjustment_hint += "To improve latency, **increase RAN bandwidth (ran_bandwidth_mhz)**. "
                                    elif self.role == "Edge Latency Agent":
                                        adjustment_hint += "To improve latency, **increase Edge CPU frequency (edge_cpu_frequency_ghz)**. "
                                    
                                    adjustment_hint += f"Remember to consider the current traffic level ({traffic_level_category} traffic, {current_traffic_bps} bps) when deciding the magnitude of your adjustment. "

                                if predicted_cpu_conflict > current_metrics["cpu_allocation_conflict_count"]:
                                    adjustment_hint += "Additionally, your proposed Edge CPU frequency is causing a conflict. You MUST reduce it or ensure it's within the maximum allowed limits (up to 50 GHz)." # Updated fmax reference

                                # Emphasize reasoning and conflict avoidance
                                adjustment_hint += "**Crucially, NO_AGREEMENT_POSSIBLE should only be declared as a last resort, after you have made multiple, reasonable attempts to find a configuration that meets the SLA and avoids conflicts through intelligent reasoning and adaptation.** Do not give up after a single failed internal test. Adjust, re-test, and iterate. **Prioritize avoiding negotiation conflicts.**"

                                self_correction_prompt_parts = [
                                    {"text": f"Current network metrics: {json.dumps(current_metrics, indent=2)}\n"},
                                    {"text": f"You are the {self.role}. Your goal is to {self.negotiation_goal}.\n"},
                                    {"text": f"The opposing agent said: '{sanitized_opposing_agent_message}'\n"},
                                    {"text": f"**INTERNAL DIGITAL TWIN TEST FAILED (Attempt {attempt + 1}/{self.max_llm_response_retries}):** Your last proposed action (RAN BW: {proposed_ran_bw:.1f}, CPU: {proposed_edge_cpu:.1f}) resulted in: "
                                             f"Predicted Latency: {predicted_latency:.2f}ms (SLA: {SLA_LATENCY_THRESHOLD_MS}ms), Predicted CPU Conflicts: {predicted_cpu_conflict}. "
                                             f"This VIOLATES the SLA or causes CPU conflicts. {adjustment_hint} "
                                             f"**You MUST adjust your proposal to meet the latency SLA and avoid CPU conflicts.** "
                                             f"Provide a new `PROPOSE_ACTION` with adjusted values. Formulate your reasoning for the proposal. The system will then append the predicted latency and energy results from your Digital Twin test to your reason. "
                                             f"Explain *how* you are adjusting your proposal based on this feedback, detailing your new strategy for meeting the SLA and avoiding conflicts. **Focus on reasoning and adapting from past experiences to avoid conflicts.**"
                                             f"**IMPORTANT: The 'reason' field MUST be a plain text string, NOT a nested JSON object, dictionary, or any other structured format. Example: 'This proposal balances energy and latency.'**"
                                             f"Ensure any double quotes within this plain text reason string are STRICTLY ESCAPED using a backslash (e.g., `\\\"This is an \\\\\\\"example\\\\\\\" text with escaped quotes.\\\"` or `\\\"Configuration: \\\\\\\"RAN BW 39.2 MHz, Edge CPU 50.0 GHz.\\\\\\\"\"`). Do NOT attempt to embed JSON or dictionary-like structures within the 'reason' string."} # Updated fmax reference
                                ]
                                if sanitized_memory_summary:
                                    self_correction_prompt_parts.append({"text": sanitized_memory_summary})
                                
                                response = self.chat_session.send_message(self_correction_prompt_parts)
                                continue # Continue to the next attempt in the internal loop
                        else:
                            # Digital Twin test passed, update the reason with predicted values
                            parsed_move["parameters"]["reason"] = (
                                f"{parsed_move['parameters'].get('reason', '')}. "
                                f"Predicted Latency: {predicted_latency:.2f} ms, Predicted Energy: {predicted_energy:.2f} W."
                            )
                            # Re-escape double quotes in the reason string
                            parsed_move["parameters"]["reason"] = parsed_move["parameters"]["reason"].replace('"', '\\"')
                            final_negotiation_message = f"PROPOSE_ACTION: {json.dumps(parsed_move['parameters'])}"
                            break # Digital Twin test passed, break out of internal loop
                    else: # No Digital Twin available, proceed with agent's raw proposal (shouldn't happen if DT is always passed)
                        print(f"[{self.name}] Warning: Digital Twin not available. Proceeding without internal validation.")
                        final_negotiation_message = negotiation_message
                        break

                elif parsed_move["intent"] == "ACCEPT_AGREEMENT":
                    accepted_ran_bw = parsed_move["parameters"].get("ran_bandwidth_mhz")
                    accepted_edge_cpu = parsed_move["parameters"].get("edge_cpu_frequency_ghz")

                    if accepted_ran_bw is None or accepted_edge_cpu is None:
                        raise ValueError("ACCEPT_AGREEMENT missing ran_bandwidth_mhz or edge_cpu_frequency_ghz.")

                    if self.digital_twin:
                        self.digital_twin.reset_to_current_state(current_metrics)
                        predicted_metrics = self.digital_twin.simulate_step_for_prediction(
                            proposed_ran_bandwidth_mhz=accepted_ran_bw,
                            proposed_edge_cpu_frequency_ghz=accepted_edge_cpu
                        )
                        predicted_latency = predicted_metrics["predicted_latency_ms"]
                        predicted_energy = predicted_metrics["predicted_energy_watts"]
                        predicted_cpu_conflict = predicted_metrics["predicted_cpu_allocation_conflict_count"]

                        if predicted_latency > SLA_LATENCY_THRESHOLD_MS or predicted_cpu_conflict > current_metrics["cpu_allocation_conflict_count"]:
                            print(f"[{self.name}] Digital Twin test failed for ACCEPTED proposal (BW: {accepted_ran_bw:.1f}, CPU: {accepted_edge_cpu:.1f}). Predicted Latency: {predicted_latency:.2f}ms (SLA: {SLA_LATENCY_THRESHOLD_MS}ms), Predicted CPU Conflicts: {predicted_cpu_conflict}.")
                            
                            # If late iteration and DT failed, try memory-guided fallback for counter-proposal
                            if is_late_iteration_fallback_attempt and memory_insights.get("inferred_successful_config"):
                                inferred_config = memory_insights["inferred_successful_config"]
                                print(f"[{self.name}] Late iteration DT failure for acceptance. Attempting memory-guided counter-proposal: {inferred_config}")

                                # Use inferred config as the new proposal
                                proposed_ran_bw_fallback = inferred_config["ran_bandwidth_mhz"]
                                proposed_edge_cpu_fallback = inferred_config["edge_cpu_frequency_ghz"]

                                # Re-run DT test with fallback proposal
                                predicted_metrics_fallback = self.digital_twin.simulate_step_for_prediction(
                                    proposed_ran_bandwidth_mhz=proposed_ran_bw_fallback,
                                    proposed_edge_cpu_frequency_ghz=proposed_edge_cpu_fallback
                                )
                                predicted_latency_fallback = predicted_metrics_fallback["predicted_latency_ms"]
                                predicted_energy_fallback = predicted_metrics_fallback["predicted_energy_watts"]
                                predicted_cpu_conflict_fallback = predicted_metrics_fallback["predicted_cpu_allocation_conflict_count"]

                                if predicted_latency_fallback <= SLA_LATENCY_THRESHOLD_MS and predicted_cpu_conflict_fallback <= current_metrics["cpu_allocation_conflict_count"]:
                                    # Memory-guided fallback passed DT test, so counter-propose it
                                    parsed_move["intent"] = "PROPOSE_ACTION" # Change intent to PROPOSE_ACTION
                                    parsed_move["parameters"]["ran_bandwidth_mhz"] = proposed_ran_bw_fallback
                                    parsed_move["parameters"]["edge_cpu_frequency_ghz"] = proposed_edge_cpu_fallback
                                    parsed_move["parameters"]["reason"] = (
                                        f"Cannot accept opposing proposal due to DT failure. This is a memory-guided counter-proposal to ensure SLA compliance in late iteration. "
                                        f"Predicted Latency: {predicted_latency_fallback:.2f} ms, Predicted Energy: {predicted_energy_fallback:.2f} W."
                                    )
                                    parsed_move["parameters"]["reason"] = parsed_move["parameters"]["reason"].replace('"', '\\"')
                                    final_negotiation_message = f"PROPOSE_ACTION: {json.dumps(parsed_move['parameters'])}"
                                    break # Break out of internal loop
                                else:
                                    # Memory-guided fallback also failed DT test
                                    print(f"[{self.name}] Memory-guided fallback also failed DT. Declaring NO_AGREEMENT_POSSIBLE.")
                                    final_negotiation_message = "NO_AGREEMENT_POSSIBLE: Failed to find a valid counter-proposal, even with memory-guided fallback, in late iteration."
                                    break # Break out of internal loop
                            else:
                                # Not late iteration or no memory insights for fallback, guide agent to reconsider
                                self_correction_prompt_parts = [
                                    {"text": f"Current network metrics: {json.dumps(current_metrics, indent=2)}\n"},
                                    {"text": f"You are the {self.role}. Your goal is to {self.negotiation_goal}.\n"},
                                    {"text": f"The opposing agent said: '{sanitized_opposing_agent_message}'\n"},
                                    {"text": f"**INTERNAL DIGITAL TWIN TEST FAILED (Attempt {attempt + 1}/{self.max_llm_response_retries}):** Your last `ACCEPT_AGREEMENT` for (RAN BW: {accepted_ran_bw:.1f}, Edge CPU: {accepted_edge_cpu:.1f}) resulted in: "
                                             f"Predicted Latency: {predicted_latency:.2f}ms (SLA: {SLA_LATENCY_THRESHOLD_MS}ms) and Predicted CPU Conflicts: {predicted_cpu_conflict}. "
                                             f"This VIOLATES the SLA or causes CPU conflicts. You MUST NOT accept this proposal as is. "
                                             f"Instead, you must `PROPOSE_ACTION` with adjusted values that meet the SLA and avoid conflicts, or declare `NO_AGREEMENT_POSSIBLE` if no viable compromise exists. "
                                             f"Remember to include the new predicted latency and energy from your DT test in your reason, and escape any double quotes. "
                                             f"Explain *how* you are adjusting your proposal based on this feedback, detailing your new strategy for meeting the SLA and avoiding conflicts. **Focus on reasoning and adapting from past experiences to avoid conflicts.**"
                                             f"**IMPORTANT: The 'reason' field MUST be a plain text string, NOT a nested JSON object, dictionary, or any other structured format. Example: 'This proposal balances energy and latency.'**"
                                             f"Ensure any double quotes within this plain text reason string are STRICTLY ESCAPED using a backslash (e.g., `\\\"This is an \\\\\\\"example\\\\\\\" text with escaped quotes.\\\"` or `\\\"Configuration: \\\\\\\"RAN BW 39.2 MHz, Edge CPU 50.0 GHz.\\\\\\\"\"`). Do NOT attempt to embed JSON or dictionary-like structures within the 'reason' string.`"} # Updated fmax reference
                                ]
                                if sanitized_memory_summary:
                                    self_correction_prompt_parts.append({"text": sanitized_memory_summary})
                                
                                response = self.chat_session.send_message(self_correction_prompt_parts)
                                continue # Continue to the next attempt in the internal loop
                        else:
                            # Digital Twin test passed for acceptance, update the reason with predicted values
                            parsed_move["parameters"]["reason"] = (
                                f"{parsed_move['parameters'].get('reason', '')}. "
                                f"Predicted Latency: {predicted_latency:.2f} ms, Predicted Energy: {predicted_energy:.2f} W."
                            )
                            parsed_move["parameters"]["reason"] = parsed_move["parameters"]["reason"].replace('"', '\\"')
                            final_negotiation_message = f"ACCEPT_AGREEMENT: {json.dumps(parsed_move['parameters'])}"
                            break # Digital Twin test passed, break out of internal loop
                    else:
                        print(f"[{self.name}] Warning: Digital Twin not available. Proceeding without internal validation for ACCEPT_AGREEMENT.")
                        final_negotiation_message = negotiation_message
                        break

                elif parsed_move["intent"] == "NO_AGREEMENT_POSSIBLE":
                    final_negotiation_message = negotiation_message
                    break # No DT test needed for NO_AGREEMENT_POSSIBLE

                elif parsed_move["intent"] == "PARSING_FAILED": # New condition for parsing failures
                    final_negotiation_message = "NO_AGREEMENT_POSSIBLE: An internal error occurred due to unparseable message format."
                    break # Break the loop, as this is a terminal state for this turn

                else: # Invalid format, retry
                    raise ValueError(f"Invalid negotiation format: {negotiation_message}")

            except (generation_types.StopCandidateException, ValueError, Exception) as e:
                print(f"[{self.name}] Caught exception during text response parsing or DT testing on attempt {attempt+1}: {e}. Declaring NO_AGREEMENT_POSSIBLE for this negotiation turn.")
                # If an error occurs, force NO_AGREEMENT_POSSIBLE and break the retry loop
                final_negotiation_message = "NO_AGREEMENT_POSSIBLE: An internal error occurred during negotiation move generation or validation."
                # Set parsed_move to PARSING_FAILED if an exception occurs during parsing/DT test
                parsed_move = {"intent": "PARSING_FAILED", "parameters": {"reason": f"Exception during parsing/DT test: {e}"}}
                break # Exit the retry loop immediately

        # The 'else' block for the 'for attempt' loop is now redundant as the 'break' handles all failure cases.
        # The 'final_negotiation_message' will be set by the 'break' or if a successful path is taken.
        if final_negotiation_message is None:
             # This case should ideally not be reached if all paths lead to 'break' or successful message.
             # As a fallback, if somehow no message was set, force NO_AGREEMENT_POSSIBLE.
            print(f"[{self.name}] No valid negotiation message was set after all attempts. Forcing NO_AGREEMENT_POSSIBLE.")
            final_negotiation_message = "NO_AGREEMENT_POSSIBLE: Failed to generate a valid negotiation message after multiple internal Digital Twin test attempts."


        if parsed_move["intent"] == "PROPOSE_ACTION":
            self.last_proposed_config = parsed_move["parameters"]

        return final_negotiation_message




