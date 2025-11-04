# LLM-powered Agentic 6G Cross-Domain Negotiation

This project simulates a negotiation between two LLM-powered agents, a RAN (Energy Saving) Agent and an Edge (Latency Assurance) Agent, to reconcile their conflicting goals. The simulation evaluates different strategies, including using a collective memory with and without debiasing mechanisms.

<img width="661" height="505" alt="use_case" src="https://github.com/user-attachments/assets/78c104a6-21ec-4d3e-87be-c90f93d34031" />

## [Citation] 
If you use this code or any (modified) part of it, please cite it as: 
H. Chergui, F. Rezazadeh, M. Debbah, and C. Verikoukis, “A Tutorial on Cognitive Biases in Agentic AI-Driven 6G Autonomous Networks,” arXiv preprint arXiv:2510.19973, 2025. [Online]. Available: https://arxiv.org/abs/2510.19973

## Project Structure

- `main.py`: The main entry point to run the simulation and generate plots.
- `config.py`: Contains global simulation parameters and constants.
- `network_simulator.py`: Defines the core `NetworkSimulator` class, which models the network environment.
- `e2_api_tool.py`: Provides the `E2APISimulator`, an interface for agents to interact with the network simulator.
- `digital_twin.py`: Contains the `DigitalTwin` class, a model used by agents for internal testing of proposals.
- `collective_memory.py`: Implements the `CollectiveMemory` class for storing and retrieving negotiation strategies.
- `llm_agent.py`: Defines the base `LLMAgent` class for the negotiating agents.
- `agents.py`: Contains the specialized `RanAgent` and `EdgeAgent` classes.
- `a2a.py`: Implements the `A2ANegotiationManager` to orchestrate the negotiation process.
- `negotiation_parser.py`: Implements the parsing of negotiation messages.
- `requirements.txt`: Lists the necessary Python packages for this project.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set API Key:**
    Make sure your `GOOGLE_API_KEY` is set as an environment variable.

3.  **Run the Simulation:**
    ```bash
    python main.py
    ```
    The first run will execute the full simulation and save the results to `simulation_results.pkl`. Subsequent runs will load from this file to generate plots without re-running the simulation. To force a new simulation, delete `simulation_results.pkl`.
