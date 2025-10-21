from typing import Dict, Any
from network_simulator import NetworkSimulator

class E2APISimulator:
    """
    Acts as the tool interface for agents to interact with the NetworkSimulator.
    It exposes methods to get metrics and enforce actions.
    """
    def __init__(self, simulator_instance: NetworkSimulator):
        self.simulator = simulator_instance
        self.MAX_RAN_BW_MHZ = 40.0
        self.MIN_RAN_BW_MHZ = 5.0
        self.MAX_EDGE_CPU_GHZ = simulator_instance.fmax

    def get_metrics(self) -> Dict[str, Any]:
        """Retrieves the current metrics from the network simulator."""
        return self.simulator.get_metrics()

    def enforce_actions(self, ran_bandwidth_mhz: float, edge_cpu_frequency_ghz: float) -> Dict[str, Any]:
        """Enforces the actions proposed by an agent on the network simulator."""
        if not (self.MIN_RAN_BW_MHZ <= ran_bandwidth_mhz <= self.MAX_RAN_BW_MHZ):
            return {"status": "error", "message": f"Invalid RAN bandwidth value. Must be between {self.MIN_RAN_BW_MHZ} and {self.MAX_RAN_BW_MHZ} MHz."}
        if not (0 <= edge_cpu_frequency_ghz <= self.MAX_EDGE_CPU_GHZ):
            return {"status": "error", "message": f"Invalid Edge CPU frequency. Must be between 0 and {self.MAX_EDGE_CPU_GHZ} GHz."}

        self.simulator.set_actions(ran_bandwidth_mhz, edge_cpu_frequency_ghz)
        updated_metrics = self.simulator.get_metrics()
        return {
            "status": "success",
            "message": "Actions enforced successfully.",
            "enforced_actions": {
                "ran_bandwidth_mhz": ran_bandwidth_mhz,
                "edge_cpu_frequency_ghz": edge_cpu_frequency_ghz
            },
            "current_metrics": updated_metrics
        }


# 1. Create a single instance of the underlying network simulator.
#    This instance will be shared across the application.
_underlying_simulator = NetworkSimulator()

# 2. Create a single instance of the E2APISimulator, wrapping the underlying simulator.
_api_simulator_instance = E2APISimulator(simulator_instance=_underlying_simulator)

# 3. Expose the simulator instance at the module level.
#    This allows main.py to call e2_api_tool.simulator.reset_simulation().
simulator = _api_simulator_instance.simulator

# 4. Expose the API methods as module-level functions for the LLM agent tools.
#    This allows the agent to call e2_api_tool.get_metrics() directly.
def get_metrics() -> Dict[str, Any]:
    """Module-level wrapper for the get_metrics tool."""
    return _api_simulator_instance.get_metrics()

def enforce_actions(ran_bandwidth_mhz: float, edge_cpu_frequency_ghz: float) -> Dict[str, Any]:
    """Module-level wrapper for the enforce_actions tool."""
    return _api_simulator_instance.enforce_actions(ran_bandwidth_mhz, edge_cpu_frequency_ghz)

# 5. Expose the configuration constants at the module level for global access.
MAX_RAN_BW_MHZ = _api_simulator_instance.MAX_RAN_BW_MHZ
MIN_RAN_BW_MHZ = _api_simulator_instance.MIN_RAN_BW_MHZ
MAX_EDGE_CPU_GHZ = _api_simulator_instance.MAX_EDGE_CPU_GHZ



