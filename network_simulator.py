import numpy as np
from typing import Dict, Any, Optional
from config import NUM_TIME_STEPS, POWER_PER_20MHZ_CARRIER_W, REFERENCE_BANDWIDTH_MHZ

_TRAFFIC_BASE_SLICE = 50_000_000  # 50 Mbps
_TRAFFIC_VARIATION_SLICE = 30_000_000  # 30 Mbps
_CAPACITY_BASE_SLICE = 65_000_000  # 65 Mbps
_CAPACITY_VARIATION_SLICE = 15_000_000  # 15 Mbps

class NetworkSimulator:

    def __init__(self):
        self.id_traffic = 0
        self.tau = 0.01  # Simulation time step in seconds
        self.fmax = 45.0  # Max CPU frequency in GHz (assuming multi-core processing)
        self.U = 0.0017  # CPU processing capacity factor

        self.cqueue = 0.0  # Computation queue size in bits
        self.rqueue = 0.0  # Radio queue size in bits
        self.Q = 0.0  # Total current queue size
        self.latency = 0.0  # Current latency in seconds
        self.transmission_rate = 0.0  # Current transmission rate in bps
        self.energy_consumption_watts = 0.0  # Current energy consumption in Watts

        self.cpu_frequency_allocated = 25.0  # Default starting CPU frequency
        self.bandwidth_allocated = 20.0  # Default starting bandwidth

        self.cpu_allocation_conflict_count_internal = 0

        # Expose module-level constants as instance attributes for backward compatibility
        # This allows external modules to access them via the simulator instance.
        self._TRAFFIC_BASE_SLICE = _TRAFFIC_BASE_SLICE
        self._TRAFFIC_VARIATION_SLICE = _TRAFFIC_VARIATION_SLICE
        self._CAPACITY_BASE_SLICE = _CAPACITY_BASE_SLICE
        self._CAPACITY_VARIATION_SLICE = _CAPACITY_VARIATION_SLICE


        # Initialize traffic and capacity using the module-level constants
        self.traffic = self._generate_random_traffic()
        self.capacity = self._generate_random_capacity()
        self.arrival_rate = np.mean(self.traffic[:, 0])

        self.t = 0  # Current time step

        # Spectral efficiency parameters
        self.min_spectral_efficiency = 6.0
        self.max_spectral_efficiency = 8.0
        self.current_spectral_efficiency = (self.min_spectral_efficiency + self.max_spectral_efficiency) / 2

        self._update_current_state_metrics_only()

    def _generate_random_traffic(self):
        """Generates a new random traffic pattern."""
        # Use the module-level constant directly
        return (_TRAFFIC_BASE_SLICE + _TRAFFIC_VARIATION_SLICE * np.random.rand(NUM_TIME_STEPS, 1)).astype(int)

    def _generate_random_capacity(self):
        """Generates a new random capacity pattern."""
        # Use the module-level constant directly
        return (_CAPACITY_BASE_SLICE + _CAPACITY_VARIATION_SLICE * np.random.rand(NUM_TIME_STEPS, 1)).astype(int)

    def _calculate_transmission_rate(self, bandwidth_mhz: float) -> float:
        """Calculates the transmission rate based on bandwidth and spectral efficiency."""
        return bandwidth_mhz * (10**6) * self.current_spectral_efficiency

    def _calculate_energy_consumption(self, bandwidth_mhz: float, cpu_frequency_ghz: float) -> float:
        """Calculates the RAN power consumption."""
        ran_power = 0.0
        if bandwidth_mhz > 0:
            num_reference_carriers = bandwidth_mhz / REFERENCE_BANDWIDTH_MHZ
            ran_power += (num_reference_carriers * POWER_PER_20MHZ_CARRIER_W)
        return ran_power

    def _update_current_state_metrics_only(self):
        """Updates all internal metrics for one time step based on the current state."""
        current_traffic_at_id = self.traffic[self.id_traffic, 0]
        arriving_data_in_tau = self.tau * current_traffic_at_id

        U_proc_capacity_in_tau = self.tau * self.U * self.cpu_frequency_allocated * (10**9)

        self.cqueue += arriving_data_in_tau
        processed_data_in_tau = min(self.cqueue, U_proc_capacity_in_tau)
        self.cqueue = max(0, self.cqueue - processed_data_in_tau)

        self.rqueue += processed_data_in_tau
        radio_capacity_in_tau = self.tau * self._calculate_transmission_rate(self.bandwidth_allocated)
        transmitted_data_in_tau = min(self.rqueue, radio_capacity_in_tau)
        self.rqueue = max(0, self.rqueue - transmitted_data_in_tau)

        self.Q = self.cqueue + self.rqueue

        if self.Q > 0 and transmitted_data_in_tau > 0:
            self.latency = self.Q / (transmitted_data_in_tau / self.tau)
        elif self.Q == 0:
            self.latency = 0.0
        else:
            self.latency = 1.0  # Stall represented by 1 second latency

        self.transmission_rate = transmitted_data_in_tau / self.tau
        self.energy_consumption_watts = self._calculate_energy_consumption(self.bandwidth_allocated, self.cpu_frequency_allocated)

        if self.cpu_frequency_allocated > self.fmax:
            self.cpu_allocation_conflict_count_internal += 1

    def set_actions(self, ran_bandwidth_mhz: float, edge_cpu_frequency_ghz: float):
        """Sets the actions for the current time step and advances the simulation."""
        self.bandwidth_allocated = ran_bandwidth_mhz
        self.cpu_frequency_allocated = edge_cpu_frequency_ghz

        self.current_spectral_efficiency = self.min_spectral_efficiency + \
            np.random.rand() * (self.max_spectral_efficiency - self.min_spectral_efficiency)

        self._update_current_state_metrics_only()

        self.t += 1
        self.id_traffic = (self.id_traffic + 1) % NUM_TIME_STEPS

    def get_metrics(self) -> Dict[str, Any]:
        """Returns the current state of the simulator."""
        return {
            "latency_ms": self.latency * 1000,
            "transmission_rate_bps": self.transmission_rate,
            "cqueue_bits": self.cqueue,
            "rqueue_bits": self.rqueue,
            "energy_consumption_watts": self.energy_consumption_watts,
            "cpu_frequency_ghz_allocated": self.cpu_frequency_allocated,
            "bandwidth_mhz_allocated": self.bandwidth_allocated,
            "cpu_allocation_conflict_count": self.cpu_allocation_conflict_count_internal,
            "current_time_step": self.t,
            "current_traffic_arrival_rate_bps": float(self.traffic[self.id_traffic, 0]),
            "average_traffic_arrival_rate_bps": self.arrival_rate,
            "current_spectral_efficiency_bits_per_hz_per_s": self.current_spectral_efficiency
        }

    def reset_simulation(self, start_time_step: Optional[int] = None):
        """Resets the simulation to an initial state for a new trial."""
        self.traffic = self._generate_random_traffic()
        self.capacity = self._generate_random_capacity()
        self.arrival_rate = np.mean(self.traffic[:, 0])

        if start_time_step is None:
            self.id_traffic = 0
            self.t = 0
        else:
            if not (0 <= start_time_step < NUM_TIME_STEPS):
                raise ValueError(f"start_time_step must be between 0 and {NUM_TIME_STEPS - 1}")
            self.id_traffic = start_time_step
            self.t = start_time_step

        self.cqueue = 0.0
        self.rqueue = 0.0
        self.Q = 0.0
        self.latency = 0.0
        self.transmission_rate = 0.0
        self.energy_consumption_watts = 0.0
        self.cpu_frequency_allocated = 25.0
        self.bandwidth_allocated = 20.0
        self.cpu_allocation_conflict_count_internal = 0
        self.current_spectral_efficiency = (self.min_spectral_efficiency + self.max_spectral_efficiency) / 2
        self._update_current_state_metrics_only()


