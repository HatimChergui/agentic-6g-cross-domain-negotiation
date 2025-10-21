# --- Global simulation parameters (constants that don't change per trial) ---
NUM_TRIALS = 10
# Number of time steps in a traffic/capacity realization
NUM_TIME_STEPS = 8

# Service Level Agreement (SLA) threshold for latency
SLA_LATENCY_THRESHOLD_MS = 10.0

# RAN Power Parameters
POWER_PER_20MHZ_CARRIER_W = 10.0
REFERENCE_BANDWIDTH_MHZ = 20.0

# Define the reference energy for savings calculation.
# This represents a baseline or maximum energy consumption scenario.
REFERENCE_ENERGY_FOR_SAVINGS_W = 20.0
