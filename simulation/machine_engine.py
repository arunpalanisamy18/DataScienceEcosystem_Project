"""Single-machine sensor simulator.

Readings drift towards operating targets that depend on three controls and
are clamped to the range seen in the training data (so the model is never
asked to extrapolate to physically impossible values):

* temperature_stress  (-5..10)  raises air temperature and narrows the
                                process-air temperature gap
* load_stress         (-10..30) raises torque and lowers rotational speed
* wear_rate           (0..5)    how fast tool wear accumulates
"""
import random
from dataclasses import dataclass, field

# Ranges observed in the AI4I 2020 training data
AIR_RANGE = (295.3, 304.5)
PROCESS_RANGE = (305.7, 313.8)
RPM_RANGE = (1168.0, 2886.0)
TORQUE_RANGE = (3.8, 76.6)
WEAR_MAX = 253.0

BASE_AIR, BASE_GAP, BASE_RPM, BASE_TORQUE, BASE_WEAR = 300.0, 10.0, 1500.0, 40.0, 100.0
SMOOTHING = 0.25  # how quickly a reading moves towards its target each step


def _clamp(value, low, high):
    return max(low, min(high, value))


@dataclass
class MachineEngine:
    machine_id: str
    machine_type: str = "M"
    seed: int = 42

    air_temperature: float = BASE_AIR
    process_temperature: float = BASE_AIR + BASE_GAP
    rotational_speed: float = BASE_RPM
    torque: float = BASE_TORQUE
    tool_wear: float = BASE_WEAR

    temperature_stress: float = 0.0
    load_stress: float = 0.0
    wear_rate: float = 1.0
    steps_taken: int = 0

    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self):
        if self.machine_type not in {"L", "M", "H"}:
            raise ValueError("machine_type must be L, M, or H")
        self.rng = random.Random(self.seed)

    def set_conditions(self, temperature_stress=None, load_stress=None, wear_rate=None):
        """Set simulation controls (values are clamped to their valid range)."""
        if temperature_stress is not None:
            self.temperature_stress = float(_clamp(temperature_stress, -5, 10))
        if load_stress is not None:
            self.load_stress = float(_clamp(load_stress, -10, 30))
        if wear_rate is not None:
            self.wear_rate = float(_clamp(wear_rate, 0, 5))

    def step(self):
        """Generate one new simulated sensor reading."""
        g = self.rng.gauss

        air_target = BASE_AIR + 0.45 * self.temperature_stress
        self.air_temperature = _clamp(
            self.air_temperature + SMOOTHING * (air_target - self.air_temperature) + g(0, 0.15),
            *AIR_RANGE)

        gap_target = BASE_GAP - 0.12 * self.temperature_stress
        gap = self.process_temperature - self.air_temperature
        gap += SMOOTHING * (gap_target - gap) + g(0, 0.10)
        self.process_temperature = _clamp(self.air_temperature + gap, *PROCESS_RANGE)

        torque_target = BASE_TORQUE + 1.1 * self.load_stress
        self.torque = _clamp(
            self.torque + SMOOTHING * (torque_target - self.torque) + g(0, 1.0),
            *TORQUE_RANGE)

        rpm_target = BASE_RPM - 11.0 * self.load_stress
        self.rotational_speed = _clamp(
            self.rotational_speed + SMOOTHING * (rpm_target - self.rotational_speed) + g(0, 15),
            *RPM_RANGE)

        self.tool_wear = min(
            WEAR_MAX, self.tool_wear + self.wear_rate * self.rng.uniform(0.2, 1.0))

        self.steps_taken += 1
        return self.readings()

    def readings(self):
        """Return readings in the exact format expected by the ML model."""
        return {
            "Type": self.machine_type,
            "Air temperature": round(self.air_temperature, 2),
            "Process temperature": round(self.process_temperature, 2),
            "Rotational speed": round(self.rotational_speed, 2),
            "Torque": round(self.torque, 2),
            "Tool wear": round(self.tool_wear, 2),
        }

    def replace_tool(self):
        """Maintenance action: fit a new tool (tool wear back to zero)."""
        self.tool_wear = 0.0

    def reset(self):
        """Reset the machine to its baseline operating conditions."""
        self.air_temperature = BASE_AIR
        self.process_temperature = BASE_AIR + BASE_GAP
        self.rotational_speed = BASE_RPM
        self.torque = BASE_TORQUE
        self.tool_wear = BASE_WEAR
        self.temperature_stress = 0.0
        self.load_stress = 0.0
        self.wear_rate = 1.0
        self.steps_taken = 0
        self.rng = random.Random(self.seed)
