"""One-click operating scenarios for demonstrations."""

SCENARIOS = {
    "Normal operation": {"temperature_stress": 0, "load_stress": 0, "wear_rate": 1.0},
    "Worn tool under load": {"temperature_stress": 0, "load_stress": 10, "wear_rate": 5.0},
    "Heavy overload": {"temperature_stress": 0, "load_stress": 30, "wear_rate": 1.0},
    "Overheating": {"temperature_stress": 10, "load_stress": 10, "wear_rate": 1.0},
    "Light duty": {"temperature_stress": -3, "load_stress": -8, "wear_rate": 0.5},
}
