import pytest

from simulation.machine_engine import (
    AIR_RANGE, PROCESS_RANGE, RPM_RANGE, TORQUE_RANGE, WEAR_MAX, MachineEngine,
)

KEYS = {"Type", "Air temperature", "Process temperature",
        "Rotational speed", "Torque", "Tool wear"}


def test_readings_have_model_format():
    assert set(MachineEngine("T", "L").readings()) == KEYS


def test_invalid_type_rejected():
    with pytest.raises(ValueError):
        MachineEngine("T", "X")


def test_deterministic_for_same_seed():
    a, b = MachineEngine("A", "M", seed=7), MachineEngine("B", "M", seed=7)
    assert [a.step() for _ in range(25)] == [b.step() for _ in range(25)]


def test_extreme_stress_stays_in_training_range():
    m = MachineEngine("T", "H", seed=3)
    m.set_conditions(10, 30, 5)
    for _ in range(400):
        r = m.step()
        assert AIR_RANGE[0] <= r["Air temperature"] <= AIR_RANGE[1]
        assert PROCESS_RANGE[0] <= r["Process temperature"] <= PROCESS_RANGE[1]
        assert RPM_RANGE[0] <= r["Rotational speed"] <= RPM_RANGE[1]
        assert TORQUE_RANGE[0] <= r["Torque"] <= TORQUE_RANGE[1]
        assert r["Tool wear"] <= WEAR_MAX


def test_conditions_are_clamped():
    m = MachineEngine("T")
    m.set_conditions(999, -999, 999)
    assert (m.temperature_stress, m.load_stress, m.wear_rate) == (10, -10, 5)


def test_tool_wear_never_decreases_and_replace_tool_resets_it():
    m = MachineEngine("T")
    wear = [m.step()["Tool wear"] for _ in range(30)]
    assert wear == sorted(wear)
    m.replace_tool()
    assert m.readings()["Tool wear"] == 0.0


def test_reset_restores_baseline_and_sequence():
    m = MachineEngine("T", "L", seed=5)
    first = [m.step() for _ in range(5)]
    m.set_conditions(5, 20, 3)
    m.reset()
    assert m.steps_taken == 0 and m.wear_rate == 1.0
    assert [m.step() for _ in range(5)] == first
