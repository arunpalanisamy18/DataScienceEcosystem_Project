import pytest

from simulation.factory_controller import FactoryController


def test_five_machines_of_mixed_types():
    f = FactoryController()
    assert f.list_machines() == [f"MCH-00{i}" for i in range(1, 6)]
    assert {f.get_machine(m).machine_type for m in f.list_machines()} == {"L", "M", "H"}


def test_unknown_machine_raises():
    with pytest.raises(ValueError):
        FactoryController().get_machine("NOPE")


def test_conditions_and_reset_affect_one_machine_only():
    f = FactoryController()
    f.set_conditions("MCH-001", load_stress=20)
    assert f.get_machine("MCH-001").load_stress == 20
    assert f.get_machine("MCH-002").load_stress == 0
    f.reset_machine("MCH-001")
    assert f.get_machine("MCH-001").load_stress == 0


def test_step_all_and_reset_factory():
    f = FactoryController()
    assert len(f.step_all()) == 5
    f.reset_factory()
    assert all(m.steps_taken == 0 for m in f.machines.values())
