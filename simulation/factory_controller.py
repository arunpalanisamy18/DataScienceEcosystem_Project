from simulation.machine_engine import MachineEngine


class FactoryController:
    """Manage multiple independent simulated factory machines."""

    def __init__(self):
        self.machines = {
            "MCH-001": MachineEngine("MCH-001", "L", seed=1),
            "MCH-002": MachineEngine("MCH-002", "M", seed=2),
            "MCH-003": MachineEngine("MCH-003", "H", seed=3),
            "MCH-004": MachineEngine("MCH-004", "L", seed=4),
            "MCH-005": MachineEngine("MCH-005", "M", seed=5),
        }

    def list_machines(self):
        return list(self.machines.keys())

    def get_machine(self, machine_id):
        if machine_id not in self.machines:
            raise ValueError(f"Unknown machine: {machine_id}")
        return self.machines[machine_id]

    def step_all(self):
        return {mid: m.step() for mid, m in self.machines.items()}

    def set_conditions(self, machine_id, temperature_stress=None, load_stress=None, wear_rate=None):
        self.get_machine(machine_id).set_conditions(
            temperature_stress=temperature_stress,
            load_stress=load_stress,
            wear_rate=wear_rate,
        )

    def replace_tool(self, machine_id):
        self.get_machine(machine_id).replace_tool()

    def reset_machine(self, machine_id):
        self.get_machine(machine_id).reset()

    def reset_factory(self):
        for machine in self.machines.values():
            machine.reset()
