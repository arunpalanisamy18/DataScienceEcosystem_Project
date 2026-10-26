from explain import failure_mode_warnings
from inference import predict_batch
from simulation.factory_controller import FactoryController


class FactoryMonitor:
    """Combine the factory sensor simulation with trained ML predictions."""

    def __init__(self, mode="balanced"):
        self.factory = FactoryController()
        self.mode = mode

    def _evaluate(self, readings_by_machine):
        ids = list(readings_by_machine)
        predictions = predict_batch([readings_by_machine[i] for i in ids], self.mode)
        return {
            mid: {
                "sensor_readings": readings_by_machine[mid],
                "warnings": failure_mode_warnings(readings_by_machine[mid]),
                **pred,
            }
            for mid, pred in zip(ids, predictions)
        }

    def monitor_all(self, advance=True):
        """Read every machine (optionally advancing one step) and predict risk."""
        readings = {}
        for mid in self.factory.list_machines():
            machine = self.factory.get_machine(mid)
            readings[mid] = machine.step() if advance else machine.readings()
        return self._evaluate(readings)

    def monitor_one(self, machine_id, advance=False):
        machine = self.factory.get_machine(machine_id)
        readings = machine.step() if advance else machine.readings()
        return self._evaluate({machine_id: readings})[machine_id]

    def run_steps(self, n_steps, machine_id=None):
        """Advance one machine (or all) n_steps times.

        Returns (history_rows, latest_results). Each history row holds the
        readings and prediction for one machine at one simulation step.
        """
        ids = [machine_id] if machine_id else self.factory.list_machines()
        history = []
        for _ in range(int(n_steps)):
            readings = {mid: self.factory.get_machine(mid).step() for mid in ids}
            results = self._evaluate(readings)
            for mid, result in results.items():
                history.append({
                    "Step": self.factory.get_machine(mid).steps_taken,
                    "Machine": mid,
                    **result["sensor_readings"],
                    "Failure Probability (%)": result["machine_failure_probability"] * 100,
                    "Status": result["status"],
                })
        return history, self.monitor_all(advance=False)

    def set_conditions(self, machine_id, temperature_stress=None, load_stress=None, wear_rate=None):
        self.factory.set_conditions(machine_id, temperature_stress, load_stress, wear_rate)

    def replace_tool(self, machine_id):
        self.factory.replace_tool(machine_id)

    def reset_machine(self, machine_id):
        self.factory.reset_machine(machine_id)
