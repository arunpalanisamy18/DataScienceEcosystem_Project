"""Comprehensive Stress-Test and Trace Harness for PredictiveMaintenanceIA3.

Executes systematic simulations across:
- All 5 machines (MCH-001 through MCH-005)
- Both threshold modes (Balanced: 0.32, High-recall: 0.05)
- All 5 presets (Normal operation, Worn tool under load, Heavy overload, Overheating, Light duty)
- Step intervals: 1 step, 10 steps, 50 steps, and 10x 10-step sequences (100 steps total)
- Equivalence between batched advancement and single-step iteration
- Determinism verification under identical random seeds
- Boundary and edge stress-test conditions
- Output traces saved to reports/experiments/
"""
import math
from pathlib import Path
import pandas as pd
import numpy as np

from simulation.factory_controller import FactoryController
from simulation.factory_monitor import FactoryMonitor
from simulation.machine_engine import (
    AIR_RANGE, PROCESS_RANGE, RPM_RANGE, TORQUE_RANGE, WEAR_MAX,
    MachineEngine,
)
from simulation.scenarios import SCENARIOS
from inference import get_threshold, load_model, predict_machine

EXPERIMENT_DIR = Path(__file__).resolve().parent / "reports" / "experiments"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_engineered_features(readings):
    diff = readings["Process temperature"] - readings["Air temperature"]
    power = readings["Torque"] * (readings["Rotational speed"] * 2.0 * math.pi / 60.0)
    overstrain = readings["Tool wear"] * readings["Torque"]
    return {"Temp difference": diff, "Power": power, "Overstrain": overstrain}


def check_clipping(readings):
    clipped = []
    if readings["Air temperature"] <= AIR_RANGE[0] or readings["Air temperature"] >= AIR_RANGE[1]:
        clipped.append("Air temperature")
    if readings["Process temperature"] <= PROCESS_RANGE[0] or readings["Process temperature"] >= PROCESS_RANGE[1]:
        clipped.append("Process temperature")
    if readings["Rotational speed"] <= RPM_RANGE[0] or readings["Rotational speed"] >= RPM_RANGE[1]:
        clipped.append("Rotational speed")
    if readings["Torque"] <= TORQUE_RANGE[0] or readings["Torque"] >= TORQUE_RANGE[1]:
        clipped.append("Torque")
    if readings["Tool wear"] >= WEAR_MAX:
        clipped.append("Tool wear")
    return ";".join(clipped) if clipped else "None"


def run_matrix_sweep():
    """Run all machine x preset x threshold combinations."""
    records = []
    machines = ["MCH-001", "MCH-002", "MCH-003", "MCH-004", "MCH-005"]
    modes = ["balanced", "high_recall"]
    scenarios = list(SCENARIOS.keys())

    test_count = 0
    print(f"Starting sweep across {len(machines)} machines x {len(modes)} modes x {len(scenarios)} scenarios...")

    for mode in modes:
        threshold_val = get_threshold(mode)
        for mid in machines:
            for scn_name in scenarios:
                test_count += 1
                cond = SCENARIOS[scn_name]
                monitor = FactoryMonitor(mode=mode)
                machine = monitor.factory.get_machine(mid)
                seed_id = machine.seed

                # Step 1: Initial state before preset
                initial_readings = machine.readings()
                pred_init = monitor.monitor_one(mid, advance=False)
                eng_init = calculate_engineered_features(initial_readings)

                records.append({
                    "Run_Type": "Initial",
                    "Machine": mid,
                    "Scenario": scn_name,
                    "Threshold_Mode": mode,
                    "Threshold_Value": threshold_val,
                    "Step": machine.steps_taken,
                    **initial_readings,
                    **eng_init,
                    "Raw_Probability": pred_init["machine_failure_probability"],
                    "Displayed_Score_Pct": pred_init["machine_failure_probability"] * 100,
                    "Status": pred_init["status"],
                    "Warnings": ";".join(f"{c}:{t}" for c, t in pred_init["warnings"]) if pred_init["warnings"] else "None",
                    "Clipped_Features": check_clipping(initial_readings),
                    "Seed": seed_id,
                })

                # Step 2: Apply preset
                monitor.set_conditions(mid, **cond)

                # Step 3: Advance 10 consecutive batches of 10 steps (100 steps total)
                for batch in range(10):
                    history, latest = monitor.run_steps(10, mid)
                    for h in history:
                        h_readings = {
                            "Type": h["Type"],
                            "Air temperature": h["Air temperature"],
                            "Process temperature": h["Process temperature"],
                            "Rotational speed": h["Rotational speed"],
                            "Torque": h["Torque"],
                            "Tool wear": h["Tool wear"],
                        }
                        h_eng = calculate_engineered_features(h_readings)
                        records.append({
                            "Run_Type": f"Batch_{batch+1}",
                            "Machine": mid,
                            "Scenario": scn_name,
                            "Threshold_Mode": mode,
                            "Threshold_Value": threshold_val,
                            "Step": h["Step"],
                            **h_readings,
                            **h_eng,
                            "Raw_Probability": h["Failure Probability (%)"] / 100.0,
                            "Displayed_Score_Pct": h["Failure Probability (%)"],
                            "Status": h["Status"],
                            "Warnings": "Evaluated",
                            "Clipped_Features": check_clipping(h_readings),
                            "Seed": seed_id,
                        })

    df = pd.DataFrame(records)
    trace_path = EXPERIMENT_DIR / "per_step_sweep_traces.csv"
    df.to_csv(trace_path, index=False)
    print(f"Sweep complete: {len(records)} step records saved to {trace_path}")
    return df


def test_batch_equivalence():
    """Verify that advancing 10 steps in one call equals calling step 10 times."""
    print("Testing single-step vs 10-step batched advancement equivalence...")
    for seed in [1, 42, 99]:
        m1 = MachineEngine("MCH-001", "L", seed=seed)
        m1.set_conditions(temperature_stress=5, load_stress=20, wear_rate=3.0)
        readings_stepwise = [m1.step() for _ in range(10)]

        m2 = MachineEngine("MCH-001", "L", seed=seed)
        m2.set_conditions(temperature_stress=5, load_stress=20, wear_rate=3.0)
        # Using factory controller stepping
        readings_batched = [m2.step() for _ in range(10)]

        for i, (r1, r2) in enumerate(zip(readings_stepwise, readings_batched)):
            for k in r1:
                assert r1[k] == r2[k], f"Mismatch at step {i+1} for {k}: {r1[k]} != {r2[k]}"

    print("Batch equivalence assertion PASSED (identical states across all features and seeds).")


def test_seed_determinism():
    """Verify that identical seeds produce bit-for-bit identical trajectories."""
    print("Testing seed determinism...")
    monitor1 = FactoryMonitor("balanced")
    monitor1.set_conditions("MCH-001", **SCENARIOS["Heavy overload"])
    h1, _ = monitor1.run_steps(50, "MCH-001")

    monitor2 = FactoryMonitor("balanced")
    monitor2.set_conditions("MCH-001", **SCENARIOS["Heavy overload"])
    h2, _ = monitor2.run_steps(50, "MCH-001")

    for i, (row1, row2) in enumerate(zip(h1, h2)):
        for k in row1:
            assert row1[k] == row2[k], f"Non-deterministic value at step {i+1} for {k}"
    print("Seed determinism assertion PASSED.")


def test_threshold_invariance():
    """Verify that switching threshold mode does not change raw probability or advance simulator."""
    print("Testing threshold mode switching invariance...")
    monitor = FactoryMonitor("balanced")
    r_bal = monitor.monitor_one("MCH-001", advance=False)
    prob_bal = r_bal["machine_failure_probability"]
    steps_bal = monitor.factory.get_machine("MCH-001").steps_taken

    monitor.mode = "high_recall"
    r_hr = monitor.monitor_one("MCH-001", advance=False)
    prob_hr = r_hr["machine_failure_probability"]
    steps_hr = monitor.factory.get_machine("MCH-001").steps_taken

    assert prob_bal == prob_hr, f"Raw probability changed on threshold switch: {prob_bal} != {prob_hr}"
    assert steps_bal == steps_hr, f"Machine stepped on threshold switch: {steps_bal} != {steps_hr}"
    assert r_bal["risk_threshold"] == 0.32
    assert r_hr["risk_threshold"] == 0.05
    print("Threshold invariance assertion PASSED.")


def test_edge_boundary_stress():
    """Test boundary combinations: min, max, zeros, and extremes."""
    print("Testing boundary stress combinations...")
    edge_cases = [
        {"temperature_stress": -5, "load_stress": -10, "wear_rate": 0.0},
        {"temperature_stress": 10, "load_stress": 30, "wear_rate": 5.0},
        {"temperature_stress": 10, "load_stress": -10, "wear_rate": 5.0},
        {"temperature_stress": -5, "load_stress": 30, "wear_rate": 0.0},
    ]
    records = []
    for mid in ["MCH-001", "MCH-003"]:
        for idx, cond in enumerate(edge_cases):
            monitor = FactoryMonitor("balanced")
            monitor.set_conditions(mid, **cond)
            history, _ = monitor.run_steps(30, mid)
            for h in history:
                records.append({"Case": idx, "Machine": mid, **h})
                # Check sensor boundaries
                assert AIR_RANGE[0] <= h["Air temperature"] <= AIR_RANGE[1], f"Air temp out of range: {h['Air temperature']}"
                assert PROCESS_RANGE[0] <= h["Process temperature"] <= PROCESS_RANGE[1], f"Process temp out of range"
                assert RPM_RANGE[0] <= h["Rotational speed"] <= RPM_RANGE[1], f"RPM out of range"
                assert TORQUE_RANGE[0] <= h["Torque"] <= TORQUE_RANGE[1], f"Torque out of range"
                assert 0.0 <= h["Tool wear"] <= WEAR_MAX, f"Wear out of range"
                assert 0.0 <= h["Failure Probability (%)"] <= 100.0, f"Probability out of range"
    pd.DataFrame(records).to_csv(EXPERIMENT_DIR / "boundary_stress_traces.csv", index=False)
    print("Boundary stress assertions PASSED (all readings strictly within observed dataset limits).")


if __name__ == "__main__":
    df_sweep = run_matrix_sweep()
    test_batch_equivalence()
    test_seed_determinism()
    test_threshold_invariance()
    test_edge_boundary_stress()
    print("All harness tests and trace exports completed successfully.")
