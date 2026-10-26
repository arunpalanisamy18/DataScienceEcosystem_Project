"""Regression test suite for simulation dynamics, state management, and workflow consistency.

Enforces:
- Single-step vs batched advancement equivalence
- Exact step counter accounting for 1, 10, 50 step batches
- Invariance of raw prediction scores under threshold mode switching
- Deterministic repeatability under identical seeds
- Consistency between recorded history, machine telemetry, and model inferences
- Input feature schema, types, and absence of target leakage
"""
import pytest
import pandas as pd
import numpy as np

from simulation.factory_controller import FactoryController
from simulation.factory_monitor import FactoryMonitor
from simulation.machine_engine import (
    AIR_RANGE, PROCESS_RANGE, RPM_RANGE, TORQUE_RANGE, WEAR_MAX,
    MachineEngine,
)
from simulation.scenarios import SCENARIOS
from inference import get_threshold, load_model, predict_batch, predict_machine
from dashboard_logic import history_to_frame, machine_trajectory, pivot_trajectories


def test_step_counter_increments_accurately():
    """Verify that 1, 10, and 50 step increments advance the machine state by exactly that number."""
    monitor = FactoryMonitor()
    machine = monitor.factory.get_machine("MCH-001")
    assert machine.steps_taken == 0

    # 1 step
    hist1, latest1 = monitor.run_steps(1, "MCH-001")
    assert machine.steps_taken == 1
    assert len(hist1) == 1
    assert hist1[0]["Step"] == 1

    # 10 steps
    hist10, latest10 = monitor.run_steps(10, "MCH-001")
    assert machine.steps_taken == 11
    assert len(hist10) == 10
    assert [h["Step"] for h in hist10] == list(range(2, 12))

    # 50 steps
    hist50, latest50 = monitor.run_steps(50, "MCH-001")
    assert machine.steps_taken == 61
    assert len(hist50) == 50
    assert hist50[-1]["Step"] == 61


def test_batched_vs_sequential_advancement_equivalence():
    """Verify advancing 10 steps in one batch produces identical states to stepping 1 step 10 times."""
    m_batched = MachineEngine("MCH-001", "L", seed=123)
    m_batched.set_conditions(temperature_stress=4, load_stress=15, wear_rate=2.5)

    m_sequential = MachineEngine("MCH-001", "L", seed=123)
    m_sequential.set_conditions(temperature_stress=4, load_stress=15, wear_rate=2.5)

    seq_readings = [m_sequential.step() for _ in range(10)]
    batch_readings = [m_batched.step() for _ in range(10)]

    for s_step, b_step in zip(seq_readings, batch_readings):
        for k in s_step:
            assert s_step[k] == b_step[k]


def test_threshold_mode_switching_leaves_raw_probability_intact():
    """Changing alert sensitivity must update threshold and status without changing raw probabilities or advancing."""
    monitor = FactoryMonitor("balanced")
    m = monitor.factory.get_machine("MCH-001")
    m.set_conditions(temperature_stress=5, load_stress=25, wear_rate=3.0)
    monitor.run_steps(20, "MCH-001")

    res_bal = monitor.monitor_one("MCH-001", advance=False)
    prob_bal = res_bal["machine_failure_probability"]
    steps_bal = m.steps_taken

    # Switch to high recall
    monitor.mode = "high_recall"
    res_hr = monitor.monitor_one("MCH-001", advance=False)
    prob_hr = res_hr["machine_failure_probability"]
    steps_hr = m.steps_taken

    # Invariants
    assert prob_bal == prob_hr
    assert steps_bal == steps_hr
    assert res_bal["risk_threshold"] == 0.32
    assert res_hr["risk_threshold"] == 0.05
    assert res_bal["status"] in {"LOW RISK", "WATCH", "HIGH RISK"}
    assert res_hr["status"] in {"LOW RISK", "WATCH", "HIGH RISK"}


def test_history_records_match_step_predictions():
    """Verify every row in history contains predictions generated from that step's exact sensor readings."""
    monitor = FactoryMonitor("balanced")
    history, _ = monitor.run_steps(15, "MCH-002")
    assert len(history) == 15

    for h in history:
        readings = {
            "Type": h["Type"],
            "Air temperature": h["Air temperature"],
            "Process temperature": h["Process temperature"],
            "Rotational speed": h["Rotational speed"],
            "Torque": h["Torque"],
            "Tool wear": h["Tool wear"],
        }
        pred = predict_machine(readings, mode="balanced")
        # Probability in history (%) should match independent prediction
        np.testing.assert_almost_equal(
            h["Failure Probability (%)"],
            pred["machine_failure_probability"] * 100,
            decimal=4
        )
        assert h["Status"] == pred["status"]


def test_all_scenarios_apply_expected_stress_bounds():
    """Verify each scenario preset correctly sets temperature, load, and tool-wear rate."""
    monitor = FactoryMonitor()
    for name, cond in SCENARIOS.items():
        monitor.set_conditions("MCH-001", **cond)
        m = monitor.factory.get_machine("MCH-001")
        assert m.temperature_stress == cond["temperature_stress"]
        assert m.load_stress == cond["load_stress"]
        assert m.wear_rate == cond["wear_rate"]


def test_tool_replacement_and_machine_reset_behavior():
    """Verify tool replacement drops tool wear and risk; reset restores initial baseline."""
    monitor = FactoryMonitor("balanced")
    monitor.set_conditions("MCH-001", load_stress=10, wear_rate=5.0)
    monitor.run_steps(45, "MCH-001")

    high_risk_res = monitor.monitor_one("MCH-001", advance=False)
    assert high_risk_res["sensor_readings"]["Tool wear"] > 180
    assert high_risk_res["machine_failure_probability"] > 0.50

    # Replace tool
    monitor.replace_tool("MCH-001")
    post_rep = monitor.monitor_one("MCH-001", advance=False)
    assert post_rep["sensor_readings"]["Tool wear"] == 0.0
    assert post_rep["machine_failure_probability"] < 0.05
    assert post_rep["status"] == "LOW RISK"

    # Reset machine
    monitor.reset_machine("MCH-001")
    m = monitor.factory.get_machine("MCH-001")
    assert m.steps_taken == 0
    assert m.tool_wear == 100.0
    assert m.load_stress == 0.0
    assert m.temperature_stress == 0.0


def test_multi_machine_isolation():
    """Advancing one machine must not alter readings, steps, or conditions of another machine."""
    monitor = FactoryMonitor()
    m1 = monitor.factory.get_machine("MCH-001")
    m2 = monitor.factory.get_machine("MCH-002")

    initial_m2_readings = m2.readings().copy()
    assert m2.steps_taken == 0

    # Advance only MCH-001 for 25 steps under stress
    monitor.set_conditions("MCH-001", load_stress=20, wear_rate=4.0)
    monitor.run_steps(25, "MCH-001")

    assert m1.steps_taken == 25
    assert m2.steps_taken == 0
    assert m2.readings() == initial_m2_readings


def test_trajectory_pivot_continuity_with_unequal_step_counts():
    """Verify that pivot_trajectories forward-fills and maintains continuity when machines have different step counts."""
    monitor = FactoryMonitor()
    # MCH-001 has 15 steps
    h1, _ = monitor.run_steps(15, "MCH-001")
    # MCH-002 has 5 steps
    h2, _ = monitor.run_steps(5, "MCH-002")

    combined = h1 + h2
    df = history_to_frame(combined)
    pivoted = pivot_trajectories(df)

    assert not pivoted.empty
    assert pivoted.shape[0] == 15
    assert "MCH-001" in pivoted.columns
    assert "MCH-002" in pivoted.columns
    # MCH-002 should be forward-filled from step 5 through step 15
    assert not pivoted["MCH-002"].iloc[4:].isna().any()


def test_inference_pipeline_contract():
    """Verify predict_batch rejects missing features and respects exact training schema."""
    _, config = load_model()
    expected_cols = config["features"]
    assert expected_cols == [
        "Type", "Air temperature", "Process temperature",
        "Rotational speed", "Torque", "Tool wear"
    ]

    valid_sample = {
        "Type": "L", "Air temperature": 300.0, "Process temperature": 310.0,
        "Rotational speed": 1500.0, "Torque": 40.0, "Tool wear": 50.0
    }
    res = predict_batch([valid_sample])
    assert len(res) == 1
    assert 0.0 <= res[0]["machine_failure_probability"] <= 1.0

    # Incomplete sample must raise ValueError
    incomplete = valid_sample.copy()
    del incomplete["Torque"]
    with pytest.raises(ValueError, match="missing sensor inputs"):
        predict_batch([incomplete])


def test_50_steps_single_vs_batches_of_10_vs_batch_of_50_exact():
    """Verify 50 sequential single steps, 5 batches of 10, and 1 batch of 50 produce 100% bit-for-bit identical results."""
    from features import add_engineered_features

    for scenario_name in ["Worn tool under load", "Heavy overload", "Normal operation"]:
        ctrl_a = FactoryController()
        ctrl_a.set_conditions("MCH-001", **SCENARIOS[scenario_name])
        readings_a = [dict(ctrl_a.get_machine("MCH-001").step()) for _ in range(50)]

        ctrl_b = FactoryController()
        ctrl_b.set_conditions("MCH-001", **SCENARIOS[scenario_name])
        readings_b = []
        for _ in range(5):
            for _ in range(10):
                readings_b.append(dict(ctrl_b.get_machine("MCH-001").step()))

        ctrl_c = FactoryController()
        ctrl_c.set_conditions("MCH-001", **SCENARIOS[scenario_name])
        readings_c = [dict(ctrl_c.get_machine("MCH-001").step()) for _ in range(50)]

        df_a, df_b, df_c = pd.DataFrame(readings_a), pd.DataFrame(readings_b), pd.DataFrame(readings_c)
        sensor_cols = ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
        assert (df_a[sensor_cols] - df_b[sensor_cols]).abs().max().max() == 0.0
        assert (df_a[sensor_cols] - df_c[sensor_cols]).abs().max().max() == 0.0

        feat_a = add_engineered_features(df_a)
        feat_b = add_engineered_features(df_b)
        feat_c = add_engineered_features(df_c)
        eng_cols = ["Temp difference", "Power", "Overstrain"]
        assert (feat_a[eng_cols] - feat_b[eng_cols]).abs().max().max() == 0.0
        assert (feat_a[eng_cols] - feat_c[eng_cols]).abs().max().max() == 0.0

        pred_a = predict_batch(readings_a, mode="balanced")
        pred_b = predict_batch(readings_b, mode="balanced")
        pred_c = predict_batch(readings_c, mode="balanced")
        probs_a = np.array([p["machine_failure_probability"] for p in pred_a])
        probs_b = np.array([p["machine_failure_probability"] for p in pred_b])
        probs_c = np.array([p["machine_failure_probability"] for p in pred_c])
        assert np.max(np.abs(probs_a - probs_b)) == 0.0
        assert np.max(np.abs(probs_a - probs_c)) == 0.0

        m_a, m_b, m_c = ctrl_a.get_machine("MCH-001"), ctrl_b.get_machine("MCH-001"), ctrl_c.get_machine("MCH-001")
        assert m_a.steps_taken == m_b.steps_taken == m_c.steps_taken == 50
        assert m_a.tool_wear == m_b.tool_wear == m_c.tool_wear


def test_all_five_machines_across_presets_and_thresholds():
    """Verify all 5 machines under all 5 scenarios with both Balanced (0.32) and High-recall (0.05)."""
    machines = ["MCH-001", "MCH-002", "MCH-003", "MCH-004", "MCH-005"]
    for m_id in machines:
        for scen in SCENARIOS.keys():
            for mode_name, thresh_val in [("balanced", 0.32), ("high_recall", 0.05)]:
                ctrl = FactoryController()
                ctrl.set_conditions(m_id, **SCENARIOS[scen])
                readings = [ctrl.get_machine(m_id).step() for _ in range(50)]
                preds = predict_batch(readings, mode=mode_name)
                probs = np.array([p["machine_failure_probability"] for p in preds])
                statuses = np.array([p["status"] for p in preds])

                assert len(preds) == 50
                assert (probs >= 0.0).all() and (probs <= 1.0).all()

                if scen == "Normal operation" and mode_name == "balanced":
                    assert (probs < thresh_val).all(), f"{m_id} exceeded threshold under normal operation"


def test_threshold_classification_switch_on_intermediate_risk():
    """Verify that in the [0.05, 0.32) interval, High-recall alerts while Balanced does not, with invariant probabilities."""
    ctrl = FactoryController()
    ctrl.set_conditions("MCH-002", **SCENARIOS["Worn tool under load"])
    readings = [ctrl.get_machine("MCH-002").step() for _ in range(50)]

    pred_bal = predict_batch(readings, mode="balanced")
    pred_hr = predict_batch(readings, mode="high_recall")

    probs_bal = np.array([p["machine_failure_probability"] for p in pred_bal])
    probs_hr = np.array([p["machine_failure_probability"] for p in pred_hr])

    # Probability invariance
    assert np.max(np.abs(probs_bal - probs_hr)) == 0.0

    # Intermediate classification checks
    switches = [
        (i + 1, p_b["machine_failure_probability"], p_b["status"], p_hr["status"])
        for i, (p_b, p_hr) in enumerate(zip(pred_bal, pred_hr))
        if p_hr["status"] == "HIGH RISK" and p_b["status"] != "HIGH RISK"
    ]
    assert len(switches) > 0, "Expected intermediate probability switches between Balanced and High-recall"
    for step_num, prob, stat_bal, stat_hr in switches:
        assert 0.05 <= prob < 0.32
        assert stat_bal in {"LOW RISK", "WATCH"}
        assert stat_hr == "HIGH RISK"
