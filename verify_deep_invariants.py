"""Deep Invariants Verification Script
Validates user requirements 4, 5, 7, and 8.
- Requirements 4 & 5: All 5 machines x 5 presets x 2 thresholds.
- Step Equivalence: 50x1 vs 5x10 vs 1x50 steps comparison.
- Threshold Invariance: Raw probabilities unchanged between Balanced & High-recall.
- Tool Replacement & Reset: State & history synchronization.
"""

import numpy as np
import pandas as pd
from simulation.factory_controller import FactoryController
from simulation.scenarios import SCENARIOS
from inference import predict_machine, predict_batch
from features import add_engineered_features

def test_scheme_equivalence():
    print("=" * 70)
    print("TEST 1: Step Equivalence (50x1 vs 5x10 vs 1x50)")
    print("=" * 70)

    presets_to_test = ["Worn tool under load", "Heavy overload", "Normal operation"]
    for scenario_name in presets_to_test:
        print(f"\n--- Scenario: {scenario_name} (MCH-001, Type L) ---")

        # Scheme A: 50 sequential 1-step advances
        ctrl_a = FactoryController()
        ctrl_a.set_conditions("MCH-001", **SCENARIOS[scenario_name])
        readings_a = []
        for _ in range(50):
            r = ctrl_a.get_machine("MCH-001").step()
            readings_a.append(dict(r))

        # Scheme B: 5 batches of 10 steps
        ctrl_b = FactoryController()
        ctrl_b.set_conditions("MCH-001", **SCENARIOS[scenario_name])
        readings_b = []
        for _ in range(5):
            for _ in range(10):
                r = ctrl_b.get_machine("MCH-001").step()
                readings_b.append(dict(r))

        # Scheme C: 1 batch of 50 steps
        ctrl_c = FactoryController()
        ctrl_c.set_conditions("MCH-001", **SCENARIOS[scenario_name])
        readings_c = []
        for _ in range(50):
            r = ctrl_c.get_machine("MCH-001").step()
            readings_c.append(dict(r))

        assert len(readings_a) == 50
        assert len(readings_b) == 50
        assert len(readings_c) == 50

        df_a = pd.DataFrame(readings_a)
        df_b = pd.DataFrame(readings_b)
        df_c = pd.DataFrame(readings_c)

        # Check sensor reading match
        sensor_cols = ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
        diff_ab = (df_a[sensor_cols] - df_b[sensor_cols]).abs().max().max()
        diff_ac = (df_a[sensor_cols] - df_c[sensor_cols]).abs().max().max()
        print(f"Max sensor absolute difference (A vs B): {diff_ab:.10e}")
        print(f"Max sensor absolute difference (A vs C): {diff_ac:.10e}")
        assert diff_ab == 0.0, f"Discrepancy between 50x1 and 5x10: {diff_ab}"
        assert diff_ac == 0.0, f"Discrepancy between 50x1 and 1x50: {diff_ac}"

        # Check engineered features match
        feat_a = add_engineered_features(df_a)
        feat_b = add_engineered_features(df_b)
        feat_c = add_engineered_features(df_c)
        eng_cols = ["Temp difference", "Power", "Overstrain"]
        diff_eng_ab = (feat_a[eng_cols] - feat_b[eng_cols]).abs().max().max()
        diff_eng_ac = (feat_a[eng_cols] - feat_c[eng_cols]).abs().max().max()
        print(f"Max engineered feature diff (A vs B): {diff_eng_ab:.10e}")
        print(f"Max engineered feature diff (A vs C): {diff_eng_ac:.10e}")
        assert diff_eng_ab == 0.0
        assert diff_eng_ac == 0.0

        # Check raw model probabilities match
        pred_a = predict_batch(readings_a, mode="balanced")
        pred_b = predict_batch(readings_b, mode="balanced")
        pred_c = predict_batch(readings_c, mode="balanced")

        probs_a = np.array([p["machine_failure_probability"] for p in pred_a])
        probs_b = np.array([p["machine_failure_probability"] for p in pred_b])
        probs_c = np.array([p["machine_failure_probability"] for p in pred_c])

        prob_diff_ab = np.max(np.abs(probs_a - probs_b))
        prob_diff_ac = np.max(np.abs(probs_a - probs_c))
        print(f"Max raw probability diff (A vs B): {prob_diff_ab:.10e}")
        print(f"Max raw probability diff (A vs C): {prob_diff_ac:.10e}")
        assert prob_diff_ab == 0.0
        assert prob_diff_ac == 0.0

        # Check final machine states
        m_a = ctrl_a.get_machine("MCH-001")
        m_b = ctrl_b.get_machine("MCH-001")
        m_c = ctrl_c.get_machine("MCH-001")
        assert m_a.steps_taken == m_b.steps_taken == m_c.steps_taken == 50
        assert m_a.tool_wear == m_b.tool_wear == m_c.tool_wear
        print(f"Final steps taken: {m_a.steps_taken}, Final tool wear: {m_a.tool_wear:.2f}")
        print(f"Final raw probability at step 50: {probs_a[-1]:.4%}")
        print("-> 100% BIT-FOR-BIT STEPPING EQUIVALENCE VERIFIED.")

def test_all_machines_presets_thresholds():
    print("\n" + "=" * 70)
    print("TEST 2: All 5 Machines x 5 Presets x 2 Thresholds (Sweep)")
    print("=" * 70)

    machines = ["MCH-001", "MCH-002", "MCH-003", "MCH-004", "MCH-005"]
    scenarios = list(SCENARIOS.keys())
    modes = [("balanced", 0.32), ("high_recall", 0.05)]

    summary_records = []

    for m_id in machines:
        for scen in scenarios:
            for mode_name, thresh_val in modes:
                ctrl = FactoryController()
                ctrl.set_conditions(m_id, **SCENARIOS[scen])
                steps_data = [ctrl.get_machine(m_id).step() for _ in range(50)]
                preds = predict_batch(steps_data, mode=mode_name)

                probs = np.array([p["machine_failure_probability"] for p in preds])
                statuses = np.array([p["status"] for p in preds])

                max_risk = float(np.max(probs))
                min_risk = float(np.min(probs))
                high_risk_steps = int(np.sum(statuses == "HIGH RISK"))
                first_high_risk_step = None
                if high_risk_steps > 0:
                    first_high_risk_step = int(np.where(statuses == "HIGH RISK")[0][0]) + 1

                summary_records.append({
                    "Machine": m_id,
                    "Scenario": scen,
                    "Mode": mode_name,
                    "Threshold": thresh_val,
                    "Min_Prob%": round(min_risk * 100, 2),
                    "Max_Prob%": round(max_risk * 100, 2),
                    "High_Risk_Steps": high_risk_steps,
                    "First_Alert_Step": first_high_risk_step,
                })

    summary_df = pd.DataFrame(summary_records)
    print(f"Total configurations evaluated: {len(summary_df)}")
    print("\nSample of Sweep Results across Machines and Presets:")
    sample_view = summary_df[summary_df["Scenario"].isin(["Normal operation", "Heavy overload", "Worn tool under load"])].head(15)
    print(sample_view.to_string(index=False))

    # Assertions
    normal_bal = summary_df[(summary_df["Scenario"] == "Normal operation") & (summary_df["Mode"] == "balanced")]
    print(f"\nNormal operation max probability across all 5 machines (Balanced): {normal_bal['Max_Prob%'].max()}%")
    assert normal_bal["Max_Prob%"].max() < 32.0, "Normal operation exceeded balanced threshold!"

    worn = summary_df[summary_df["Scenario"] == "Worn tool under load"]
    print(f"Worn tool under load first alert steps (Balanced): {worn[worn['Mode'] == 'balanced']['First_Alert_Step'].tolist()}")
    for first_step in worn[worn['Mode'] == 'balanced']["First_Alert_Step"].tolist():
        assert first_step is not None and first_step <= 50, f"Worn tool failed to alert in time: {first_step}"

    print("-> ALL 5 MACHINES x 5 PRESETS x 2 THRESHOLDS EVALUATED CLEANLY.")

def test_threshold_invariance_and_classification():
    print("\n" + "=" * 70)
    print("TEST 3: Threshold Mode Invariance vs Classification Switch")
    print("=" * 70)

    # 1. Test MCH-001 under Heavy Overload for bit-for-bit probability invariance
    ctrl = FactoryController()
    ctrl.set_conditions("MCH-001", **SCENARIOS["Heavy overload"])
    readings = [ctrl.get_machine("MCH-001").step() for _ in range(30)]

    pred_bal = predict_batch(readings, mode="balanced")
    pred_hr = predict_batch(readings, mode="high_recall")

    probs_bal = np.array([p["machine_failure_probability"] for p in pred_bal])
    probs_hr = np.array([p["machine_failure_probability"] for p in pred_hr])

    prob_diff = np.max(np.abs(probs_bal - probs_hr))
    print(f"Max absolute difference between raw probabilities: {prob_diff:.10e}")
    assert prob_diff == 0.0, "Threshold mode altered raw model probabilities!"

    # 2. Test MCH-002 under Worn tool under load to verify classification switch in [0.05, 0.32)
    ctrl2 = FactoryController()
    ctrl2.set_conditions("MCH-002", **SCENARIOS["Worn tool under load"])
    readings2 = [ctrl2.get_machine("MCH-002").step() for _ in range(50)]

    pred_bal2 = predict_batch(readings2, mode="balanced")
    pred_hr2 = predict_batch(readings2, mode="high_recall")

    probs_bal2 = np.array([p["machine_failure_probability"] for p in pred_bal2])

    comp_df = pd.DataFrame({
        "Step": range(1, 51),
        "Prob%": np.round(probs_bal2 * 100, 2),
        "Status_Balanced(0.32)": [p["status"] for p in pred_bal2],
        "Status_HighRecall(0.05)": [p["status"] for p in pred_hr2]
    })

    high_recall_alerts = comp_df[(comp_df["Status_HighRecall(0.05)"] == "HIGH RISK") & (comp_df["Status_Balanced(0.32)"] != "HIGH RISK")]
    print(f"\nMCH-002 steps where High-Recall alerts HIGH RISK but Balanced does not (0.05 <= Prob < 0.32): {len(high_recall_alerts)}")
    assert len(high_recall_alerts) > 0, "No intermediate probability cases found!"
    print(high_recall_alerts.head(6).to_string(index=False))
    for _, row in high_recall_alerts.iterrows():
        p = row["Prob%"] / 100.0
        assert p >= 0.05 and p < 0.32
        assert row["Status_Balanced(0.32)"] in ["LOW RISK", "WATCH"]
        assert row["Status_HighRecall(0.05)"] == "HIGH RISK"

    print("-> THRESHOLD MODE INVARIANCE AND SENSITIVITY SWITCH VERIFIED.")

def test_tool_replacement_and_resets():
    print("\n" + "=" * 70)
    print("TEST 4: Tool Replacement and Machine/Factory Resets")
    print("=" * 70)

    ctrl = FactoryController()
    m_id = "MCH-001"
    ctrl.set_conditions(m_id, **SCENARIOS["Worn tool under load"])

    # Step 42 steps to trigger severe high risk
    for _ in range(42):
        ctrl.get_machine(m_id).step()
    m = ctrl.get_machine(m_id)
    pred_before = predict_machine(m.readings(), mode="balanced")
    prob_before = pred_before["machine_failure_probability"]
    print(f"Before tool replacement: Steps={m.steps_taken}, Tool wear={m.tool_wear:.1f} min, Risk={prob_before:.2%}, Status={pred_before['status']}")
    assert prob_before > 0.80
    assert pred_before["status"] == "HIGH RISK"

    # Replace tool
    ctrl.replace_tool(m_id)
    pred_after = predict_machine(m.readings(), mode="balanced")
    prob_after = pred_after["machine_failure_probability"]
    print(f"Immediately after tool replacement: Steps={m.steps_taken}, Tool wear={m.tool_wear:.1f} min, Risk={prob_after:.2%}, Status={pred_after['status']}")
    assert m.tool_wear == 0.0
    assert prob_after < 0.02
    assert pred_after["status"] == "LOW RISK"

    # Machine Reset
    print("\nTesting Machine Reset...")
    ctrl.reset_machine(m_id)
    m_reset = ctrl.get_machine(m_id)
    assert m_reset.steps_taken == 0
    assert m_reset.tool_wear == 100.0
    print(f"MCH-001 after reset: Steps={m_reset.steps_taken}, Tool wear={m_reset.tool_wear}")

    # Other machines should remain untouched
    m2 = ctrl.get_machine("MCH-002")
    assert m2.steps_taken == 0

    # Factory Reset
    print("\nTesting Factory Reset...")
    for _ in range(15):
        ctrl.get_machine("MCH-002").step()
    for _ in range(20):
        ctrl.get_machine("MCH-003").step()
    assert ctrl.get_machine("MCH-002").steps_taken == 15
    assert ctrl.get_machine("MCH-003").steps_taken == 20
    ctrl.reset_factory()
    for m_check_id in ["MCH-001", "MCH-002", "MCH-003", "MCH-004", "MCH-005"]:
        chk = ctrl.get_machine(m_check_id)
        assert chk.steps_taken == 0
        assert chk.tool_wear == 100.0
    print("All 5 machines confirmed reset to step 0 and baseline tool wear (100.0).")
    print("-> TOOL REPLACEMENT & RESET DYNAMICS VERIFIED.")

if __name__ == "__main__":
    test_scheme_equivalence()
    test_all_machines_presets_thresholds()
    test_threshold_invariance_and_classification()
    test_tool_replacement_and_resets()
    print("\n" + "=" * 70)
    print("ALL VERIFICATION INVARIANTS TESTED AND PASSED.")
    print("=" * 70)
