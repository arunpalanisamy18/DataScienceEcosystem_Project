from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard_logic import (
    history_to_frame, load_report, machine_trajectory, make_machine_dataframe,
    pivot_trajectories, summarise_factory,
)
from explain import failure_mode_warnings
from inference import MODES, get_threshold, load_model
from simulation.factory_monitor import FactoryMonitor
from simulation.scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

st.set_page_config(
    page_title="Predictive Maintenance Intelligence",
    page_icon="⚙️",
    layout="wide",
)

st.title("Predictive Maintenance Intelligence")
st.caption("Multi-Machine Factory Monitoring | Simulated Sensor Data | ML Predictions")

# ------------------------------------------------------------------ loading
try:
    load_model()
except Exception as error:
    st.error(f"Model problem: {error}")
    st.stop()


@st.cache_data
def load_dataset():
    return pd.read_csv(ROOT / "ai4i2020_raw.csv")


try:
    df = load_dataset()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

# ----------------------------------------------------------- session state
if "monitor" not in st.session_state:
    st.session_state.monitor = FactoryMonitor()
if "history" not in st.session_state:
    st.session_state.history = []
if "latest_results" not in st.session_state:
    st.session_state.latest_results = st.session_state.monitor.monitor_all(advance=False)
if "flash" not in st.session_state:
    st.session_state.flash = None

monitor = st.session_state.monitor

# ----------------------------------------------------------------- sidebar
st.sidebar.title("Factory Control")
page = st.sidebar.radio(
    "Navigate",
    ["Factory Overview", "Simulation Controls", "Analytics", "About & Limitations"],
)

mode_key = st.sidebar.selectbox(
    "Alert sensitivity",
    list(MODES.keys()),
    format_func=lambda k: MODES[k],
    help="Balanced maximises F1. High-recall catches more failures but raises "
         "more false alarms. Both thresholds were chosen on validation data.",
)
if mode_key != monitor.mode:
    monitor.mode = mode_key
    st.session_state.latest_results = monitor.monitor_all(advance=False)
st.sidebar.caption(f"Alert threshold: {get_threshold(mode_key):.0%} predicted probability")

st.sidebar.markdown("---")
st.sidebar.caption("All live machine readings are simulated.")
st.sidebar.caption("Predictions come from the trained Gradient Boosting model.")
st.sidebar.markdown("---")
st.sidebar.caption("Predictive Maintenance Intelligence | IA3 Data Science Project")


def run_and_record(n_steps, machine_id=None):
    """Advance the simulation and keep the history for charts."""
    history, latest = monitor.run_steps(n_steps, machine_id)
    st.session_state.history.extend(history)
    st.session_state.latest_results = latest


def show_flash():
    if st.session_state.flash:
        st.success(st.session_state.flash)
        st.session_state.flash = None


def status_banner(status, text):
    if status == "HIGH RISK":
        st.error(text)
    elif status == "WATCH":
        st.warning(text)
    else:
        st.success(text)


# ----------------------------------------------------- PAGE 1: OVERVIEW
if page == "Factory Overview":
    show_flash()
    st.subheader("Factory Status")

    results = st.session_state.latest_results
    machines_df = make_machine_dataframe(results)
    summary = summarise_factory(machines_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Simulated Machines", summary["total"])
    c2.metric("HIGH RISK (above threshold)", summary["high"])
    c3.metric("On WATCH", summary["watch"])
    c4.metric("Mean Predicted Failure Score", f"{summary['mean_probability']:.2f}%")

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Advance all machines: 1 step", type="primary"):
        run_and_record(1)
        st.rerun()
    if b2.button("Advance all: 10 steps"):
        run_and_record(10)
        st.rerun()
    if b3.button("Advance all: 50 steps"):
        run_and_record(50)
        st.rerun()
    if b4.button("Reset whole factory"):
        monitor.factory.reset_factory()
        st.session_state.history = []
        st.session_state.latest_results = monitor.monitor_all(advance=False)
        st.session_state.flash = "Factory reset to baseline."
        st.rerun()

    st.markdown("### Machine Monitoring Cards")
    cols = st.columns(2)
    for index, (machine_id, result) in enumerate(results.items()):
        row = machines_df.iloc[index]
        with cols[index % 2]:
            with st.container(border=True):
                st.markdown(f"#### {machine_id} — Type {row['Type']}")
                st.metric("Predicted failure probability",
                          f"{row['Failure Probability (%)']:.2f}%")
                status_banner(result["status"], f"Status: {result['status']}")
                from dashboard_logic import recommend_maintenance
                advice = recommend_maintenance(result)
                st.markdown(f"**Next step — {advice['priority']}**")
                st.write(advice["action"])
                st.caption(f"Rule warnings: {advice['warning_codes']}")
                st.caption(f"Air temperature: {row['Air Temp']:.2f} K | "
                           f"Process temperature: {row['Process Temp']:.2f} K")
                st.caption(f"RPM: {row['RPM']:.0f} | Torque: {row['Torque']:.1f} Nm | "
                           f"Tool wear: {row['Tool Wear']:.0f} min")
                for code, text in result["warnings"]:
                    st.caption(f"⚠️ **{code}** — {text}")

    st.markdown("### Risk Over Time")
    wide = pivot_trajectories(history_to_frame(st.session_state.history))
    if wide.empty:
        st.info("No history yet. Use the **Advance** buttons above to simulate machine "
                "operation; the risk trajectories will appear here.")
    else:
        st.line_chart(wide)
        st.caption("Predicted failure probability (%) per simulation step.")

    st.markdown("### Maintenance Priority Queue")
    priority_df = machines_df.sort_values(
        ["Status", "Failure Probability (%)"],
        ascending=[False, False],
        key=None,
    )
    st.dataframe(
        priority_df[["Machine", "Status", "Failure Probability (%)",
                     "Maintenance Priority", "Rule Warnings", "Suggested Action"]],
        hide_index=True,
        width="stretch",
    )
    st.caption("Suggested actions combine the model's risk band with transparent dataset-rule warnings; they are advisory, not a diagnosis.")
    st.markdown("### Compare Machines")
    st.bar_chart(machines_df.set_index("Machine")["Failure Probability (%)"])
    st.markdown("### Current Machine Readings")
    st.dataframe(machines_df, hide_index=True, width="stretch")
    st.caption("A predicted score is a model estimate from sensor readings, "
               "not a guarantee of actual failure.")

# ----------------------------------------------- PAGE 2: SIMULATION CONTROLS
elif page == "Simulation Controls":
    show_flash()
    st.subheader("Simulation Control Center")
    st.info("Choose a machine, pick a scenario or set the operating conditions, then "
            "advance the simulation to generate new sensor readings and predictions.")

    machine_ids = monitor.factory.list_machines()
    selected = st.selectbox("Select machine", machine_ids)
    machine = monitor.factory.get_machine(selected)

    st.markdown("### Quick Scenarios")
    scenario_cols = st.columns(len(SCENARIOS))
    for col, (name, cond) in zip(scenario_cols, SCENARIOS.items()):
        if col.button(name, key=f"scn_{name}"):
            monitor.set_conditions(selected, **cond)
            st.session_state[f"ts_{selected}"] = int(cond["temperature_stress"])
            st.session_state[f"ls_{selected}"] = int(cond["load_stress"])
            st.session_state[f"wr_{selected}"] = float(cond["wear_rate"])
            st.session_state[f"active_scenario_{selected}"] = name
            st.session_state.flash = f"Scenario “{name}” applied to {selected}."
            st.rerun()

    # Ensure slider state is initialized for the selected machine
    if f"ts_{selected}" not in st.session_state:
        st.session_state[f"ts_{selected}"] = int(machine.temperature_stress)
    if f"ls_{selected}" not in st.session_state:
        st.session_state[f"ls_{selected}"] = int(machine.load_stress)
    if f"wr_{selected}" not in st.session_state:
        st.session_state[f"wr_{selected}"] = float(machine.wear_rate)
    if f"active_scenario_{selected}" not in st.session_state:
        st.session_state[f"active_scenario_{selected}"] = "Normal operation"

    active_scn = st.session_state[f"active_scenario_{selected}"]
    st.caption(
        f"Active on **{selected}**: Temperature stress = `{machine.temperature_stress:+.0f}` K | "
        f"Load stress = `{machine.load_stress:+.0f}` Nm | "
        f"Tool-wear rate = `{machine.wear_rate:.1f}` min/step | "
        f"Preset: **{active_scn}**"
    )

    st.markdown("### Operating Conditions")
    temperature_stress = st.slider(
        "Temperature stress", -5, 10,
        help="Raises air temperature and narrows the process–air temperature gap.",
        key=f"ts_{selected}")
    load_stress = st.slider(
        "Load stress", -10, 30,
        help="Raises torque and lowers rotational speed.",
        key=f"ls_{selected}")
    wear_rate = st.slider(
        "Tool-wear rate", 0.0, 5.0, step=0.5,
        help="Controls how quickly simulated tool wear accumulates.",
        key=f"wr_{selected}")

    if st.button("Apply conditions", type="primary"):
        monitor.set_conditions(selected, temperature_stress=temperature_stress,
                               load_stress=load_stress, wear_rate=wear_rate)
        st.session_state[f"active_scenario_{selected}"] = "Custom"
        st.session_state.flash = f"Operating conditions applied to {selected}."
        st.rerun()

    st.markdown("---")
    a1, a2, a3, a4, a5 = st.columns(5)
    if a1.button("Advance 1 step"):
        monitor.set_conditions(selected, temperature_stress=temperature_stress,
                               load_stress=load_stress, wear_rate=wear_rate)
        run_and_record(1, selected)
        st.rerun()
    if a2.button("Advance 10 steps"):
        monitor.set_conditions(selected, temperature_stress=temperature_stress,
                               load_stress=load_stress, wear_rate=wear_rate)
        run_and_record(10, selected)
        st.rerun()
    if a3.button("Advance 50 steps"):
        monitor.set_conditions(selected, temperature_stress=temperature_stress,
                               load_stress=load_stress, wear_rate=wear_rate)
        run_and_record(50, selected)
        st.rerun()
    if a4.button("Replace tool"):
        monitor.replace_tool(selected)
        latest = monitor.monitor_all(advance=False)
        st.session_state.latest_results = latest
        result = latest[selected]
        st.session_state.history.append({
            "Step": machine.steps_taken,
            "Machine": selected,
            **result["sensor_readings"],
            "Failure Probability (%)": result["machine_failure_probability"] * 100,
            "Status": result["status"],
        })
        st.session_state.flash = f"New tool fitted on {selected} (tool wear = 0)."
        st.rerun()
    if a5.button("Reset machine"):
        monitor.reset_machine(selected)
        st.session_state.history = [h for h in st.session_state.history
                                    if h["Machine"] != selected]
        st.session_state[f"ts_{selected}"] = 0
        st.session_state[f"ls_{selected}"] = 0
        st.session_state[f"wr_{selected}"] = 1.0
        st.session_state[f"active_scenario_{selected}"] = "Normal operation"
        st.session_state.latest_results = monitor.monitor_all(advance=False)
        st.session_state.flash = f"{selected} has been reset."
        st.rerun()

    st.markdown("### Current Sensor Readings")
    st.dataframe(pd.DataFrame([machine.readings()]), hide_index=True, width="stretch")
    st.caption(f"Simulation steps taken: {machine.steps_taken}")

    st.markdown("### Current Model Prediction")
    current = st.session_state.latest_results[selected]
    p_col, s_col = st.columns(2)
    p_col.metric("Predicted failure score", f"{current['machine_failure_probability']:.2%}")
    s_col.metric("Prediction status", current["status"])
    status_banner(current["status"],
                  f"{selected} is {current['status']} "
                  f"(alert threshold {current['risk_threshold']:.0%}).")
    from dashboard_logic import recommend_maintenance
    advice = recommend_maintenance(current)
    st.markdown(f"**Recommended next step — {advice['priority']}**")
    st.write(advice["action"])
    if current["warnings"]:
        st.markdown("**Why this machine needs attention** (failure-mode rules from the dataset documentation):")
        for code, text in current["warnings"]:
            st.markdown(f"- **{code}**: {text}")
    st.caption("The score comes from the trained model using the current sensor readings. "
               "It is not a confirmed real-world failure probability.")

    st.markdown("### Risk Trajectory for this Machine")
    trajectory = machine_trajectory(history_to_frame(st.session_state.history), selected)
    if trajectory.empty:
        st.info("Advance this machine to build its risk trajectory.")
    else:
        st.line_chart(trajectory)

    st.markdown("### Factory-Wide Status")
    st.dataframe(make_machine_dataframe(st.session_state.latest_results), hide_index=True, width="stretch")

# ------------------------------------------------------- PAGE 3: ANALYTICS
elif page == "Analytics":
    st.subheader("Data Science Analytics")

    st.markdown("### Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Records", f"{len(df):,}")
    c2.metric("Dataset Columns", len(df.columns))
    c3.metric("Historical Failure Rate", f"{df['Machine failure'].mean():.2%}")
    st.caption("Dataset: UCI AI4I 2020 Predictive Maintenance Dataset (synthetic, 10,000 records).")

    split = load_report("data_split_summary.csv")
    if split is not None:
        st.markdown("**Data split (stratified)**")
        st.dataframe(split, hide_index=True)

    st.markdown("### Exploratory Analysis")
    e1, e2 = st.columns(2)
    for col, name, cap in [
        (e1, "01_failure_distribution.png", "Severe class imbalance (3.39% failures)"),
        (e2, "09_failure_modes.png", "Failures by documented failure mode"),
        (e1, "02_sensor_comparison.png", "Sensor readings by failure status"),
        (e2, "10_engineered_features.png", "Engineered physics-based features"),
        (e1, "03_correlation_heatmap.png", "Correlation heat-map"),
        (e2, "08_failure_by_type.png", "Failure rate by machine type"),
    ]:
        if (REPORTS / name).exists():
            col.image(str(REPORTS / name), caption=cap)

    st.markdown("### Model Comparison (5-fold cross-validation, training split)")
    cv_df = load_report("cv_model_comparison.csv")
    if cv_df is not None:
        show = cv_df[["Model", "pr_auc mean", "pr_auc std", "recall mean", "precision mean", "f1 mean"]].round(3)
        st.dataframe(show, hide_index=True)
    if (REPORTS / "05_model_comparison.png").exists():
        st.image(str(REPORTS / "05_model_comparison.png"))

    st.markdown("### Final Held-Out Test Results")
    final = load_report("final_test_metrics.csv")
    if final is not None:
        cols = ["Operating mode", "Threshold", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC", "TP", "FP", "FN", "TN"]
        st.dataframe(final[cols].round(3), hide_index=True)
    ci = load_report("bootstrap_confidence_intervals.csv")
    if ci is not None:
        st.markdown("**Bootstrap 95% confidence intervals (balanced mode)**")
        st.dataframe(ci.round(3), hide_index=True)

    r1, r2 = st.columns(2)
    for col, name, cap in [
        (r1, "06_confusion_matrix.png", "Confusion matrix (test set)"),
        (r2, "07_precision_recall.png", "Precision–recall: final model vs baseline"),
        (r1, "12_threshold_selection.png", "Threshold chosen on validation data"),
        (r2, "13_calibration.png", "Calibration of predicted probabilities"),
        (r1, "04_feature_importance.png", "Feature importance"),
        (r2, "14_recall_by_failure_mode.png", "Share of failures detected by mode"),
    ]:
        if (REPORTS / name).exists():
            col.image(str(REPORTS / name), caption=cap)

    st.markdown("### Simulation History")
    hist = history_to_frame(st.session_state.history)
    if hist.empty:
        st.info("No simulation history yet. Advance machines on the Factory Overview or "
                "Simulation Controls page and the recorded predictions will appear here.")
    else:
        st.line_chart(pivot_trajectories(hist))
        st.dataframe(hist.tail(200), hide_index=True)
        st.download_button("Download simulation history (CSV)",
                           hist.to_csv(index=False).encode("utf-8"),
                           file_name="simulation_history.csv", mime="text/csv")

# ----------------------------------------------------------- PAGE 4: ABOUT
else:
    st.subheader("About & Limitations")
    st.markdown(
        """
**Problem.** Unplanned machine failure is costly. This system predicts the probability
that a machine will fail from six sensor readings, so maintenance can be scheduled
*before* a breakdown.

**Model.** Gradient Boosting on the raw sensors plus three physics-based features
(temperature gap, power, overstrain), tuned with cross-validation. The alert threshold
was chosen on a separate validation set; the test set was used once for final scores.
Failure-mode label columns were excluded from the inputs to avoid target leakage.

**What the colours mean.**
- **LOW RISK** – probability below half the alert threshold
- **WATCH** – between half the threshold and the threshold
- **HIGH RISK** – at or above the alert threshold

**Limitations.**
1. Live readings are *simulated*; the model has not been tested on real telemetry.
2. The AI4I dataset is synthetic, so real-world performance may be lower.
3. Only 3.39% of records are failures; accuracy alone is misleading, so precision,
   recall, F1 and PR-AUC are reported.
4. Random failures (RNF) and the random part of tool-wear failures (TWF) are essentially
   unpredictable from sensors; the model detects them poorly (see Analytics).
5. The simulator is limited to the sensor ranges seen in training, so the model is
   never asked to extrapolate.
"""
    )
