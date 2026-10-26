"""Pure-pandas helpers used by the Streamlit dashboard (unit-testable)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

STATUS_ORDER = {"LOW RISK": 0, "WATCH": 1, "HIGH RISK": 2}

WARNING_ACTIONS = {
    "TWF": "Plan tool replacement; tool wear has reached the dataset rule threshold.",
    "HDF": "Inspect cooling and heat dissipation; the temperature gap and RPM meet the warning rule.",
    "PWF": "Inspect load, torque, and rotational-speed readings; calculated power is outside the rule band.",
    "OSF": "Inspect tool wear and torque together; the type-specific overstrain rule is triggered.",
}


def recommend_maintenance(result):
    """Return transparent, rule-based next steps; these are not ML predictions."""
    status = result.get("status", "LOW RISK")
    warnings = result.get("warnings", [])
    codes = [item[0] for item in warnings]

    actions = [WARNING_ACTIONS[code] for code in codes if code in WARNING_ACTIONS]
    if status == "HIGH RISK":
        priority = "Urgent inspection"
        actions.insert(0, "Arrange a maintenance inspection soon; the model score crossed the selected alert threshold.")
    elif status == "WATCH":
        priority = "Monitor closely"
        actions.insert(0, "Recheck the next sensor cycle and review recent operating conditions.")
    else:
        priority = "Routine monitoring"
        actions.insert(0, "Continue routine monitoring; a low score does not guarantee the machine is safe.")

    if not codes and status != "LOW RISK":
        actions.append("No failure-mode rule fired; investigate the overall sensor pattern rather than assuming a specific fault.")
    return {"priority": priority, "action": " ".join(actions), "warning_codes": ", ".join(codes) if codes else "None"}


def make_machine_dataframe(results):
    """Convert the latest per-machine results into a display table."""
    records = []
    for machine_id, result in results.items():
        r = result["sensor_readings"]
        records.append({
            "Machine": machine_id,
            "Type": r["Type"],
            "Air Temp": r["Air temperature"],
            "Process Temp": r["Process temperature"],
            "RPM": r["Rotational speed"],
            "Torque": r["Torque"],
            "Tool Wear": r["Tool wear"],
            "Failure Probability (%)": round(result["machine_failure_probability"] * 100, 2),
            "Status": result["status"],
            "Maintenance Priority": recommend_maintenance(result)["priority"],
            "Suggested Action": recommend_maintenance(result)["action"],
            "Rule Warnings": recommend_maintenance(result)["warning_codes"],
        })
    return pd.DataFrame(records)


def summarise_factory(machines_df):
    """Headline numbers for the overview page."""
    return {
        "total": len(machines_df),
        "high": int((machines_df["Status"] == "HIGH RISK").sum()),
        "watch": int((machines_df["Status"] == "WATCH").sum()),
        "mean_probability": float(machines_df["Failure Probability (%)"].mean()),
    }


def history_to_frame(history):
    """Turn the list of history rows into a DataFrame (empty-safe)."""
    return pd.DataFrame(history) if history else pd.DataFrame()


def machine_trajectory(history_df, machine_id):
    """Failure-probability-by-step series for one machine (index = step)."""
    if history_df.empty:
        return pd.Series(dtype=float)
    one = history_df[history_df["Machine"] == machine_id]
    return one.drop_duplicates("Step", keep="last").set_index("Step")["Failure Probability (%)"]


def pivot_trajectories(history_df):
    """Wide table (rows = step, columns = machine) for a multi-line chart."""
    if history_df.empty:
        return pd.DataFrame()
    wide = history_df.pivot_table(
        index="Step", columns="Machine", values="Failure Probability (%)", aggfunc="last")
    return wide.sort_index().ffill()


def load_report(filename):
    """Read a CSV from reports/, or return None if it does not exist."""
    path = REPORTS / filename
    return pd.read_csv(path) if path.exists() else None
