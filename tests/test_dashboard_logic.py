import pandas as pd

from dashboard_logic import (
    history_to_frame, load_report, machine_trajectory, make_machine_dataframe,
    pivot_trajectories, summarise_factory,
)
from simulation.factory_monitor import FactoryMonitor


def test_machine_table_and_summary():
    results = FactoryMonitor().monitor_all(advance=False)
    table = make_machine_dataframe(results)
    assert len(table) == 5 and "Failure Probability (%)" in table.columns
    s = summarise_factory(table)
    assert s["total"] == 5 and s["high"] + s["watch"] <= 5


def test_empty_history_is_handled():
    empty = history_to_frame([])
    assert empty.empty and pivot_trajectories(empty).empty
    assert machine_trajectory(empty, "MCH-001").empty


def test_trajectory_and_pivot_shapes():
    history, _ = FactoryMonitor().run_steps(7)
    frame = history_to_frame(history)
    assert len(machine_trajectory(frame, "MCH-003")) == 7
    wide = pivot_trajectories(frame)
    assert wide.shape == (7, 5)


def test_missing_report_returns_none_and_existing_loads():
    assert load_report("does_not_exist.csv") is None
    assert isinstance(load_report("final_test_metrics.csv"), pd.DataFrame)
