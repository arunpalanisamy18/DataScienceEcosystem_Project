from simulation.factory_monitor import FactoryMonitor


def test_monitor_all_covers_every_machine_with_valid_probabilities():
    results = FactoryMonitor().monitor_all(advance=False)
    assert len(results) == 5
    for r in results.values():
        assert 0 <= r["machine_failure_probability"] <= 1
        assert r["status"] in {"LOW RISK", "WATCH", "HIGH RISK"}


def test_run_steps_records_history_for_requested_machine():
    m = FactoryMonitor()
    history, latest = m.run_steps(10, "MCH-001")
    assert len(history) == 10 and {h["Machine"] for h in history} == {"MCH-001"}
    assert [h["Step"] for h in history] == list(range(1, 11))
    assert set(latest) == set(m.factory.list_machines())


def test_run_steps_all_machines():
    history, _ = FactoryMonitor().run_steps(3)
    assert len(history) == 15


def test_overload_scenario_eventually_raises_high_risk_alert():
    m = FactoryMonitor()
    m.set_conditions("MCH-001", temperature_stress=0, load_stress=30, wear_rate=5)
    history, _ = m.run_steps(60, "MCH-001")
    assert any(h["Status"] == "HIGH RISK" for h in history)


def test_replacing_a_worn_tool_lowers_the_risk():
    m = FactoryMonitor()
    m.set_conditions("MCH-001", load_stress=10, wear_rate=5)
    m.run_steps(80, "MCH-001")
    before = m.monitor_one("MCH-001")["machine_failure_probability"]
    m.replace_tool("MCH-001")
    after = m.monitor_one("MCH-001")["machine_failure_probability"]
    assert before > 0.3 and after < before


def test_light_duty_stays_low_risk():
    m = FactoryMonitor()
    m.set_conditions("MCH-002", temperature_stress=-3, load_stress=-8, wear_rate=0.5)
    history, _ = m.run_steps(100, "MCH-002")
    assert all(h["Status"] == "LOW RISK" for h in history)


def test_sensitivity_mode_changes_threshold():
    a = FactoryMonitor("balanced").monitor_one("MCH-001")["risk_threshold"]
    b = FactoryMonitor("high_recall").monitor_one("MCH-001")["risk_threshold"]
    assert b < a
