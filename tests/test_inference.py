import pytest

from inference import get_threshold, load_model, predict_batch, predict_machine, risk_level

NORMAL = {"Type": "M", "Air temperature": 300.0, "Process temperature": 310.0,
          "Rotational speed": 1500, "Torque": 40.0, "Tool wear": 100}
OVERSTRAIN = {"Type": "L", "Air temperature": 300.0, "Process temperature": 310.0,
              "Rotational speed": 1500, "Torque": 65.0, "Tool wear": 220}


def test_model_and_config_load():
    model, config = load_model()
    assert config["features"][0] == "Type"
    assert 0 < config["thresholds"]["high_recall"] < config["thresholds"]["balanced"] < 1


def test_normal_machine_is_low_risk():
    r = predict_machine(NORMAL)
    assert r["status"] == "LOW RISK" and r["predicted_failure"] == 0
    assert 0 <= r["machine_failure_probability"] < 0.05


def test_overstrained_machine_is_high_risk():
    r = predict_machine(OVERSTRAIN)
    assert r["status"] == "HIGH RISK" and r["predicted_failure"] == 1


def test_batch_matches_single_predictions():
    batch = predict_batch([NORMAL, OVERSTRAIN])
    assert batch[0] == predict_machine(NORMAL) and batch[1] == predict_machine(OVERSTRAIN)


def test_high_recall_mode_is_more_sensitive():
    assert get_threshold("high_recall") < get_threshold("balanced")


def test_missing_input_is_reported():
    bad = dict(NORMAL)
    bad.pop("Torque")
    with pytest.raises(ValueError):
        predict_machine(bad)


def test_risk_level_boundaries():
    assert risk_level(0.5, 0.4) == "HIGH RISK"
    assert risk_level(0.25, 0.4) == "WATCH"
    assert risk_level(0.05, 0.4) == "LOW RISK"
