"""Guard-rails: the shipped model must keep meeting its documented quality bar."""
from dashboard_logic import load_report


def test_final_metrics_beat_baseline_and_documented_floor():
    final = load_report("final_test_metrics.csv").set_index("Operating mode")
    balanced = final.loc["Balanced (max F1)"]
    baseline = final.loc["BASELINE: untuned GB, raw features"]
    assert balanced["F1"] > baseline["F1"]
    assert balanced["PR-AUC"] > baseline["PR-AUC"]
    assert balanced["Recall"] >= 0.80 and balanced["Precision"] >= 0.90


def test_no_leakage_columns_in_model_inputs():
    import json
    from inference import CONFIG_PATH
    features = json.loads(CONFIG_PATH.read_text())["features"]
    assert not {"TWF", "HDF", "PWF", "OSF", "RNF", "Machine failure"} & set(features)
