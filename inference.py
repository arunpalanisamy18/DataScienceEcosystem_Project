"""Model inference helpers (the saved model is loaded once and cached)."""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

import features  # noqa: F401  (needed so joblib can unpickle the pipeline)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "gradient_boosting_pipeline.joblib"
CONFIG_PATH = ROOT / "models" / "model_config.json"

MODES = {"balanced": "Balanced (best F1)", "high_recall": "High-recall (fewer missed failures)"}


@lru_cache(maxsize=1)
def load_model():
    """Load the saved model pipeline and its configuration (cached)."""
    if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found in models/. Run:  python rigorous_evaluation.py")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as error:  # e.g. scikit-learn version mismatch
        raise RuntimeError(
            "The saved model could not be loaded with the installed scikit-learn "
            f"version ({error}). Re-create it with:  python rigorous_evaluation.py"
        ) from error
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)
    return model, config


def get_threshold(mode="balanced"):
    _, config = load_model()
    return float(config["thresholds"][mode])


def risk_level(probability, threshold):
    if probability >= threshold:
        return "HIGH RISK"
    if probability >= 0.5 * threshold:
        return "WATCH"
    return "LOW RISK"


def predict_batch(readings_list, mode="balanced"):
    """Predict failure probability for many readings (list of dicts) at once."""
    model, config = load_model()
    cols = config["features"]
    for i, row in enumerate(readings_list):
        missing = [c for c in cols if c not in row]
        if missing:
            raise ValueError(f"Row {i} is missing sensor inputs: {missing}")
    frame = pd.DataFrame([{c: row[c] for c in cols} for row in readings_list], columns=cols)
    probabilities = model.predict_proba(frame)[:, 1]
    threshold = get_threshold(mode)
    return [
        {
            "machine_failure_probability": float(p),
            "predicted_failure": int(p >= threshold),
            "risk_threshold": threshold,
            "status": risk_level(float(p), threshold),
        }
        for p in probabilities
    ]


def predict_machine(sensor_readings, mode="balanced"):
    """Predict failure probability from one machine's six sensor readings."""
    return predict_batch([sensor_readings], mode)[0]


if __name__ == "__main__":
    sample = {"Type": "M", "Air temperature": 300.0, "Process temperature": 310.0,
              "Rotational speed": 1500, "Torque": 40.0, "Tool wear": 100}
    result = predict_machine(sample)
    print("Sample readings:", sample)
    print(f"Failure probability: {result['machine_failure_probability']:.2%}")
    print(f"Threshold: {result['risk_threshold']:.2f}  ->  {result['status']}")
