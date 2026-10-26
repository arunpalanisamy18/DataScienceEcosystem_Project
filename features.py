"""Shared feature engineering for the predictive-maintenance model.

The AI4I 2020 failure modes are driven by physical quantities that are
*combinations* of the raw sensors (documented in the dataset description):

* Heat dissipation  -> small gap between process and air temperature
                       together with low rotational speed
* Power             -> torque x angular speed (W), too low or too high
* Overstrain        -> tool wear x torque (min.Nm)

Giving the model these quantities directly (instead of making it rediscover
them from raw columns) is the main modelling improvement in this project.
Only the six raw sensor columns are used as inputs - the failure-mode label
columns (TWF, HDF, PWF, OSF, RNF) are never touched, so there is no leakage.
"""
import numpy as np
import pandas as pd

RAW_FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]
CATEGORICAL = ["Type"]
RAW_NUMERIC = [c for c in RAW_FEATURES if c not in CATEGORICAL]
ENGINEERED = ["Temp difference", "Power", "Overstrain"]
NUMERIC_ALL = RAW_NUMERIC + ENGINEERED


def add_engineered_features(X):
    """Return a copy of X with the three physics-based features added."""
    X = pd.DataFrame(X).copy()
    X["Temp difference"] = X["Process temperature"] - X["Air temperature"]
    # Power (W) = torque (Nm) x angular speed (rad/s)
    X["Power"] = X["Torque"] * X["Rotational speed"] * 2.0 * np.pi / 60.0
    X["Overstrain"] = X["Tool wear"] * X["Torque"]
    return X
