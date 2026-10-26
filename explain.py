"""Human-readable failure-mode warnings.

The thresholds below are the failure rules published in the UCI AI4I 2020
dataset description. They are used ONLY to explain a prediction to the
operator on the dashboard - they are never given to the model as labels.
"""
import math

OVERSTRAIN_LIMIT = {"L": 11000.0, "M": 12000.0, "H": 13000.0}


def power_watts(torque, rpm):
    return torque * rpm * 2.0 * math.pi / 60.0


def failure_mode_warnings(readings):
    """Return a list of (short code, plain-English warning) for one machine."""
    warnings = []
    wear = readings["Tool wear"]
    torque = readings["Torque"]
    rpm = readings["Rotational speed"]
    diff = readings["Process temperature"] - readings["Air temperature"]
    power = power_watts(torque, rpm)
    limit = OVERSTRAIN_LIMIT.get(readings["Type"], 12000.0)

    if wear >= 200:
        warnings.append(("TWF", f"Tool wear is {wear:.0f} min (>= 200): tool replacement due."))
    if diff < 8.6 and rpm < 1380:
        warnings.append(("HDF", f"Heat dissipation: temperature gap {diff:.1f} K (< 8.6 K) at {rpm:.0f} rpm (< 1380)."))
    if power < 3500 or power > 9000:
        warnings.append(("PWF", f"Power {power:,.0f} W is outside the safe 3,500-9,000 W band."))
    if wear * torque > limit:
        warnings.append(("OSF", f"Overstrain: wear x torque = {wear * torque:,.0f} (limit {limit:,.0f} for Type {readings['Type']})."))
    return warnings
