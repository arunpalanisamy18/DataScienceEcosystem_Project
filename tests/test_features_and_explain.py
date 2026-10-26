import math

import pandas as pd

from explain import failure_mode_warnings
from features import add_engineered_features

R = {"Type": "L", "Air temperature": 300.0, "Process temperature": 310.0,
     "Rotational speed": 1500.0, "Torque": 40.0, "Tool wear": 100.0}


def test_engineered_features_are_correct_and_do_not_mutate_input():
    frame = pd.DataFrame([R])
    out = add_engineered_features(frame)
    assert out.loc[0, "Temp difference"] == 10.0
    assert math.isclose(out.loc[0, "Power"], 40 * 1500 * 2 * math.pi / 60)
    assert out.loc[0, "Overstrain"] == 4000.0
    assert "Power" not in frame.columns


def test_no_warnings_for_normal_machine():
    assert failure_mode_warnings(R) == []


def test_each_failure_mode_rule_triggers():
    codes = lambda **kw: {c for c, _ in failure_mode_warnings({**R, **kw})}
    assert "TWF" in codes(**{"Tool wear": 210.0})
    assert "HDF" in codes(**{"Process temperature": 308.0, "Rotational speed": 1300.0})
    assert "PWF" in codes(**{"Torque": 70.0, "Rotational speed": 1500.0})
    assert "OSF" in codes(**{"Tool wear": 200.0, "Torque": 60.0})


def test_overstrain_limit_depends_on_machine_type():
    heavy = {**R, "Tool wear": 200.0, "Torque": 58.0}   # product 11,600
    assert any(c == "OSF" for c, _ in failure_mode_warnings({**heavy, "Type": "L"}))
    assert not any(c == "OSF" for c, _ in failure_mode_warnings({**heavy, "Type": "H"}))
