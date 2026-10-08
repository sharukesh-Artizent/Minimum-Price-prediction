import numpy as np
import pandas as pd

from reserve.siblings import enforce_monotone

DAY = pd.Timestamp("2026-03-01")


def _df(rows):
    df = pd.DataFrame(rows)
    df["wh"], df["day"], df["nb"] = "W1", DAY, 1.0
    return df


def test_capacity_violation_is_pooled():
    df = _df([dict(model="M", grade="A", capacity="128GB", reserve=200.0),
              dict(model="M", grade="A", capacity="256GB", reserve=180.0)])
    out = enforce_monotone(df)
    assert np.allclose(out.reserve_mono, [190.0, 190.0])


def test_grade_violation_is_pooled_and_unknown_grade_untouched():
    df = _df([dict(model="M", grade="B", capacity="128GB", reserve=210.0),
              dict(model="M", grade="A", capacity="128GB", reserve=200.0),
              dict(model="M", grade="REC", capacity="128GB", reserve=50.0)])
    out = enforce_monotone(df).set_index("grade").reserve_mono
    assert np.isclose(out["B"], 205.0) and np.isclose(out["A"], 205.0)
    assert out["REC"] == 50.0


def test_already_monotone_unchanged():
    df = _df([dict(model="M", grade="A", capacity="128GB", reserve=180.0),
              dict(model="M", grade="A", capacity="256GB", reserve=200.0)])
    assert np.allclose(enforce_monotone(df).reserve_mono, [180.0, 200.0])
