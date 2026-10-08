import numpy as np
import pandas as pd

from reserve.floor_model import FLOOR_NUM, FloorModel
from reserve.frame import CAT_COLS


def _bounds(n, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FLOOR_NUM})
    for c in CAT_COLS:
        df[c] = "X"
    df["grade"] = rng.choice(["A", "B"], n)
    df["m"] = 100.0
    floor = np.where(df.grade == "A", 90.0, 80.0)
    b = rng.uniform(60, 110, n)
    acc = b >= floor
    df["lower"] = np.where(acc, np.nan, b)
    df["upper"] = np.where(acc, b, floor * 1.02)
    df["consistent"] = True
    return df


def test_floor_model_recovers_grade_floors():
    fm = FloorModel().fit(_bounds(4000, 0), _bounds(1000, 1))
    test = _bounds(200, 2)
    r = fm.predict_ratio(test)
    assert abs(np.median(r[test.grade == "A"]) - 0.90) < 0.05
    assert abs(np.median(r[test.grade == "B"]) - 0.80) < 0.05
