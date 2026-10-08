"""Isotonic projection so reserves rise with capacity and along the grade ladder."""
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from reserve.config import CAPACITY_ORDER, GRADE_ORDER

PASSES = [
    (["model", "grade", "wh", "day"], "capacity", CAPACITY_ORDER),
    (["model", "capacity", "wh", "day"], "grade", GRADE_ORDER),
]


def enforce_monotone(df: pd.DataFrame, value_col: str = "reserve", weight_col: str = "nb") -> pd.DataFrame:
    out = df.copy()
    out["reserve_mono"] = out[value_col].astype(float)
    w = out[weight_col].fillna(0.0) + 1.0
    for group_keys, col, order in PASSES:
        rank_of = {v: i for i, v in enumerate(order)}
        sub = out[out[col].isin(order)]
        sub = sub[sub.groupby(group_keys)[col].transform("nunique") >= 2]
        for _, g in sub.groupby(group_keys):
            x = g[col].map(rank_of).to_numpy(float)
            iso = IsotonicRegression(increasing=True)
            out.loc[g.index, "reserve_mono"] = iso.fit(x, g.reserve_mono, sample_weight=w[g.index]).predict(x)
    return out
