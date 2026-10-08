"""M4: P(buyer takes a counter) as a decreasing function of the counter gap (c - b) / b."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass
class CounterCurve:
    grid: np.ndarray
    curves: dict
    max_gap: float

    def pi(self, gap, wh) -> np.ndarray:
        gap = np.asarray(gap, float)
        wh = np.asarray(wh).astype(str)
        out = np.zeros_like(gap)
        for w in np.unique(wh):
            rows = wh == w
            curve = self.curves.get(str(w), self.curves.get(w, self.curves["__all__"]))
            out[rows] = np.interp(gap[rows], self.grid, curve)
        out[(gap < 0) | (gap > self.max_gap)] = 0.0
        return out


def fit_counter_curve(matched: pd.DataFrame, cutoff: pd.Timestamp, recall: pd.Series,
                      min_obs: int = 500, n_grid: int = 61) -> CounterCurve:
    c = matched[(matched.status == "COUNTERED") & (matched.day < cutoff) & matched.counter_price.notna()]
    gap = (c.counter_price - c.offer_price) / c.offer_price
    c, gap = c[gap > 0], gap[gap > 0]
    max_gap = float(np.quantile(gap, 0.99))
    keep = gap <= max_gap
    c, gap = c[keep], gap[keep]
    grid = np.linspace(0.0, max_gap, n_grid)

    def fit(g, y):
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip", y_min=0.0, y_max=1.0)
        return iso.fit(g, y).predict(grid)

    curves = {"__all__": np.clip(fit(gap, c.counter_taken.astype(float)) / recall["__all__"], 0, 1)}
    for w, idx in c.groupby("wh").groups.items():
        if len(idx) >= min_obs:
            r = recall.get(w, recall["__all__"])
            curves[str(w)] = np.clip(fit(gap.loc[idx], c.loc[idx, "counter_taken"].astype(float)) / r, 0, 1)
    return CounterCurve(grid, curves, max_gap)
