"""Value of waiting: depreciation, realization ratio, continuation value W."""
import numpy as np
import pandas as pd

GROUP = ["model", "capacity", "grade"]


def estimate_delta(sales_p: pd.DataFrame, cutoff: pd.Timestamp,
                   min_weeks: int = 8, cap: float = 0.01) -> float:
    s = sales_p[sales_p.day < cutoff]
    if s.empty:
        return 0.0
    s = s.assign(week=(s.day - s.day.min()).dt.days // 7, logp=np.log(s.unit_price))
    wk = s.groupby(GROUP + ["week"], as_index=False).logp.mean()
    slopes = [np.polyfit(g.week, g.logp, 1)[0] / 7.0
              for _, g in wk.groupby(GROUP) if len(g) >= min_weeks]
    if not slopes:
        return 0.0
    return float(np.clip(-np.median(slopes), 0.0, cap))


def estimate_rho(rows: pd.DataFrame) -> float:
    return float(np.median(np.exp(rows.y - rows.log_m)))


def continuation_value(m, tau, delta: float, rho: float, h_pct, salvage_pct: float) -> np.ndarray:
    m = np.asarray(m, float)
    tau = np.asarray(tau, float)
    w = m * rho * (1.0 - delta) ** tau - np.asarray(h_pct, float) * m * tau
    return np.maximum(w, salvage_pct * m)
