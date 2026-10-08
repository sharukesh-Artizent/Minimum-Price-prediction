"""Uplift confidence intervals, Pareto segments and breakdown tables."""
import numpy as np
import pandas as pd

from reserve.replay import summarize

DEV_BINS = [-1e-9, 0.05, 0.10, 0.15, np.inf]
DEV_LABELS = ["<=5%", "5-10%", "10-15%", ">15%"]


def bootstrap_uplift(res_a: pd.DataFrame, res_b: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    da = res_a.groupby("day").value.sum()
    db = res_b.groupby("day").value.sum().reindex(da.index, fill_value=0.0)
    diff = (da - db).to_numpy()
    rng = np.random.default_rng(seed)
    sums = np.array([diff[rng.integers(0, len(diff), len(diff))].sum() for _ in range(n_boot)])
    return {"uplift": float(diff.sum()), "uplift_pct": float(diff.sum() / db.sum()),
            "ci_lo": float(np.quantile(sums, 0.025)), "ci_hi": float(np.quantile(sums, 0.975))}


def pareto_segment(offers: pd.DataFrame, share: float = 0.8) -> pd.DataFrame:
    c = offers.groupby(["sku", "wh"]).size().sort_values(ascending=False)
    cum = c.cumsum() / c.sum()
    in_a = cum.shift(fill_value=0.0) < share
    return pd.DataFrame({"segment": np.where(in_a, "A", "BC")}, index=c.index).reset_index()


def segment_table(results: dict, col: str) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        for seg, part in res.groupby(col, observed=True):
            rows.append({col: seg, "policy": name, **summarize(part)})
    return pd.DataFrame(rows)


def deviation_bucket(reserve, b0) -> pd.Categorical:
    dev = np.abs(np.asarray(reserve, float) / np.asarray(b0, float) - 1.0)
    return pd.cut(dev, DEV_BINS, labels=DEV_LABELS)
