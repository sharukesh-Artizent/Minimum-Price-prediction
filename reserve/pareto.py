"""Pareto (A/B/C) tiers by offer volume and the flexible per-node weight."""
import numpy as np
import pandas as pd


def pareto_tiers(counts: pd.Series, a_share: float = 0.80, b_share: float = 0.95) -> pd.DataFrame:
    s = counts.sort_values(ascending=False, kind="mergesort")
    cum = s.cumsum() / s.sum()
    before = cum.shift(fill_value=0.0)
    tier = np.where(before < a_share, "A", np.where(before < b_share, "B", "C"))
    return pd.DataFrame({"n": s, "cum_share": cum, "tier": tier}, index=s.index)


def node_tiers(counts: np.ndarray, a_share: float = 0.80, b_share: float = 0.95) -> np.ndarray:
    tiers = np.full(len(counts), "unseen", dtype=object)
    seen = np.flatnonzero(counts > 0)
    if seen.size:
        t = pareto_tiers(pd.Series(counts[seen], index=seen), a_share, b_share)
        tiers[t.index.to_numpy()] = t["tier"].to_numpy()
    return tiers


def pareto_weight(n, k: float) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    return n / (n + k)
