"""Hierarchical empirical-Bayes shrinkage of trailing means (coarse -> fine)."""
import numpy as np
import pandas as pd

from reserve.rolling import trailing_sum


def level_trailing_mean(events: pd.DataFrame, value_col: str, weight_col: str,
                        queries: pd.DataFrame, levels: list[list[str]], window_days: int,
                        prefix: str) -> pd.DataFrame:
    out = queries.copy()
    ev = events.assign(_s=events[value_col] * events[weight_col], _w=events[weight_col], _c=1.0)
    for i, keys in enumerate(levels):
        t = trailing_sum(ev, keys, ["_s", "_w", "_c"], out, window_days)
        t[f"{prefix}_mean_L{i}"] = t["_s"] / t["_w"].where(t["_w"] > 0)
        t = t.rename(columns={"_c": f"{prefix}_n_L{i}"}).drop(columns=["_s", "_w"])
        out = out.merge(t, on=keys + ["day"], how="left")
    return out


def shrink(values: list[np.ndarray], counts: list[np.ndarray], k: float, prior: float) -> np.ndarray:
    est = np.full(len(values[0]), float(prior))
    for v, n in zip(values, counts):
        v = np.asarray(v, float)
        n = np.where(np.isnan(v), 0.0, np.asarray(n, float))
        est = (n * np.nan_to_num(v) + k * est) / (n + k)
    return est


def shrink_frame(df: pd.DataFrame, prefix: str, n_levels: int, k: float, prior: float) -> np.ndarray:
    vals = [df[f"{prefix}_mean_L{i}"].to_numpy(float) for i in range(n_levels)]
    cnts = [df[f"{prefix}_n_L{i}"].to_numpy(float) for i in range(n_levels)]
    return shrink(vals, cnts, k, prior)
