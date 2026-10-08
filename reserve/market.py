"""Per-node depreciation and stock inputs taken from historical sales."""
import numpy as np
import pandas as pd

from reserve.observations import attach_nodes, window


def daily_depreciation(sales: pd.DataFrame, nodes: pd.DataFrame, default: float,
                       min_rows: int = 30, cap: float = 0.01) -> np.ndarray:
    """Family-level daily price decline from a within-node regression of log price on day."""
    idx = attach_nodes(sales, nodes)
    s = sales[idx >= 0].copy()
    if s.empty:
        return np.full(len(nodes), default)
    s["node"] = idx[idx >= 0]
    s["family"] = nodes["family"].to_numpy()[s["node"]]
    s["t"] = (s["date"] - s["date"].min()).dt.days.astype(float)
    s["lp"] = np.log(s["unit_price"])
    s["t_c"] = s["t"] - s.groupby("node")["t"].transform("mean")
    s["lp_c"] = s["lp"] - s.groupby("node")["lp"].transform("mean")
    s["xy"] = s["t_c"] * s["lp_c"]
    s["xx"] = s["t_c"] ** 2
    agg = s.groupby("family")[["xy", "xx"]].sum()
    rows = s.groupby("family").size()
    slope = agg["xy"] / agg["xx"].where(agg["xx"] > 0)
    delta = (-slope).clip(lower=0.0, upper=cap).where(rows >= min_rows)
    return nodes["family"].map(delta).fillna(default).to_numpy(dtype=float)


def proxy_inventory(sales: pd.DataFrame, nodes: pd.DataFrame, start, end) -> np.ndarray:
    """Units Assurant actually sold per node in [start, end): stand-in for stock to clear.

    The Inventory extract (Jul-Aug 2025) does not overlap the offer history, so this
    proxy is used until a full inventory extract arrives.
    """
    w = window(sales, start, end)
    idx = attach_nodes(w, nodes)
    keep = idx >= 0
    return np.bincount(idx[keep], weights=w["qty"].to_numpy(dtype=float)[keep],
                       minlength=len(nodes))
