"""Phase 2: choose r = m * g maximizing expected per-unit value of today's bids vs waiting."""
import numpy as np
import pandas as pd

from reserve.config import Params
from reserve.counter_curve import CounterCurve


def price_grid(p: Params) -> np.ndarray:
    return np.round(np.arange(p.grid_lo, p.grid_hi + p.grid_step / 2, p.grid_step), 4)


def expected_value(m, W, xq, wh, g, curve: CounterCurve):
    r = m[:, None] * g[None, :]                      # N, G
    b = (xq * m[:, None])[:, None, :]                # N, 1, Q
    rr = r[:, :, None]                               # N, G, 1
    accept = b >= rr
    pi = curve.pi((rr - b) / b, wh)
    rev = np.where(accept, b, pi * rr)
    sold = np.where(accept, 1.0, pi)
    val = rev + (1.0 - sold) * W[:, None, None]
    return val.mean(axis=2), sold.mean(axis=2), rev.mean(axis=2)


def optimize(df: pd.DataFrame, xq: np.ndarray, curve: CounterCurve, p: Params,
             b0_ratio=None, chunk: int = 5000) -> pd.DataFrame:
    g = price_grid(p)
    G = len(g)
    m = df.m.to_numpy(float)
    W = df.W.to_numpy(float)
    wh = df.wh.astype(str).to_numpy()
    b0 = None if b0_ratio is None else np.asarray(b0_ratio, float)
    n = len(df)
    cols = {k: np.empty(n) for k in ["g_star", "exp_value", "p_sell", "exp_rev"]}
    for s in range(0, n, chunk):
        e = slice(s, min(s + chunk, n))
        V, S, R = expected_value(m[e], W[e], xq[e], wh[e], g, curve)
        feas = np.broadcast_to(g >= p.salvage_floor_pct, V.shape).copy()
        if b0 is not None:
            lo, hi = b0[e, None] * (1 - p.trust_region), b0[e, None] * (1 + p.trust_region)
            feas &= (g[None, :] >= lo) & (g[None, :] <= hi)
        Vm = np.where(feas, V, -np.inf)
        idx = G - 1 - Vm[:, ::-1].argmax(axis=1)     # tie -> highest price
        none = ~feas.any(axis=1)
        if none.any():
            target = b0[e][none] if b0 is not None else np.ones(none.sum())
            idx[none] = np.abs(g[None, :] - target[:, None]).argmin(axis=1)
        rows = np.arange(len(idx))
        cols["g_star"][e] = g[idx]
        cols["exp_value"][e] = V[rows, idx]
        cols["p_sell"][e] = S[rows, idx]
        cols["exp_rev"][e] = R[rows, idx]
    out = pd.DataFrame(cols, index=df.index)
    out["reserve"] = m * out.g_star.to_numpy()
    return out
