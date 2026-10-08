"""Replay historical bids under a reserve policy. Accept if bid >= r, else counter at r."""
import numpy as np
import pandas as pd

from reserve.config import Params
from reserve.counter_curve import CounterCurve
from reserve.optimizer import price_grid


def _finish(ev: pd.DataFrame, reserve, sold, rev, decision) -> pd.DataFrame:
    unsold = ev.qty.to_numpy(float) - sold
    return ev.assign(reserve=reserve, sold=sold, revenue=rev,
                     value=rev + unsold * ev.W.to_numpy(float), decision=decision)


def replay_policy(ev: pd.DataFrame, reserve, curve: CounterCurve) -> pd.DataFrame:
    b = ev.offer_price.to_numpy(float)
    q = ev.qty.to_numpy(float)
    r = np.asarray(reserve, float)
    accept = b >= r
    gap = (r - b) / b
    pi = curve.pi(gap, ev.wh.astype(str).to_numpy())
    sold = q * np.where(accept, 1.0, pi)
    rev = q * np.where(accept, b, pi * r)
    decision = np.where(accept, "ACCEPTED", np.where(gap <= curve.max_gap, "COUNTERED", "NOT_ACCEPTED"))
    return _finish(ev, r, sold, rev, decision)


def replay_actual(ev: pd.DataFrame, curve: CounterCurve) -> pd.DataFrame:
    b = ev.offer_price.to_numpy(float)
    q = ev.qty.to_numpy(float)
    c = ev.counter_price.to_numpy(float)
    st = ev.status.to_numpy()
    is_c = st == "COUNTERED"
    gap = np.where(is_c & ~np.isnan(c), (c - b) / b, -1.0)
    pi = curve.pi(gap, ev.wh.astype(str).to_numpy())
    sold = q * np.select([st == "ACCEPTED", is_c], [1.0, pi], 0.0)
    rev = q * np.select([st == "ACCEPTED", is_c], [b, pi * np.nan_to_num(c)], 0.0)
    return _finish(ev, np.nan, sold, rev, st)


def summarize(res: pd.DataFrame) -> dict:
    rev, sold, bid = res.revenue.sum(), res.sold.sum(), res.qty.sum()
    denom_m = (res.sold * res.m).sum()
    realization = float(rev / denom_m) if denom_m > 0 else np.nan
    sell_through = float(sold / bid) if bid > 0 else 0.0
    return {
        "revenue": float(rev),
        "units_sold": float(sold),
        "units_bid": float(bid),
        "sell_through": sell_through,
        "realization": realization,
        "total_value": float(res.value.sum()),
        "n_offers": int(len(res)),
        "accept_rate": float((res.decision == "ACCEPTED").mean()),
        "counter_rate": float((res.decision == "COUNTERED").mean()),
        "reject_rate": float((res.decision == "NOT_ACCEPTED").mean()),
    }


def hindsight_reserve(ev: pd.DataFrame, curve: CounterCurve, p: Params) -> pd.DataFrame:
    keys = ["sku", "wh", "day"]
    g = price_grid(p)
    per_g = []
    for gi in g:
        v = replay_policy(ev, ev.m.to_numpy(float) * gi, curve)
        per_g.append(v.groupby(keys).value.sum().rename(gi))
    tab = pd.concat(per_g, axis=1)
    best = tab.columns.to_numpy()[len(g) - 1 - tab.to_numpy()[:, ::-1].argmax(axis=1)]
    out = tab.index.to_frame(index=False).assign(g_oracle=best)
    m = ev.groupby(keys).m.first().reset_index()
    out = out.merge(m, on=keys)
    out["reserve_oracle"] = out.m * out.g_oracle
    return out[keys + ["reserve_oracle"]]
