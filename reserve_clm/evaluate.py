"""Metric primitives: interval hit, price error, band accuracy, offer replay, calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd

NAN = float("nan")


def decode_bands(probs: np.ndarray, how: str = "median") -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if how == "argmax":
        return probs.argmax(1)
    return (probs.cumsum(1) >= 0.5).argmax(1)


def interval_metrics(price, lo, hi, mid) -> dict:
    price, lo, hi, mid = (np.asarray(x, dtype=float) for x in (price, lo, hi, mid))
    keys = ["interval_hit_rate", "too_high_rate", "too_low_rate", "mape_vs_mid", "median_ape_vs_mid", "bias_vs_mid"]
    if len(price) == 0:
        return {"n": 0, **{k: NAN for k in keys}}
    rel = (price - mid) / mid
    return {
        "n": int(len(price)),
        "interval_hit_rate": float(((price >= lo) & (price <= hi)).mean()),
        "too_high_rate": float((price > hi).mean()),
        "too_low_rate": float((price < lo).mean()),
        "mape_vs_mid": float(np.abs(rel).mean()),
        "median_ape_vs_mid": float(np.median(np.abs(rel))),
        "bias_vs_mid": float(rel.mean()),
    }


def band_metrics(pred_band, true_band) -> dict:
    p, t = np.asarray(pred_band, dtype=int), np.asarray(true_band, dtype=int)
    if len(p) == 0:
        return {"band_exact": NAN, "band_within_1": NAN, "band_within_3": NAN, "band_within_5": NAN, "band_mae": NAN}
    d = np.abs(p - t)
    return {
        "band_exact": float((d == 0).mean()),
        "band_within_1": float((d <= 1).mean()),
        "band_within_3": float((d <= 3).mean()),
        "band_within_5": float((d <= 5).mean()),
        "band_mae": float(d.mean()),
    }


def replay(offers: pd.DataFrame, groups: pd.DataFrame) -> dict:
    """Use each group's reserve as that day's base price: accept every offer with bid >= reserve.

    Compared with Assurant's OFFER_STATUS per offer and per SKU x warehouse x day.
    """
    keys = ["sku", "wh", "day"]
    m = offers.merge(groups[keys + ["reserve"]], on=keys, how="inner")
    out = {"n_groups": int(len(groups)), "n_offers": int(len(m))}
    if m.empty:
        return out
    m = m.assign(pred=(m.price >= m.reserve).to_numpy(), act=m.accepted.to_numpy(dtype=bool))
    m["agree"] = m.pred == m.act
    pred, act = m.pred.to_numpy(), m.act.to_numpy()
    both = int((pred & act).sum())
    day = m.groupby(keys).agg(pred_n=("pred", "sum"), act_n=("act", "sum"), exact=("agree", "all"))
    out.update(
        agreement=float(m.agree.mean()),
        lost_sales_rate=float((~pred & act).mean()),
        underpricing_rate=float((pred & ~act).mean()),
        pred_accept_rate=float(pred.mean()),
        actual_accept_rate=float(act.mean()),
        accept_precision=both / pred.sum() if pred.any() else NAN,
        accept_recall=both / act.sum() if act.any() else NAN,
        n_days=int(len(day)),
        day_exact_match=float(day.exact.mean()),
        day_count_mae=float((day.pred_n - day.act_n).abs().mean()),
        accepted_per_day_pred=float(day.pred_n.mean()),
        accepted_per_day_actual=float(day.act_n.mean()),
    )
    return out


def calibration_table(conf, hit, n_bins: int = 5) -> pd.DataFrame:
    conf, hit = np.asarray(conf, dtype=float), np.asarray(hit, dtype=float)
    if len(conf) == 0:
        return pd.DataFrame(columns=["bin", "n", "mean_confidence", "interval_hit_rate"])
    d = pd.DataFrame({"conf": conf, "hit": hit, "bin": pd.qcut(conf, n_bins, duplicates="drop")})
    g = d.groupby("bin", observed=True)
    t = pd.DataFrame({"n": g.size(), "mean_confidence": g.conf.mean(), "interval_hit_rate": g.hit.mean()})
    t = t.reset_index()
    t["bin"] = t["bin"].astype(str)
    return t


def band_ece(probs, true_band, n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float)
    if len(probs) == 0:
        return NAN
    conf = probs.max(1)
    correct = probs.argmax(1) == np.asarray(true_band, dtype=int)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
