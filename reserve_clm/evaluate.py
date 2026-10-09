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
        return {"band_exact": NAN, "band_within_1": NAN, "band_mae": NAN}
    d = np.abs(p - t)
    return {
        "band_exact": float((d == 0).mean()),
        "band_within_1": float((d <= 1).mean()),
        "band_mae": float(d.mean()),
    }


def replay(offers: pd.DataFrame, groups: pd.DataFrame) -> dict:
    """Accept every offer whose bid >= the group's reserve; compare with Assurant's OFFER_STATUS."""
    m = offers.merge(groups[["sku", "wh", "day", "reserve"]], on=["sku", "wh", "day"], how="inner")
    out = {"n_groups": int(len(groups)), "n_offers": int(len(m))}
    if m.empty:
        return out
    pred = (m.price >= m.reserve).to_numpy()
    act = m.accepted.to_numpy(dtype=bool)
    value = (m.price * m.qty).to_numpy()
    out.update(
        agreement=float((pred == act).mean()),
        lost_sales_rate=float((~pred & act).mean()),
        underpricing_rate=float((pred & ~act).mean()),
        pred_accept_rate=float(pred.mean()),
        actual_accept_rate=float(act.mean()),
        accepted_value_pred=float(value[pred].sum()),
        accepted_value_actual=float(value[act].sum()),
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


# Convenience aliases
eval_intervals = interval_metrics
eval_bands = band_metrics
eval_replay = replay
eval_calibration = calibration_table
