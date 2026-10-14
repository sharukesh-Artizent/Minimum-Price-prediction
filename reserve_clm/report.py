"""Comparison tables: every method scored the same way on the same rows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bands import BAND_LABELS, BAND_MIDS, band_of
from .baselines import carry_forward_ratio, fixed_ratio, majority_band
from .evaluate import band_metrics, decode_bands, interval_metrics, replay

KEYS3 = ["sku", "wh", "day"]


def method_ratios(df: pd.DataFrame, train_df: pd.DataFrame, runs: list[dict], split_key: str):
    """Reserve ratio (x reference price) per method for the rows of df."""
    fallback = float(train_df.mid_ratio.median())
    maj = majority_band(train_df.true_band.to_numpy())
    ratios = {"B0 carry-forward": carry_forward_ratio(df, fallback),
              "B1 fixed ratio": fixed_ratio(len(df), fallback),
              "B2 majority band": fixed_ratio(len(df), BAND_MIDS[maj])}
    for r in runs:
        ratios[f"CLM seed {r['seed']}"] = BAND_MIDS[decode_bands(r["probs"][split_key])]
    ens = np.mean([r["probs"][split_key] for r in runs], axis=0)
    ratios["CLM ensemble (median band)"] = BAND_MIDS[decode_bands(ens)]
    ratios["CLM ensemble (expected ratio)"] = ens @ BAND_MIDS
    return ratios, ens


def score_table(df: pd.DataFrame, ratios: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, r in ratios.items():
        m = interval_metrics(df.ref_price.to_numpy() * r, df.lo, df.hi, df.mid)
        m.update(band_metrics(band_of(r), df.true_band.to_numpy()))
        rows.append({"method": name, **m})
    t = pd.DataFrame(rows)
    seeds = t[t.method.str.startswith("CLM seed")]
    if len(seeds) > 1:
        mean = seeds.drop(columns="method").mean().to_dict()
        mean.update(method="CLM seeds (mean)", hit_rate_std=float(seeds.interval_hit_rate.std()))
        t = pd.concat([t, pd.DataFrame([mean])], ignore_index=True)
    return t


def replay_table(rep: pd.DataFrame, offers: pd.DataFrame, train_df: pd.DataFrame, runs: list[dict]) -> pd.DataFrame:
    ratios, _ = method_ratios(rep, train_df, runs, "replay")
    rows = []
    for src in ("sku", "family"):
        mask = (rep.ref_source == src).to_numpy()
        if not mask.any():
            continue
        for name, r in ratios.items():
            groups = rep.loc[mask, KEYS3].assign(reserve=rep.ref_price.to_numpy()[mask] * r[mask])
            rows.append({"reference": src, "method": name, **replay(offers, groups)})
    return pd.DataFrame(rows)


def acceptance_table(df: pd.DataFrame, offers: pd.DataFrame, ratios: dict[str, np.ndarray]) -> pd.DataFrame:
    """Daily acceptance on the labelled days themselves (reserve = market price x method ratio)."""
    rows = []
    for name, r in ratios.items():
        groups = df[KEYS3].assign(reserve=df.ref_price.to_numpy() * r)
        rows.append({"method": name, **replay(offers, groups)})
    return pd.DataFrame(rows)


def breakdowns(df: pd.DataFrame, hits: dict[str, np.ndarray]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    top = d.manufacturer.value_counts().index[:6]
    d["brand"] = np.where(d.manufacturer.isin(top), d.manufacturer, "other")
    d["price_tier"] = pd.cut(d.ref_price, [0, 150, 350, np.inf], labels=["<$150", "$150-350", ">=$350"])
    d["bid_history"] = pd.cut(d.n_offers_30, [-1, 9, 29, np.inf], labels=["<10 bids/30d", "10-29", "30+"])
    for name, h in hits.items():
        d[name] = h
    out = []
    for dim in ("brand", "price_tier", "bid_history"):
        g = d.groupby(dim, observed=True)
        t = g[list(hits)].mean()
        t.insert(0, "n", g.size())
        t = t.reset_index().rename(columns={dim: "value"})
        t.insert(0, "dimension", dim)
        t["value"] = t["value"].astype(str)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def prediction_examples(df: pd.DataFrame, probs: np.ndarray, n: int = 8, seed: int = 0) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    idx = np.random.default_rng(seed).choice(len(df), size=min(n, len(df)), replace=False)
    rows = []
    for i in idx:
        r, p = df.iloc[i], probs[i]
        b = decode_bands(p[None])[0]
        top = np.argsort(p)[::-1][:3]
        rows.append({"sku": r.sku, "wh": r.wh, "day": r.day.date(), "market_price": round(r.ref_price, 2),
                     "assurant_floor_interval": f"{r.lo:.0f}-{r.hi:.0f}",
                     "clm_reserve": round(r.ref_price * BAND_MIDS[b], 2),
                     "top_bands": ", ".join(f"{BAND_LABELS[k]} ({p[k]:.0%})" for k in top)})
    return pd.DataFrame(rows)
