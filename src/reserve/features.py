"""One point-in-time feature row per SKU x WH x day."""
import numpy as np
import pandas as pd

from reserve.config import DATA_START, LEVELS, WARMUP_DAYS, Params
from reserve.rolling import trailing_sum
from reserve.shrink import level_trailing_mean, shrink_frame

PRODUCT_COLS = ["category", "manufacturer", "model", "capacity", "grade"]


def attach_product(df: pd.DataFrame, product: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(product, on="sku", how="left")
    for c in PRODUCT_COLS:
        if c not in out.columns:
            out[c] = "UNK"
    out[PRODUCT_COLS] = out[PRODUCT_COLS].fillna("UNK")
    return out


def _prior(s: pd.DataFrame) -> float:
    early = s[s.day < pd.Timestamp(DATA_START) + pd.Timedelta(days=WARMUP_DAYS)]
    base = early if len(early) else s
    return float(np.average(base.logp, weights=base.units))


def build_features(queries: pd.DataFrame, offers: pd.DataFrame, sales: pd.DataFrame,
                   product: pd.DataFrame, p: Params) -> pd.DataFrame:
    keys = ["sku", "wh", "day"]
    q = attach_product(queries[keys].drop_duplicates().reset_index(drop=True), product)
    s = attach_product(sales, product).assign(logp=lambda d: np.log(d.unit_price))

    q = level_trailing_mean(s, "logp", "units", q, LEVELS, p.price_window_days, "px")
    q["log_m_naive"] = shrink_frame(q, "px", len(LEVELS), p.shrink_k, _prior(s))

    vel = trailing_sum(s.assign(units_sold_w=s.units), ["sku", "wh"], ["units_sold_w"],
                       q, p.velocity_window_days)
    q = q.merge(vel, on=keys, how="left")
    q["tau"] = np.clip(p.velocity_window_days / (q.units_sold_w + 0.5), 1.0, p.max_tau_days)

    ob = trailing_sum(offers.assign(nb=1.0, ub=offers.qty), ["sku", "wh"], ["nb", "ub"],
                      q, p.velocity_window_days)
    q = q.merge(ob, on=keys, how="left")

    first = (pd.concat([offers[keys], sales[keys]])
             .groupby(["sku", "wh"], as_index=False).day.min()
             .rename(columns={"day": "first_day"}))
    q = q.merge(first, on=["sku", "wh"], how="left")
    q["age_days"] = (q.day - q.first_day).dt.days.clip(lower=0).fillna(0).astype(float)
    q["left_censored"] = (q.first_day <= pd.Timestamp(DATA_START)).fillna(False).astype(bool)
    q["dow"] = q.day.dt.dayofweek
    return q.drop(columns="first_day")
