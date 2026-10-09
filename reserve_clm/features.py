"""Features for each target SKU x warehouse x day, using only days strictly before it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MIN_REF_ACCEPTS, REF_WINDOW_DAYS, TREND_WINDOW_DAYS

KEYS = ["sku", "wh"]
GKEYS = ["sku", "wh", "day"]


def lagged_join(events: pd.DataFrame, targets: pd.DataFrame, keys: list[str], window: int) -> pd.DataFrame:
    """Event rows from days d-window ... d-1 for every target (keys, d); `day` is the target day."""
    t = targets[keys + ["day"]].dropna(subset=keys).drop_duplicates()
    parts = []
    for lag in range(1, window + 1):
        e = events.assign(day=events["day"] + pd.Timedelta(days=lag), lag=lag)
        parts.append(t.merge(e, on=keys + ["day"], how="inner"))
    if not parts:
        cols = list(dict.fromkeys(keys + ["day"] + list(events.columns) + ["lag"]))
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


def _window_ref(daily: pd.DataFrame, t: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    j = lagged_join(daily, t, keys, REF_WINDOW_DAYS)
    gk = keys + ["day"]
    g = j.groupby(gk)
    f = pd.DataFrame({"ref30": g.med.median(), "ref_n": g.cnt.sum()})
    f["ref7"] = j[j.lag <= TREND_WINDOW_DAYS].groupby(gk).med.median()
    return f.reset_index()


def reference_prices(offers: pd.DataFrame, products: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """30-day median of daily median accepted prices: SKU level, else product family, else none."""
    fam = products[KEYS + ["family"]].drop_duplicates(subset=KEYS)
    acc = offers[offers["accepted"].astype(bool)].merge(fam, on=KEYS, how="left")
    sku_daily = acc.groupby(["sku", "day"]).price.agg(med="median", cnt="size").reset_index()
    fam_daily = (acc.dropna(subset=["family"]).groupby(["family", "day"]).price
                 .agg(med="median", cnt="size").reset_index())
    tt = targets[GKEYS].drop_duplicates().merge(fam, on=KEYS, how="left")
    s = _window_ref(sku_daily, tt, ["sku"]).rename(columns={"ref30": "ref30_sku", "ref_n": "ref_n_sku", "ref7": "ref7_sku"})
    f = _window_ref(fam_daily, tt, ["family"]).rename(columns={"ref30": "ref30_fam", "ref_n": "ref_n_fam", "ref7": "ref7_fam"})
    out = tt.merge(s, on=["sku", "day"], how="left").merge(f, on=["family", "day"], how="left")
    use_sku = (out.ref_n_sku.fillna(0) >= MIN_REF_ACCEPTS).to_numpy()
    use_fam = ~use_sku & (out.ref_n_fam.fillna(0) >= MIN_REF_ACCEPTS).to_numpy()
    out["ref_source"] = np.where(use_sku, "sku", np.where(use_fam, "family", "none"))
    out["ref_price"] = np.where(use_sku, out.ref30_sku, np.where(use_fam, out.ref30_fam, np.nan))
    out["ref_n"] = np.where(use_sku, out.ref_n_sku, np.where(use_fam, out.ref_n_fam, 0.0))
    ref7 = np.where(use_sku, out.ref7_sku, np.where(use_fam, out.ref7_fam, np.nan))
    out["ref_trend"] = ref7 / out["ref_price"]
    return out[GKEYS + ["family", "ref_source", "ref_price", "ref_n", "ref_trend"]]


def bid_features(offers: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    daily = offers.groupby(GKEYS).agg(n=("price", "size"), qty=("qty", "sum"), acc=("accepted", "sum"),
                                      med=("price", "median"), top=("price", "max")).reset_index()
    j = lagged_join(daily, t, KEYS, REF_WINDOW_DAYS)
    g = j.groupby(GKEYS)
    f = pd.DataFrame({"n_offers_30": g.n.sum(), "units_req_30": g.qty.sum(), "acc_30": g.acc.sum(),
                      "bid_med_30": g.med.median(), "bid_top_30": g.top.max()})
    f["n_offers_7"] = j[j.lag <= TREND_WINDOW_DAYS].groupby(GKEYS).n.sum()
    bidders = offers[GKEYS + ["buyer_id"]].drop_duplicates()
    f["n_bidders_30"] = lagged_join(bidders, t, KEYS, REF_WINDOW_DAYS).groupby(GKEYS).buyer_id.nunique()
    out = t[GKEYS].merge(f.reset_index(), on=GKEYS, how="left")
    counts = ["n_offers_30", "units_req_30", "acc_30", "n_offers_7", "n_bidders_30"]
    out[counts] = out[counts].fillna(0)
    return out


def _last_before(events: pd.DataFrame, t: pd.DataFrame, name: str) -> pd.DataFrame:
    """Most recent event strictly before each target day (same sku, wh); event day renamed to `name`."""
    r = events.rename(columns={"day": name}).dropna(subset=[name]).sort_values(name)
    left = t[GKEYS].sort_values("day")
    if r.empty or left.empty:
        res = left.copy()
        for col in r.columns:
            if col not in KEYS and col != name:
                res[col] = np.nan
        res[name] = pd.NaT
        return res
    r[name] = pd.to_datetime(r[name]).astype(left["day"].dtype)
    return pd.merge_asof(left, r, left_on="day", right_on=name, by=KEYS,
                         allow_exact_matches=False, direction="backward")


def sales_features(sales: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    daily = sales.groupby(GKEYS, as_index=False).units.sum()
    j = lagged_join(daily, t, KEYS, REF_WINDOW_DAYS)
    f = j.groupby(GKEYS).units.sum().rename("units_sold_30").to_frame()
    f["units_sold_7"] = j[j.lag <= TREND_WINDOW_DAYS].groupby(GKEYS).units.sum()
    stock = sales[sales.order_type == "STOCKLIST"].groupby(GKEYS, as_index=False).unit_price.median()
    f["sale_price_med_30"] = lagged_join(stock, t, KEYS, REF_WINDOW_DAYS).groupby(GKEYS).unit_price.median()
    out = t[GKEYS].merge(f.reset_index(), on=GKEYS, how="left")
    out[["units_sold_30", "units_sold_7"]] = out[["units_sold_30", "units_sold_7"]].fillna(0)
    last = _last_before(daily[GKEYS], t, "last_sale_day")
    out = out.merge(last, on=GKEYS, how="left")
    out["days_since_last_sale"] = (out.day - out.last_sale_day).dt.days
    return out.drop(columns=["last_sale_day"])


def last_floor_features(floors: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    ev = floors[GKEYS + ["lo", "hi"]].rename(columns={"lo": "last_lo", "hi": "last_hi"})
    m = _last_before(ev, t, "last_floor_day")
    m["days_since_last_floor"] = (m.day - m.last_floor_day).dt.days
    return m.drop(columns=["last_floor_day"])


FEATURE_COLS = [
    "sku", "wh", "day",
    "family", "ref_source", "ref_price", "ref_n", "ref_trend",
    "n_offers_30", "units_req_30", "acc_30", "bid_med_30", "bid_top_30", "n_offers_7",
    "n_bidders_30", "units_sold_30", "units_sold_7", "sale_price_med_30",
    "days_since_last_sale", "last_lo", "last_hi", "days_since_last_floor",
    "manufacturer", "model", "capacity", "grade", "color", "carrier", "lock_status", "category",
    "bid_med_ratio_30", "bid_top_ratio_30", "sale_price_ratio_30",
    "last_lo_ratio", "last_hi_ratio", "accept_share_30",
]


def build_features(offers: pd.DataFrame, sales: pd.DataFrame, products: pd.DataFrame,
                   floors: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    t = targets[GKEYS].drop_duplicates().reset_index(drop=True)
    out = t.merge(reference_prices(offers, products, t), on=GKEYS, how="left")
    out = out.merge(bid_features(offers, t), on=GKEYS, how="left")
    out = out.merge(sales_features(sales, t), on=GKEYS, how="left")
    out = out.merge(last_floor_features(floors, t), on=GKEYS, how="left")
    out = out.merge(products.drop(columns=["family"], errors="ignore"), on=KEYS, how="left")
    for src, dst in [("bid_med_30", "bid_med_ratio_30"), ("bid_top_30", "bid_top_ratio_30"),
                     ("sale_price_med_30", "sale_price_ratio_30"), ("last_lo", "last_lo_ratio"),
                     ("last_hi", "last_hi_ratio")]:
        out[dst] = out[src] / out["ref_price"]
    out["accept_share_30"] = out.acc_30 / out.n_offers_30.where(out.n_offers_30 > 0)
    cols = [c for c in FEATURE_COLS if c in out.columns] + [c for c in out.columns if c not in FEATURE_COLS]
    return out[cols]
