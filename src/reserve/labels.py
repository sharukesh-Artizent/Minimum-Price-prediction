"""Labels derived from Assurant's decisions (OFFER_STATUS) and realized STOCKLIST sales."""
import numpy as np
import pandas as pd

from reserve.config import DECISION_STATUSES

MIN_WH_OBS = 500


def floor_bounds(offers: pd.DataFrame) -> pd.DataFrame:
    """Per SKU x WH x day, the interval (lower, upper] that contains Assurant's implicit floor."""
    o = offers[offers.status.isin(DECISION_STATUSES)]
    lower = o.offer_price.where(o.status.isin(["COUNTERED", "NOT_ACCEPTED"]))
    upper = o.offer_price.where(o.status.eq("ACCEPTED"))
    upper = upper.fillna(o.counter_price.where(o.status.eq("COUNTERED")))
    t = pd.DataFrame({"sku": o.sku, "wh": o.wh, "day": o.day, "lower": lower, "upper": upper})
    g = t.groupby(["sku", "wh", "day"], as_index=False).agg(
        lower=("lower", "max"), upper=("upper", "min"), n_offers=("lower", "size"))
    g["consistent"] = g.lower.isna() | g.upper.isna() | (g.lower < g.upper)
    return g


def match_offers_to_sales(offers: pd.DataFrame, sales: pd.DataFrame,
                          window_days: int, price_tol: float) -> pd.DataFrame:
    """Link each offer to the buyer's next sale of the same SKU within the window."""
    o = offers.sort_values("ts")
    s = (sales[["buyer_id", "sku", "ts", "unit_price"]]
         .rename(columns={"ts": "sale_ts", "unit_price": "sale_price"})
         .sort_values("sale_ts"))
    m = pd.merge_asof(o, s, left_on="ts", right_on="sale_ts", by=["buyer_id", "sku"],
                      direction="forward", tolerance=pd.Timedelta(days=window_days))
    is_counter = m.status.eq("COUNTERED") & m.counter_price.notna()
    m["ref_price"] = m.counter_price.where(is_counter, m.offer_price)
    m["sale_match"] = m.sale_ts.notna() & ((m.sale_price - m.ref_price).abs() / m.ref_price <= price_tol)
    m["counter_taken"] = is_counter & m.sale_match
    return m.drop(columns=["sale_ts", "sale_price"]).reset_index(drop=True)


def match_recall(matched: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Share of ACCEPTED offers before cutoff that link to a sale; per warehouse with global fallback."""
    acc = matched[(matched.status == "ACCEPTED") & (matched.day < cutoff)]
    glob = float(acc.sale_match.mean())
    by_wh = acc.groupby("wh").sale_match.agg(["mean", "size"])
    rate = pd.Series(np.where(by_wh["size"] >= MIN_WH_OBS, by_wh["mean"], glob), index=by_wh.index)
    rate["__all__"] = glob
    return rate.clip(lower=0.05)
