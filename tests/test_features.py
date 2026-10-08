import numpy as np
import pandas as pd

from factories import offers_df, product_df, sales_df
from reserve.config import Params
from reserve.features import build_features


def _data():
    product = product_df([dict(sku="S1"), dict(sku="S2", capacity="256GB")])
    sales = sales_df([dict(sku="S1", day=f"2025-10-{d:02d}", unit_price=100 + d, units=1)
                      for d in range(10, 31)])
    offers = offers_df([dict(sku="S1", day=f"2025-10-{d:02d}", offer_price=95) for d in range(10, 31)])
    return offers, sales, product


def test_point_in_time_no_future_leakage():
    # Query day is after the 14-day warm-up, so the shrinkage prior is identical in both runs.
    offers, sales, product = _data()
    q = pd.DataFrame({"sku": ["S1"], "wh": ["W1"], "day": [pd.Timestamp("2025-10-28")]})
    full = build_features(q, offers, sales, product, Params()).iloc[0]
    cut = pd.Timestamp("2025-10-28")
    part = build_features(q, offers[offers.day < cut], sales[sales.day < cut], product, Params()).iloc[0]
    for c in ["log_m_naive", "tau", "nb", "ub", "units_sold_w", "px_mean_L5"]:
        assert np.isclose(full[c], part[c]), c


def test_tau_age_and_counts():
    offers, sales, product = _data()
    q = pd.DataFrame({"sku": ["S1"], "wh": ["W1"], "day": [pd.Timestamp("2025-10-20")]})
    r = build_features(q, offers, sales, product, Params()).iloc[0]
    assert r.units_sold_w == 10                      # Oct 10..19
    assert np.isclose(r.tau, 28 / 10.5)
    assert r.nb == 10
    assert r.age_days == 10
    assert bool(r.left_censored)
    assert r.dow == pd.Timestamp("2025-10-20").dayofweek


def test_unseen_sku_falls_back_to_parent_level():
    offers, sales, product = _data()
    q = pd.DataFrame({"sku": ["S2"], "wh": ["W1"], "day": [pd.Timestamp("2025-10-25")]})
    r = build_features(q, offers, sales, product, Params()).iloc[0]
    assert np.isnan(r.px_mean_L5)
    assert np.log(100) < r.log_m_naive < np.log(130)  # borrowed from model M1 / category
    assert r.tau == Params().velocity_window_days / 0.5 or r.tau == Params().max_tau_days


def test_attach_product():
    from reserve.features import attach_product
    product = pd.DataFrame([{"sku": "S1", "category": "Phones", "manufacturer": "Apple"}])
    df = pd.DataFrame([{"sku": "S1"}, {"sku": "S2"}])
    out = attach_product(df, product)
    for col in ["category", "manufacturer", "model", "capacity", "grade"]:
        assert col in out.columns
    assert (out[out.sku == "S1"]["category"] == "Phones").all()
    assert (out[out.sku == "S1"]["model"] == "UNK").all()
    assert (out[out.sku == "S2"]["category"] == "UNK").all()

