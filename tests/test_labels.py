import numpy as np
import pandas as pd

from factories import offers_df, sales_df
from reserve.labels import floor_bounds, match_offers_to_sales, match_recall


def test_floor_bounds_interval_and_consistency():
    o = offers_df([
        dict(sku="S", day="2026-01-01", offer_price=100, status="ACCEPTED"),
        dict(sku="S", day="2026-01-01", offer_price=95, status="ACCEPTED"),
        dict(sku="S", day="2026-01-01", offer_price=80, status="NOT_ACCEPTED"),
        dict(sku="S", day="2026-01-01", offer_price=85, status="COUNTERED", counter_price=92),
        dict(sku="S", day="2026-01-01", offer_price=10, status="PENDING_ACCEPTANCE"),
        dict(sku="T", day="2026-01-01", offer_price=90, status="ACCEPTED"),
        dict(sku="T", day="2026-01-01", offer_price=95, status="NOT_ACCEPTED"),
    ])
    b = floor_bounds(o).set_index("sku")
    assert b.loc["S", "lower"] == 85
    assert b.loc["S", "upper"] == 92
    assert bool(b.loc["S", "consistent"])
    assert b.loc["S", "n_offers"] == 4
    assert not bool(b.loc["T", "consistent"])


def test_match_offers_to_sales():
    o = offers_df([
        dict(buyer_id="b1", sku="S", day="2026-01-01", offer_price=100, status="ACCEPTED"),
        dict(buyer_id="b1", sku="S2", day="2026-01-01", offer_price=100, counter_price=104, status="COUNTERED"),
        dict(buyer_id="b2", sku="S", day="2026-01-01", offer_price=100, status="ACCEPTED"),
        dict(buyer_id="b3", sku="S", day="2026-01-01", offer_price=100, status="ACCEPTED"),
        dict(buyer_id="b4", sku="S", day="2026-01-01", offer_price=100, status="NOT_ACCEPTED"),
    ])
    s = sales_df([
        dict(buyer_id="b1", sku="S", day="2026-01-02", unit_price=100),
        dict(buyer_id="b1", sku="S2", day="2026-01-01", unit_price=104),
        dict(buyer_id="b2", sku="S", day="2026-01-06", unit_price=100),
        dict(buyer_id="b3", sku="S", day="2026-01-02", unit_price=90),
    ])
    m = match_offers_to_sales(o, s, window_days=3, price_tol=0.01).set_index("buyer_id")
    m = m.reset_index().set_index(["buyer_id", "sku"])
    assert bool(m.loc[("b1", "S"), "sale_match"])
    assert bool(m.loc[("b1", "S2"), "sale_match"])
    assert bool(m.loc[("b1", "S2"), "counter_taken"])
    assert m.loc[("b1", "S2"), "ref_price"] == 104
    assert not bool(m.loc[("b2", "S"), "sale_match"])      # outside 3-day window
    assert not bool(m.loc[("b3", "S"), "sale_match"])      # price off by 10%
    assert not bool(m.loc[("b4", "S"), "sale_match"])      # no sale
    assert not bool(m.loc[("b1", "S"), "counter_taken"])   # accepted, not countered


def test_match_recall_per_wh_with_global_fallback():
    rows = ([dict(wh="W1", status="ACCEPTED", sale_match=i < 300) for i in range(600)]
            + [dict(wh="W2", status="ACCEPTED", sale_match=True) for _ in range(10)]
            + [dict(wh="W1", status="COUNTERED", sale_match=False) for _ in range(50)])
    m = pd.DataFrame(rows)
    m["day"] = pd.Timestamp("2026-01-01")
    r = match_recall(m, cutoff=pd.Timestamp("2026-02-01"))
    assert r["W1"] == 0.5
    assert np.isclose(r["W2"], 310 / 610)
    assert np.isclose(r["__all__"], 310 / 610)
