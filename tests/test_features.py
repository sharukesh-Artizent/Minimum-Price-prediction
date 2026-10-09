import numpy as np
import pandas as pd

from reserve_clm.features import build_features, lagged_join, reference_prices

D0 = pd.Timestamp("2026-01-10")
FAM = "Apple|iPhone 13|128GB|B"


def day(k):
    return D0 - pd.Timedelta(days=k)


def products():
    rows = [("A", "W1"), ("B", "W1")]
    p = pd.DataFrame(rows, columns=["sku", "wh"])
    for c, v in [("manufacturer", "Apple"), ("model", "iPhone 13"), ("capacity", "128GB"), ("grade", "B"),
                 ("color", "Blue"), ("carrier", "OTH"), ("lock_status", "UNLOCKED"), ("category", "Phones")]:
        p[c] = v
    p["family"] = FAM
    return p


def offers():
    rows = [  # sku, wh, day, price, accepted, buyer, qty
        ("B", "W1", day(1), 100, True, "b1", 2), ("B", "W1", day(2), 110, True, "b2", 1),
        ("B", "W1", day(3), 120, True, "b1", 1), ("B", "W1", day(3), 80, False, "b3", 5),
        ("A", "W1", day(1), 200, True, "b1", 1), ("A", "W1", D0, 999, True, "b9", 1),   # same day: excluded
        ("A", "W1", day(40), 50, True, "b1", 1),                                          # outside window
    ]
    o = pd.DataFrame(rows, columns=["sku", "wh", "day", "price", "accepted", "buyer_id", "qty"])
    o["day"] = o.day.astype("datetime64[ns]")
    return o


def sales():
    s = pd.DataFrame([("B", "W1", day(2), 3.0, 105.0, "STOCKLIST"), ("B", "W1", D0, 9.0, 1.0, "STOCKLIST"),
                      ("B", "W1", day(10), 1.0, 99.0, "AUCTION")],
                     columns=["sku", "wh", "day", "units", "unit_price", "order_type"])
    s["day"] = s.day.astype("datetime64[ns]")
    return s


def floors():
    f = pd.DataFrame([("B", "W1", day(5), 90.0, 95.0), ("B", "W1", D0, 50.0, 60.0)],
                     columns=["sku", "wh", "day", "lo", "hi"])
    f["day"] = f.day.astype("datetime64[ns]")
    return f


TARGETS = pd.DataFrame({"sku": ["A", "B"], "wh": ["W1", "W1"], "day": [D0, D0]}).astype({"day": "datetime64[ns]"})


def test_lagged_join_window_is_strictly_before():
    ev = pd.DataFrame({"sku": ["A"] * 4, "day": [day(0), day(1), day(30), day(31)], "v": [1, 2, 3, 4]})
    j = lagged_join(ev, TARGETS, ["sku"], 30)
    assert sorted(j.v.tolist()) == [2, 3] and sorted(j.lag.tolist()) == [1, 30]


def test_reference_sku_then_family():
    r = reference_prices(offers(), products(), TARGETS).set_index("sku")
    assert r.loc["B", "ref_source"] == "sku" and r.loc["B", "ref_price"] == 110 and r.loc["B", "ref_n"] == 3
    # A has 1 accepted offer in window -> family: daily medians day1 median(100,200)=150, day2 110, day3 120
    assert r.loc["A", "ref_source"] == "family" and r.loc["A", "ref_price"] == 120 and r.loc["A", "ref_n"] == 4


def test_reference_none_when_under_min():
    # Only 2 accepted offers for family across the window -> both should be "none"
    rows = [
        ("B", "W1", day(1), 100, True, "b1", 1),
        ("A", "W1", day(2), 200, True, "b2", 1),
    ]
    o = pd.DataFrame(rows, columns=["sku", "wh", "day", "price", "accepted", "buyer_id", "qty"])
    o["day"] = o.day.astype("datetime64[ns]")
    r = reference_prices(o, products(), TARGETS).set_index("sku")
    assert r.loc["A", "ref_source"] == "none" and np.isnan(r.loc["A", "ref_price"]) and r.loc["A", "ref_n"] == 0
    assert r.loc["B", "ref_source"] == "none" and np.isnan(r.loc["B", "ref_price"]) and r.loc["B", "ref_n"] == 0
    assert np.isnan(r.loc["A", "ref_trend"])


def test_reference_trend_calc():
    # SKU B has 3 accepted offers: day(2) at 120, day(20) at 100, day(25) at 100
    # 30-day median = median(120, 100, 100) = 100.
    # 7-day median = day(2) = 120.
    # ref_trend = 120 / 100 = 1.2.
    rows = [
        ("B", "W1", day(2), 120, True, "b1", 1),
        ("B", "W1", day(20), 100, True, "b2", 1),
        ("B", "W1", day(25), 100, True, "b3", 1),
    ]
    o = pd.DataFrame(rows, columns=["sku", "wh", "day", "price", "accepted", "buyer_id", "qty"])
    o["day"] = o.day.astype("datetime64[ns]")
    target_b = TARGETS[TARGETS.sku == "B"]
    r = reference_prices(o, products(), target_b).iloc[0]
    assert r.ref_source == "sku"
    assert r.ref_price == 100.0
    assert np.isclose(r.ref_trend, 1.2)


def test_build_features_values():
    f = build_features(offers(), sales(), products(), floors(), TARGETS).set_index("sku")
    b = f.loc["B"]
    assert b.n_offers_30 == 4 and b.n_offers_7 == 4 and b.n_bidders_30 == 3 and b.units_req_30 == 9
    assert b.acc_30 == 3 and np.isclose(b.accept_share_30, 0.75)
    assert b.bid_top_30 == 120 and np.isclose(b.bid_top_ratio_30, 120 / 110)
    assert b.units_sold_30 == 4 and b.units_sold_7 == 3 and b.days_since_last_sale == 2
    assert b.sale_price_med_30 == 105                       # auction sale ignored for price
    assert b.last_lo == 90 and b.last_hi == 95 and b.days_since_last_floor == 5
    assert b.grade == "B" and b.family == FAM
    a = f.loc["A"]
    assert a.n_offers_30 == 1 and np.isnan(a.days_since_last_sale) and np.isnan(a.last_lo_ratio)


def test_build_features_columns_and_empty():
    empty_offers = pd.DataFrame(columns=["sku", "wh", "day", "price", "accepted", "buyer_id", "qty"])
    empty_sales = pd.DataFrame(columns=["sku", "wh", "day", "units", "unit_price", "order_type"])
    empty_floors = pd.DataFrame(columns=["sku", "wh", "day", "lo", "hi"])
    f = build_features(empty_offers, empty_sales, products(), empty_floors, TARGETS)
    expected_cols = [
        "sku", "wh", "day", "family", "ref_source", "ref_price", "ref_n", "ref_trend",
        "n_offers_30", "units_req_30", "acc_30", "bid_med_30", "bid_top_30", "n_offers_7",
        "n_bidders_30", "units_sold_30", "units_sold_7", "sale_price_med_30",
        "days_since_last_sale", "last_lo", "last_hi", "days_since_last_floor",
        "manufacturer", "model", "capacity", "grade", "color", "carrier", "lock_status",
        "category", "bid_med_ratio_30", "bid_top_ratio_30", "sale_price_ratio_30",
        "last_lo_ratio", "last_hi_ratio", "accept_share_30"
    ]
    assert list(f.columns) == expected_cols
    assert len(f) == 2
    row = f.iloc[0]
    assert row.n_offers_30 == 0
    assert row.units_sold_30 == 0
    assert row.ref_source == "none"
    assert np.isnan(row.ref_price)
    assert np.isnan(row.days_since_last_sale)
    assert np.isnan(row.days_since_last_floor)
