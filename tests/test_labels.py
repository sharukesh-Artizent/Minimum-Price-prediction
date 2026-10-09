import pandas as pd

from reserve_clm.labels import assign_split, floor_intervals


def _offers(rows):
    df = pd.DataFrame(rows, columns=["sku", "wh", "day", "price", "status"])
    df["day"] = pd.to_datetime(df.day)
    df["accepted"] = df.status.eq("ACCEPTED")
    return df


def test_floor_intervals():
    o = _offers([
        ("A", "W1", "2026-01-05", 90, "NOT_ACCEPTED"), ("A", "W1", "2026-01-05", 95, "COUNTERED"),
        ("A", "W1", "2026-01-05", 100, "ACCEPTED"), ("A", "W1", "2026-01-05", 105, "ACCEPTED"),
        ("A", "W1", "2026-01-06", 100, "ACCEPTED"),                                   # one-sided
        ("A", "W2", "2026-01-05", 110, "NOT_ACCEPTED"), ("A", "W2", "2026-01-05", 100, "ACCEPTED"),  # inconsistent
    ])
    f = floor_intervals(o)
    assert len(f) == 1
    r = f.iloc[0]
    assert (r.sku, r.wh, r.lo, r.hi, r.mid, r.n_offers_day) == ("A", "W1", 95, 100, 97.5, 4)


def test_floor_intervals_empty_and_equal():
    # Empty offers
    o_empty = _offers([])
    f_empty = floor_intervals(o_empty)
    assert len(f_empty) == 0
    assert list(f_empty.columns) == ["sku", "wh", "day", "lo", "hi", "n_offers_day", "mid"]

    # lo == hi case
    o_equal = _offers([
        ("B", "W1", "2026-01-10", 100, "NOT_ACCEPTED"),
        ("B", "W1", "2026-01-10", 100, "ACCEPTED"),
    ])
    f_equal = floor_intervals(o_equal)
    assert len(f_equal) == 1
    r = f_equal.iloc[0]
    assert (r.lo, r.hi, r.mid, r.n_offers_day) == (100, 100, 100.0, 2)


def test_floor_intervals_disjoint_days():
    o_disjoint = _offers([
        ("C", "W1", "2026-01-10", 100, "NOT_ACCEPTED"),
        ("C", "W1", "2026-01-11", 100, "ACCEPTED"),
    ])
    f_disjoint = floor_intervals(o_disjoint)
    assert len(f_disjoint) == 0
    assert list(f_disjoint.columns) == ["sku", "wh", "day", "lo", "hi", "n_offers_day", "mid"]


def test_assign_split():
    days = pd.Series(pd.to_datetime(["2026-01-31", "2026-02-01", "2026-02-28", "2026-03-01"]))
    assert assign_split(days).tolist() == ["train", "val", "val", "test"]
