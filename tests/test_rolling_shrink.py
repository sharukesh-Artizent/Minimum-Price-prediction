import numpy as np
import pandas as pd

from reserve.rolling import trailing_sum
from reserve.shrink import level_trailing_mean, shrink, shrink_frame

D0 = pd.Timestamp("2026-01-01")


def d(i):
    return D0 + pd.Timedelta(days=i)


def test_trailing_sum_excludes_query_day_and_window_edges():
    ev = pd.DataFrame({"sku": ["A"] * 3, "day": [d(1), d(2), d(5)], "v": [1.0, 2.0, 4.0]})
    q = pd.DataFrame({"sku": ["A"] * 4 + ["B"], "day": [d(2), d(3), d(6), d(9), d(3)]})
    out = trailing_sum(ev, ["sku"], ["v"], q, window_days=3).set_index(["sku", "day"]).v
    assert out[("A", d(2))] == 1.0     # [d-1, d2) -> day1 only, query day excluded
    assert out[("A", d(3))] == 3.0     # [d0, d3) -> days 1, 2
    assert out[("A", d(6))] == 4.0     # [d3, d6) -> day 5
    assert out[("A", d(9))] == 0.0     # [d6, d9) -> nothing
    assert out[("B", d(3))] == 0.0     # unseen key


def test_shrink_moves_toward_parent_by_count():
    v = [np.array([10.0, 10.0]), np.array([np.nan, 20.0])]
    n = [np.array([100.0, 100.0]), np.array([0.0, 5.0])]
    est = shrink(v, n, k=5.0, prior=0.0)
    root = (100 * 10 + 5 * 0) / 105
    assert np.isclose(est[0], root)
    assert np.isclose(est[1], (5 * 20 + 5 * root) / 10)


def test_level_trailing_mean_weighted_and_counted():
    ev = pd.DataFrame({"model": ["M", "M", "M"], "sku": ["a", "a", "b"],
                       "day": [d(1), d(1), d(1)], "logp": [1.0, 3.0, 5.0], "units": [1.0, 3.0, 1.0]})
    q = pd.DataFrame({"model": ["M"], "sku": ["a"], "day": [d(2)]})
    out = level_trailing_mean(ev, "logp", "units", q, [["model"], ["sku"]], 30, "px").iloc[0]
    assert np.isclose(out.px_mean_L0, (1 + 9 + 5) / 5)
    assert out.px_n_L0 == 3
    assert np.isclose(out.px_mean_L1, (1 + 9) / 4)
    assert out.px_n_L1 == 2


def test_shrink_frame():
    df = pd.DataFrame({
        "px_mean_L0": [10.0, 10.0],
        "px_n_L0": [100.0, 100.0],
        "px_mean_L1": [np.nan, 20.0],
        "px_n_L1": [0.0, 5.0],
    })
    est = shrink_frame(df, prefix="px", n_levels=2, k=5.0, prior=0.0)
    root = (100 * 10 + 5 * 0) / 105
    assert np.isclose(est[0], root)
    assert np.isclose(est[1], (5 * 20 + 5 * root) / 10)
