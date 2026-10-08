import numpy as np
import pandas as pd

from reserve.market import daily_depreciation, proxy_inventory


def _nodes():
    return pd.DataFrame({"sku": ["1", "2", "3"], "warehouse": ["W1", "W1", "W1"],
                         "family": ["F", "F", "G"]})


def _sales():
    days = np.arange(40)
    f = pd.DataFrame({"sku": "1", "warehouse": "W1",
                      "date": pd.Timestamp("2026-01-01") + pd.to_timedelta(days, unit="D"),
                      "unit_price": 100.0 * np.exp(-0.005 * days), "qty": 1.0})
    g = pd.DataFrame({"sku": "3", "warehouse": "W1",
                      "date": pd.to_datetime(["2026-01-01", "2026-01-05"]),
                      "unit_price": [50.0, 40.0], "qty": [2.0, 3.0]})
    return pd.concat([f, g], ignore_index=True)


def test_daily_depreciation_recovers_slope_and_falls_back_for_thin_family():
    delta = daily_depreciation(_sales(), _nodes(), default=0.002)
    assert np.allclose(delta[:2], 0.005, atol=1e-9)
    assert delta[2] == 0.002


def test_proxy_inventory_sums_units_in_window():
    inv = proxy_inventory(_sales(), _nodes(), "2026-01-01", "2026-01-03")
    assert inv.tolist() == [2.0, 0.0, 2.0]


def test_daily_depreciation_empty_and_unmatched_sales():
    empty_sales = pd.DataFrame(columns=["sku", "warehouse", "date", "unit_price", "qty"])
    delta_empty = daily_depreciation(empty_sales, _nodes(), default=0.003)
    assert np.allclose(delta_empty, [0.003, 0.003, 0.003])

    unmatched_sales = pd.DataFrame({
        "sku": ["999"], "warehouse": ["W999"],
        "date": [pd.Timestamp("2026-01-01")],
        "unit_price": [100.0], "qty": [1.0]
    })
    delta_unmatched = daily_depreciation(unmatched_sales, _nodes(), default=0.004)
    assert np.allclose(delta_unmatched, [0.004, 0.004, 0.004])


def test_daily_depreciation_clipping_bounds():
    days = np.arange(40)
    # Price increasing rapidly: slope > 0 -> -slope < 0 -> clipped to 0.0
    f_up = pd.DataFrame({"sku": "1", "warehouse": "W1",
                         "date": pd.Timestamp("2026-01-01") + pd.to_timedelta(days, unit="D"),
                         "unit_price": 100.0 * np.exp(0.01 * days), "qty": 1.0})
    delta_up = daily_depreciation(f_up, _nodes(), default=0.002, cap=0.01)
    assert delta_up[0] == 0.0

    # Price dropping faster than cap: slope = -0.05 -> -slope = 0.05 -> clipped to cap (0.01)
    f_down = pd.DataFrame({"sku": "1", "warehouse": "W1",
                           "date": pd.Timestamp("2026-01-01") + pd.to_timedelta(days, unit="D"),
                           "unit_price": 100.0 * np.exp(-0.05 * days), "qty": 1.0})
    delta_down = daily_depreciation(f_down, _nodes(), default=0.002, cap=0.01)
    assert delta_down[0] == 0.01


def test_proxy_inventory_empty_window_and_unmatched():
    inv_empty = proxy_inventory(_sales(), _nodes(), "2025-01-01", "2025-01-02")
    assert inv_empty.tolist() == [0.0, 0.0, 0.0]
