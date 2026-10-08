import numpy as np
import pandas as pd

from reserve.economics import continuation_value, estimate_delta, estimate_rho


def test_continuation_value_formula_and_salvage_floor():
    w = continuation_value(np.array([100.0, 100.0]), np.array([10.0, 90.0]),
                           delta=0.01, rho=1.0, h_pct=np.array([0.001, 0.01]), salvage_pct=0.5)
    assert np.isclose(w[0], 100 * 0.99 ** 10 - 0.001 * 100 * 10)
    assert w[1] == 50.0


def test_estimate_delta_recovers_slope():
    days = pd.date_range("2025-10-10", periods=84, freq="D")
    s = pd.DataFrame({"model": "M", "capacity": "128GB", "grade": "A", "day": days,
                      "unit_price": 100 * np.exp(-0.002 * np.arange(84)), "units": 1.0})
    delta = estimate_delta(s, cutoff=pd.Timestamp("2026-06-01"))
    assert abs(delta - 0.002) < 2e-4


def test_estimate_delta_zero_when_no_long_series():
    s = pd.DataFrame({"model": "M", "capacity": "1", "grade": "A",
                      "day": pd.date_range("2025-10-10", periods=10), "unit_price": 100.0, "units": 1.0})
    assert estimate_delta(s, cutoff=pd.Timestamp("2026-01-01")) == 0.0


def test_estimate_rho_median_ratio():
    rows = pd.DataFrame({"y": np.log([102.0, 102.0, 102.0]), "log_m": np.log([100.0, 100.0, 100.0])})
    assert np.isclose(estimate_rho(rows), 1.02)
