import numpy as np
import pandas as pd

from reserve.config import Params
from reserve.counter_curve import CounterCurve
from reserve.optimizer import expected_value, optimize, price_grid

ZERO = CounterCurve(np.array([0.0, 0.2]), {"__all__": np.array([0.0, 0.0])}, 0.2)
HALF = CounterCurve(np.array([0.0, 0.2]), {"__all__": np.array([0.5, 0.5])}, 0.2)


def _df(W):
    return pd.DataFrame({"m": [100.0], "W": [W], "wh": ["W1"]})


def test_price_grid():
    g = price_grid(Params())
    assert g[0] == 0.70 and g[-1] == 1.15 and len(g) == 46


def test_expected_value_counter_branch():
    V, S, R = expected_value(np.array([100.0]), np.array([80.0]), np.array([[0.9]]),
                             np.array(["W1"]), np.array([1.0]), HALF)
    assert np.isclose(V[0, 0], 0.5 * 100 + 0.5 * 80)
    assert np.isclose(S[0, 0], 0.5)
    assert np.isclose(R[0, 0], 50.0)


def test_optimize_sells_when_bid_beats_waiting():
    out = optimize(_df(W=50.0), np.full((1, 4), 0.9), ZERO, Params())
    assert np.isclose(out.reserve.iloc[0], 90.0)          # highest price that still accepts
    assert np.isclose(out.exp_value.iloc[0], 90.0)
    assert np.isclose(out.p_sell.iloc[0], 1.0)


def test_optimize_holds_when_waiting_beats_bid():
    out = optimize(_df(W=95.0), np.full((1, 4), 0.9), ZERO, Params())
    assert out.reserve.iloc[0] > 90.0
    assert np.isclose(out.exp_value.iloc[0], 95.0)


def test_trust_region_limits_reserve():
    out = optimize(_df(W=95.0), np.full((1, 4), 0.9), ZERO, Params(), b0_ratio=np.array([0.80]))
    assert out.g_star.iloc[0] <= 0.80 * 1.15 + 1e-9
