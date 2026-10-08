import numpy as np
import pandas as pd

from reserve.config import Params
from reserve.counter_curve import CounterCurve
from reserve.metrics import bootstrap_uplift, deviation_bucket, pareto_segment, segment_table
from reserve.replay import hindsight_reserve, replay_actual, replay_policy, summarize

HALF = CounterCurve(np.array([0.0, 0.2]), {"__all__": np.array([0.5, 0.5])}, 0.2)
ZERO = CounterCurve(np.array([0.0, 0.2]), {"__all__": np.array([0.0, 0.0])}, 0.2)
DAY = pd.Timestamp("2026-03-02")


def _ev():
    return pd.DataFrame({"sku": "S", "wh": "W1", "day": DAY, "offer_price": [100.0, 90.0, 50.0],
                         "qty": [1.0, 2.0, 1.0], "m": 100.0, "W": 80.0,
                         "status": ["ACCEPTED", "COUNTERED", "NOT_ACCEPTED"],
                         "counter_price": [np.nan, 99.0, np.nan]})


def test_replay_policy_accept_counter_reject():
    r = replay_policy(_ev(), np.full(3, 95.0), HALF)
    assert list(r.decision) == ["ACCEPTED", "COUNTERED", "NOT_ACCEPTED"]
    assert np.allclose(r.sold, [1.0, 1.0, 0.0])
    assert np.allclose(r.revenue, [100.0, 95.0, 0.0])
    assert np.allclose(r.value, [100.0, 95.0 + 80.0, 80.0])


def test_replay_actual_uses_real_decisions():
    r = replay_actual(_ev(), HALF)
    assert np.allclose(r.revenue, [100.0, 2 * 0.5 * 99.0, 0.0])
    assert np.allclose(r.sold, [1.0, 1.0, 0.0])


def test_summarize():
    s = summarize(replay_policy(_ev(), np.full(3, 95.0), HALF))
    assert np.isclose(s["revenue"], 195.0)
    assert np.isclose(s["sell_through"], 2.0 / 4.0)
    assert np.isclose(s["realization"], 195.0 / 200.0)
    assert np.isclose(s["total_value"], 355.0)
    assert np.isclose(s["accept_rate"], 1 / 3)


def test_hindsight_picks_value_maximizing_highest_price():
    ev = pd.DataFrame({"sku": "S", "wh": "W1", "day": DAY, "offer_price": [100.0, 80.0],
                       "qty": 1.0, "m": 100.0, "W": 0.0})
    h = hindsight_reserve(ev, ZERO, Params())
    assert np.isclose(h.reserve_oracle.iloc[0], 80.0)


def test_bootstrap_constant_uplift():
    days = pd.date_range("2026-03-01", periods=5)
    b = pd.DataFrame({"day": days, "value": 100.0})
    a = b.assign(value=110.0)
    u = bootstrap_uplift(a, b, n_boot=200)
    assert np.isclose(u["uplift"], 50.0) and np.isclose(u["ci_lo"], 50.0) and np.isclose(u["ci_hi"], 50.0)
    assert np.isclose(u["uplift_pct"], 0.1)


def test_pareto_segment_top_share():
    o = pd.DataFrame({"sku": ["A"] * 8 + ["B", "C"], "wh": "W1"})
    seg = pareto_segment(o).set_index("sku").segment
    assert seg["A"] == "A" and seg["B"] == "BC" and seg["C"] == "BC"


def test_segment_table_and_deviation_bucket():
    r = replay_policy(_ev(), np.full(3, 95.0), HALF).assign(segment="A")
    t = segment_table({"P2": r}, "segment")
    assert set(t.columns) >= {"segment", "policy", "revenue", "sell_through", "total_value"}
    b = deviation_bucket(np.array([100.0, 107.0, 112.0, 130.0]), np.array([100.0] * 4))
    assert list(b.astype(str)) == ["<=5%", "5-10%", "10-15%", ">15%"]
