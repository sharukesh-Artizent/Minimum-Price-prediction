import numpy as np
import pandas as pd

from reserve.counter_curve import CounterCurve
from reserve.rule_b1 import apply_rule, bands, tau_edges_from, tune_rule

ZERO = CounterCurve(np.array([0.0, 0.2]), {"__all__": np.array([0.0, 0.0])}, 0.2)


def test_bands():
    df = pd.DataFrame({"age_days": [0, 45, 75, 200], "tau": [1.0, 5.0, 20.0, 60.0]})
    a, t = bands(df, np.array([4.0, 30.0]))
    assert list(a) == [0, 1, 2, 3]
    assert list(t) == [0, 1, 1, 2]


def test_tau_edges_terciles():
    e = tau_edges_from(pd.DataFrame({"tau": np.arange(1.0, 10.0)}))
    assert len(e) == 2 and e[0] < e[1]


def test_tune_rule_picks_smallest_discount_that_clears():
    ev = pd.DataFrame({"sku": "S", "wh": "W1", "day": pd.Timestamp("2026-02-10"),
                       "offer_price": 90.0, "qty": 1.0, "m": 100.0, "W": 0.0,
                       "age_days": 10.0, "tau": 2.0}, index=[0])
    edges = np.array([5.0, 10.0])
    table = tune_rule(ev, ZERO, edges)
    assert table[(0, 0)] == 0.10
    r = apply_rule(pd.DataFrame({"m": [100.0, 100.0], "age_days": [10.0, 100.0], "tau": [2.0, 50.0]}),
                   table, edges, default=0.05)
    assert np.allclose(r, [90.0, 95.0])
