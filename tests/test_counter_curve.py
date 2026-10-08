import numpy as np
import pandas as pd

from reserve.counter_curve import CounterCurve, fit_counter_curve


def _matched(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    gap = rng.uniform(0.005, 0.10, n)
    true_p = 0.8 - 5 * gap
    taken = rng.random(n) < 0.5 * true_p            # 50% match recall
    b = np.full(n, 100.0)
    return pd.DataFrame({"status": "COUNTERED", "offer_price": b, "counter_price": b * (1 + gap),
                         "counter_taken": taken, "wh": "W1", "day": pd.Timestamp("2026-01-01")})


def test_fit_counter_curve_recall_corrected_and_monotone():
    recall = pd.Series({"W1": 0.5, "__all__": 0.5})
    c = fit_counter_curve(_matched(), pd.Timestamp("2026-02-01"), recall)
    assert np.all(np.diff(c.curves["W1"]) <= 1e-12)
    p = c.pi(np.array([0.01, 0.08]), np.array(["W1", "W1"]))
    assert abs(p[0] - 0.75) < 0.1
    assert abs(p[1] - 0.40) < 0.1


def test_pi_zero_outside_range_and_global_fallback():
    c = CounterCurve(grid=np.array([0.0, 0.1]), curves={"__all__": np.array([0.6, 0.2])}, max_gap=0.1)
    p = c.pi(np.array([[0.05, 0.2, -0.01]]), np.array(["UNSEEN"]))
    assert np.allclose(p, [[0.4, 0.0, 0.0]])
