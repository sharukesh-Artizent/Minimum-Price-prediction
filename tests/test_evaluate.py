import numpy as np
import pandas as pd

from reserve_clm.baselines import carry_forward_ratio, fixed_ratio, majority_band
from reserve_clm.evaluate import (band_ece, band_metrics, calibration_table, decode_bands,
                                  eval_bands, eval_calibration, eval_intervals, eval_replay,
                                  interval_metrics, replay)


def test_decode_median_vs_argmax():
    p = np.zeros((1, 14))
    p[0, [2, 9, 10]] = [0.4, 0.3, 0.3]
    assert decode_bands(p, "argmax")[0] == 2 and decode_bands(p)[0] == 9


def test_interval_metrics():
    m = interval_metrics([100, 120, 80, 95], [90, 90, 90, 95], [110, 110, 110, 95], [100, 100, 100, 95])
    assert m["n"] == 4 and m["interval_hit_rate"] == 0.5
    assert m["too_high_rate"] == 0.25 and m["too_low_rate"] == 0.25
    assert np.isclose(m["mape_vs_mid"], 0.1) and np.isclose(m["bias_vs_mid"], 0.0)
    assert np.isnan(interval_metrics([], [], [], [])["interval_hit_rate"])


def test_band_metrics():
    m = band_metrics(np.array([3, 4, 9]), np.array([3, 5, 5]))
    assert np.isclose(m["band_exact"], 1 / 3) and np.isclose(m["band_within_1"], 2 / 3) and np.isclose(m["band_mae"], 5 / 3)


def test_replay_counts():
    d = pd.Timestamp("2026-03-02")
    offers = pd.DataFrame({"sku": ["A"] * 4, "wh": ["W"] * 4, "day": [d] * 4, "price": [90.0, 100, 110, 120],
                           "qty": [1.0, 1, 2, 1], "accepted": [False, True, False, True]})
    groups = pd.DataFrame({"sku": ["A"], "wh": ["W"], "day": [d], "reserve": [105.0]})
    r = replay(offers, groups)
    assert r["n_offers"] == 4 and r["agreement"] == 0.5
    assert r["lost_sales_rate"] == 0.25 and r["underpricing_rate"] == 0.25
    assert r["accepted_value_pred"] == 340 and r["accepted_value_actual"] == 220


def test_calibration_and_ece():
    t = calibration_table(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]), n_bins=2)
    assert t.n.tolist() == [2, 2] and t.interval_hit_rate.tolist() == [0.0, 1.0]
    p = np.zeros((2, 14))
    p[:, 0] = 1.0
    assert band_ece(p, np.array([0, 0])) == 0.0 and band_ece(p, np.array([1, 1])) == 1.0


def test_baselines():
    df = pd.DataFrame({"last_lo_ratio": [0.9, np.nan], "last_hi_ratio": [1.0, np.nan]})
    np.testing.assert_allclose(carry_forward_ratio(df, 0.97), [0.95, 0.97])
    assert fixed_ratio(2, 0.9).tolist() == [0.9, 0.9]
    assert majority_band(np.array([3, 3, 5])) == 3


def test_evaluate_edge_cases():
    # Empty inputs
    assert np.isnan(band_metrics([], [])["band_exact"])
    assert np.isnan(band_ece([], []))
    assert calibration_table([], []).empty
    assert list(calibration_table([], []).columns) == ["bin", "n", "mean_confidence", "interval_hit_rate"]

    # Replay with empty merge
    d = pd.Timestamp("2026-03-02")
    offers = pd.DataFrame({"sku": ["A"], "wh": ["W"], "day": [d], "price": [90.0], "qty": [1.0], "accepted": [False]})
    groups = pd.DataFrame({"sku": ["B"], "wh": ["W"], "day": [d], "reserve": [105.0]})
    r = replay(offers, groups)
    assert r["n_groups"] == 1 and r["n_offers"] == 0
    assert "agreement" not in r

    # Aliases
    assert eval_intervals is interval_metrics
    assert eval_bands is band_metrics
    assert eval_replay is replay
    assert eval_calibration is calibration_table
