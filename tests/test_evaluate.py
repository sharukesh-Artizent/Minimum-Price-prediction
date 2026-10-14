import numpy as np
import pandas as pd

from reserve_clm.baselines import carry_forward_ratio, fixed_ratio, majority_band
from reserve_clm.evaluate import (band_ece, band_metrics, calibration_table, decode_bands,
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


def test_band_metrics_wide_windows():
    m = band_metrics(np.array([3, 4, 9, 0]), np.array([3, 5, 5, 13]))     # distances 0, 1, 4, 13
    assert m["band_within_1"] == 0.5 and m["band_within_3"] == 0.5 and m["band_within_5"] == 0.75


def test_replay_counts():
    d = pd.Timestamp("2026-03-02")
    offers = pd.DataFrame({"sku": ["A"] * 4, "wh": ["W"] * 4, "day": [d] * 4, "price": [90.0, 100, 110, 120],
                           "qty": [1.0, 1, 2, 1], "accepted": [False, True, False, True]})
    groups = pd.DataFrame({"sku": ["A"], "wh": ["W"], "day": [d], "reserve": [105.0]})
    r = replay(offers, groups)
    assert r["n_offers"] == 4 and r["agreement"] == 0.5
    assert r["lost_sales_rate"] == 0.25 and r["underpricing_rate"] == 0.25
    assert "accepted_value_pred" not in r                  # offer qty overstates real sales ~3x: not reported


def test_replay_daily_acceptance():
    d = pd.Timestamp("2026-03-02")
    rows = [("A", 90, False), ("A", 100, True), ("A", 110, False), ("A", 120, True),   # reserve 105
            ("B", 80, False), ("B", 130, True),                                        # reserve 125: exact day
            ("C", 50, False), ("C", 60, False)]                                        # reserve 55
    offers = pd.DataFrame(rows, columns=["sku", "price", "accepted"]).assign(wh="W", day=d, qty=1.0)
    groups = pd.DataFrame({"sku": ["A", "B", "C", "D"], "wh": "W", "day": d, "reserve": [105.0, 125, 55, 1]})
    r = replay(offers, groups)
    assert r["n_groups"] == 4 and r["n_days"] == 3 and r["n_offers"] == 8      # D has no offers
    assert r["agreement"] == 5 / 8 and r["pred_accept_rate"] == 0.5 and r["actual_accept_rate"] == 3 / 8
    assert r["accept_precision"] == 0.5 and np.isclose(r["accept_recall"], 2 / 3)
    assert np.isclose(r["day_exact_match"], 1 / 3) and np.isclose(r["day_count_mae"], 1 / 3)
    assert np.isclose(r["accepted_per_day_pred"], 4 / 3) and r["accepted_per_day_actual"] == 1.0


def test_calibration_and_ece():
    t = calibration_table(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]), n_bins=2)
    assert t.n.tolist() == [2, 2] and t.interval_hit_rate.tolist() == [0.0, 1.0]
    p = np.zeros((2, 14))
    p[:, 0] = 1.0
    assert band_ece(p, np.array([0, 0])) == 0.0 and band_ece(p, np.array([1, 1])) == 1.0
    p_zero = np.zeros((1, 14))
    assert band_ece(p_zero, np.array([0])) == 1.0 and band_ece(p_zero, np.array([1])) == 0.0


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

