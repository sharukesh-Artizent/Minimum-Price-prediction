import numpy as np
import pandas as pd
import pytest

from reserve_clm.bands import BAND_MIDS, N_BANDS
from reserve_clm.baselines import carry_forward_ratio, fit_baselines, fixed_ratio, majority_band, predict_baseline


def test_carry_forward_ratio_with_and_without_nan():
    df = pd.DataFrame({
        "last_lo_ratio": [0.85, np.nan, 0.90],
        "last_hi_ratio": [0.95, np.nan, 1.00]
    })
    fallback = 0.92
    res = carry_forward_ratio(df, fallback)
    expected = [0.90, 0.92, 0.95]
    np.testing.assert_allclose(res, expected)


def test_fixed_ratio():
    res = fixed_ratio(5, 0.95)
    assert len(res) == 5
    assert np.allclose(res, 0.95)
    assert res.dtype == float


def test_majority_band():
    true_bands = np.array([1, 2, 2, 3, 2, 4, 13])
    assert majority_band(true_bands) == 2


def test_fit_and_predict_baselines():
    train_df = pd.DataFrame({
        "mid_ratio": [0.90, 0.94, 0.96, 0.98, 1.00],
        "true_band": [4, 5, 5, 6, 5]
    })
    fitted = fit_baselines(train_df)
    assert np.isclose(fitted["fallback_ratio"], 0.96)
    assert fitted["majority_band"] == 5

    test_df = pd.DataFrame({
        "last_lo_ratio": [0.80, np.nan],
        "last_hi_ratio": [0.90, np.nan]
    })

    b0 = predict_baseline(test_df, "B0", fitted)
    np.testing.assert_allclose(b0, [0.85, 0.96])

    b1 = predict_baseline(test_df, "B1", fitted)
    np.testing.assert_allclose(b1, [0.96, 0.96])

    b2 = predict_baseline(test_df, "B2", fitted)
    np.testing.assert_allclose(b2, [BAND_MIDS[5], BAND_MIDS[5]])

    with pytest.raises(ValueError):
        predict_baseline(test_df, "unknown", fitted)
