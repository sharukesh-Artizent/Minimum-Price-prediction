import numpy as np
import pandas as pd

from reserve_clm.baselines import carry_forward_ratio, fixed_ratio, majority_band


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
