import numpy as np
import pandas as pd

from factories import sales_df
from reserve.frame import CAT_COLS
from reserve.market import FEATURES_NUM, MarketModel, mape, realized_targets


def test_realized_targets_weighted_mean():
    s = sales_df([dict(sku="S", day="2026-01-01", unit_price=100, units=1),
                  dict(sku="S", day="2026-01-01", unit_price=200, units=3)])
    t = realized_targets(s)
    assert np.isclose(t.y.iloc[0], np.log(175))


def _synthetic(n, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FEATURES_NUM})
    df["log_m_naive"] = rng.uniform(4, 6, n)
    for c in CAT_COLS:
        df[c] = "X"
    df["grade"] = rng.choice(["A", "B"], n)
    df["y"] = df.log_m_naive + np.where(df.grade == "A", 0.2, -0.1) + rng.normal(0, 0.01, n)
    return df


def test_market_model_learns_residual_and_beats_naive():
    tr, va = _synthetic(3000, 0), _synthetic(1000, 1)
    m = MarketModel().fit(tr, va)
    assert m.use_lgb
    assert mape(va.y, m.predict(va)) < 0.03
    assert mape(va.y, va.log_m_naive) > 0.09
