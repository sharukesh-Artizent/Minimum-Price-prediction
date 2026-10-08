import numpy as np
import pandas as pd

from reserve.bids import BidDist
from reserve.config import Params

LV = [["category"], ["model"]]


def _offers():
    rows = [dict(category="Phones", model="X", offer_price=90.0) for _ in range(40)]
    rows += [dict(category="Phones", model="Y", offer_price=50.0) for _ in range(5)]
    rows += [dict(category="Phones", model="Z", offer_price=70.0) for _ in range(40)]
    df = pd.DataFrame(rows)
    df["m"] = 100.0
    df["day"] = pd.Timestamp("2026-01-15")
    return df


def test_lookup_uses_finest_level_with_enough_obs():
    p = Params(n_bid_quantiles=4, min_group_obs=30)
    bd = BidDist.fit(_offers(), pd.Timestamp("2026-02-01"), p, levels=LV)
    q = bd.lookup(pd.DataFrame({"category": ["Phones", "Phones", "Tablet"], "model": ["X", "Y", "Q"]}))
    assert q.shape == (3, 4)
    assert np.allclose(q[0], 0.9)                         # model X has 40 obs
    assert 0.5 < q[1].mean() < 0.9                        # Y has 5 obs -> category level mix
    assert np.allclose(q[2], q[1])                        # unknown category -> global == Phones here


def test_fit_ignores_offers_outside_window():
    o = _offers()
    o.loc[o.model == "X", "day"] = pd.Timestamp("2025-01-01")    # older than 90 days
    bd = BidDist.fit(o, pd.Timestamp("2026-02-01"), Params(n_bid_quantiles=4), levels=LV)
    q = bd.lookup(pd.DataFrame({"category": ["Phones"], "model": ["X"]}))
    assert not np.allclose(q[0], 0.9)


def test_fit_default_levels():
    df = _offers()
    df["grade"] = "A"
    df["capacity"] = "64GB"
    df["sku"] = df["model"] + "_sku"
    df["wh"] = "WH1"
    bd = BidDist.fit(df, pd.Timestamp("2026-02-01"), Params(n_bid_quantiles=4, min_group_obs=30))
    query = pd.DataFrame([{
        "category": "Phones", "model": "X", "grade": "A", "capacity": "64GB", "sku": "X_sku", "wh": "WH1"
    }])
    q = bd.lookup(query)
    assert q.shape == (1, 4)
    assert np.allclose(q[0], 0.9)
