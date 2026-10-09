from types import SimpleNamespace

import numpy as np
import pandas as pd

from reserve_clm.bands import BAND_KEYS
from reserve_clm.text import QUESTION, band_descriptions, state_context, state_text


def row(**kw):
    base = dict(manufacturer="Apple", model="iPhone 13", capacity="128GB", grade="B", color="Blue",
                carrier="OTH", lock_status="UNLOCKED", wh="W23", ref_price=412.4, ref_source="sku", ref_n=17.0,
                ref_trend=0.95, n_offers_7=11.0, n_offers_30=40.0, n_bidders_30=5.0, units_req_30=60.0,
                bid_med_ratio_30=0.94, bid_top_ratio_30=0.99, accept_share_30=0.55, units_sold_7=23.0,
                units_sold_30=80.0, days_since_last_sale=1.0, sale_price_ratio_30=0.97, last_lo_ratio=0.95,
                last_hi_ratio=0.97, days_since_last_floor=3.0, day=pd.Timestamp("2026-03-03"))
    base.update(kw)
    return SimpleNamespace(**base)


def test_state_text_layout_and_content():
    t = state_text(row())
    assert t.endswith("\n\n" + QUESTION)
    assert "Apple iPhone 13, 128GB, grade B" in t and "$412" in t and "falling (-5.0%)" in t
    assert "11 (high demand)" in t and "0.94x market" in t and "55%" in t
    assert "Previous reserve between 0.95x and 0.97x market, 3 days ago." in t and "Tuesday" in t
    assert "nan" not in t.lower()


def test_state_text_missing_values():
    t = state_text(row(ref_source="family", ref_trend=np.nan, bid_med_ratio_30=np.nan, accept_share_30=np.nan,
                       days_since_last_sale=np.nan, last_lo_ratio=np.nan, n_offers_7=0.0))
    assert "similar products" in t and "No previous reserve observed" in t and "No sale recorded" in t
    assert "0 (none demand)" in t and "nan" not in t.lower()


def test_band_descriptions():
    d = band_descriptions()
    assert list(d) == BAND_KEYS and len(set(d.values())) == 14
    assert "below 0.80x" in d["b00"] and "1.10x" in d["b13"]
    assert "between 0.950x and 0.975x" in d["b07"]


def test_state_context_variations():
    # Test rising and flat trends
    r_rising = row(ref_trend=1.05)
    assert "rising (+5.0%)" in state_context(r_rising)
    r_flat = row(ref_trend=1.00)
    assert "flat (+0.0%)" in state_context(r_flat)
