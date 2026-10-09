"""What CLM reads: the state text (one SKU x warehouse x day) and the 14 band candidates."""
from __future__ import annotations

import math

import pandas as pd

from .bands import BAND_KEYS, EDGES, N_BANDS

QUESTION = ("Which reserve (minimum) price band should be set today for this SKU at this warehouse? "
            "Bands are multiples of the SKU's recent market price.")


def _int(x) -> int:
    return 0 if pd.isna(x) else int(x)


def _ratio(x) -> str:
    return "unknown" if pd.isna(x) else f"{x:.2f}x market"


def _pct(x) -> str:
    return "unknown" if pd.isna(x) else f"{x:.0%}"


def _trend(x) -> str:
    if pd.isna(x):
        return "unknown"
    pct = (x - 1) * 100
    word = "rising" if pct > 2 else "falling" if pct < -2 else "flat"
    return f"{word} ({pct:+.1f}%)"


def _demand(n: int) -> str:
    return "none" if n == 0 else "low" if n < 3 else "moderate" if n < 10 else "high"


def _velocity(u: int) -> str:
    return "no sales" if u == 0 else "slow" if u < 5 else "steady" if u < 30 else "fast"


def state_context(r) -> str:
    source = "this SKU" if r.ref_source == "sku" else "similar products (same model, capacity and grade)"
    last_sale = ("No sale recorded before today" if pd.isna(r.days_since_last_sale)
                 else f"Last sale {_int(r.days_since_last_sale)} days ago")
    if pd.isna(r.last_lo_ratio) or pd.isna(r.last_hi_ratio):
        last_floor = "No previous reserve observed for this SKU at this warehouse."
    else:
        last_floor = (f"Previous reserve between {r.last_lo_ratio:.2f}x and {r.last_hi_ratio:.2f}x market, "
                      f"{_int(r.days_since_last_floor)} days ago.")
    n7 = _int(r.n_offers_7)
    sold30 = _int(r.units_sold_30)
    return "\n".join([
        f"Product: {r.manufacturer} {r.model}, {r.capacity}, grade {r.grade}, colour {r.color}, "
        f"carrier {r.carrier}, {r.lock_status}. Warehouse {r.wh}.",
        f"Market price: ${r.ref_price:.0f} from {source}, last 30 days, {_int(r.ref_n)} accepted offers. "
        f"Trend last 7 days vs 30 days: {_trend(r.ref_trend)}.",
        f"Bids last 7 days: {n7} ({_demand(n7)} demand). Bids last 30 days: {_int(r.n_offers_30)} "
        f"from {_int(r.n_bidders_30)} buyers for {_int(r.units_req_30)} units.",
        f"Median bid {_ratio(r.bid_med_ratio_30)}, top bid {_ratio(r.bid_top_ratio_30)}. "
        f"Share of recent bids accepted: {_pct(r.accept_share_30)}.",
        f"Units sold last 7 days: {_int(r.units_sold_7)}, last 30 days: {sold30} ({_velocity(sold30)}). "
        f"{last_sale}. Median sale price {_ratio(r.sale_price_ratio_30)}.",
        last_floor,
        f"Day: {pd.Timestamp(r.day):%A}.",
    ])


def state_text(r) -> str:
    """Context first, question last: the layout CLM's heads were trained on (clm.schema.state_text)."""
    return f"{state_context(r).strip()}\n\n{QUESTION}"


_TIERS = [  # (upper bound on band midpoint, name, when to use)
    (0.875, "Heavy discount reserve", "Use when stock is slow or ageing, bids are well below market and prices are falling."),
    (0.95, "Moderate discount reserve", "Use when demand is soft, bids sit below market and sell-through is slow."),
    (1.00, "Slight discount reserve", "Use when demand is steady, several buyers bid close to market and prices are flat."),
    (1.05, "At-market reserve", "Use when demand is healthy, bids reach market price and stock sells steadily."),
    (math.inf, "Premium reserve", "Use when stock is scarce, competing bids exceed market and prices are rising."),
]


def band_descriptions() -> dict[str, str]:
    out = {}
    for i, key in enumerate(BAND_KEYS):
        lo, hi = EDGES[i], EDGES[i + 1]
        if i == 0:
            out[key] = ("Deep clearance reserve: below 0.80x the market price. Use for stale stock with very "
                        "weak, low bids and falling prices.")
        elif i == N_BANDS - 1:
            out[key] = ("Strong premium reserve: at or above 1.10x the market price. Use for scarce stock with "
                        "strong competing bids well above market and rising prices.")
        else:
            name, hint = next((n, h) for bound, n, h in _TIERS if (lo + hi) / 2 < bound)
            out[key] = f"{name}: between {lo:.3f}x and {hi:.3f}x the market price. {hint}"
    return out
