"""Replay hold-out offers against a reserve policy with a stock cap per node."""
import numpy as np

from reserve.observations import Observations


def build_fair_actual(obs: Observations) -> tuple[np.ndarray, np.ndarray]:
    """Return (accept_mask, prices) crediting accepted bids and redeemed counters at counter_price."""
    mask = obs.accepted == 1.0
    prices = np.exp(obs.log_bid).copy()
    if obs.order_status is not None and obs.counter_price is not None:
        redeemed_counter = (obs.accepted == 0.0) & (obs.order_status == "REDEEMED") & (obs.counter_price > 0)
        mask = mask | redeemed_counter
        prices = np.where(redeemed_counter, obs.counter_price, prices)
    return mask, prices


def replay(
    reserve_log: np.ndarray,
    obs: Observations,
    inventory: np.ndarray,
    value: np.ndarray,
    delta: np.ndarray,
    horizon_days: int,
    accept_mask: np.ndarray | None = None,
    prices: np.ndarray | None = None,
) -> dict:
    """Walk offers by date, node, highest bid first; accept if bid >= reserve and stock left.

    accept_mask overrides the reserve rule (used to replay Assurant's actual decisions).
    prices optionally overrides the transacted unit price (used for counter prices).
    Unsold stock is marked at value * (1 - delta) ** horizon_days.
    """
    order = np.lexsort((-obs.log_bid, obs.node, obs.date))
    remaining = inventory.astype(float).copy()
    revenue = units = 0.0
    accepted = 0
    for k in order:
        i = obs.node[k]
        ok = accept_mask[k] if accept_mask is not None else obs.log_bid[k] >= reserve_log[i]
        if ok and remaining[i] > 0:
            take = min(obs.qty[k], remaining[i])
            remaining[i] -= take
            units += take
            price = prices[k] if prices is not None else float(np.exp(obs.log_bid[k]))
            revenue += take * float(price)
            accepted += 1
    terminal = float(np.sum(remaining * value * (1.0 - delta) ** horizon_days))
    return {
        "revenue": revenue,
        "units": units,
        "accepted_offers": accepted,
        "terminal_value": terminal,
        "total_value": revenue + terminal,
    }
