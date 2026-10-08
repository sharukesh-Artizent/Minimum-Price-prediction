"""Replay hold-out offers against a reserve policy with a stock cap per node."""
import numpy as np

from reserve.observations import Observations


def replay(
    reserve_log: np.ndarray,
    obs: Observations,
    inventory: np.ndarray,
    value: np.ndarray,
    delta: np.ndarray,
    horizon_days: int,
    accept_mask: np.ndarray | None = None,
) -> dict:
    """Walk offers by date, node, highest bid first; accept if bid >= reserve and stock left.

    accept_mask overrides the reserve rule (used to replay Assurant's actual decisions).
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
            revenue += take * float(np.exp(obs.log_bid[k]))
            accepted += 1
    terminal = float(np.sum(remaining * value * (1.0 - delta) ** horizon_days))
    return {
        "revenue": revenue,
        "units": units,
        "accepted_offers": accepted,
        "terminal_value": terminal,
        "total_value": revenue + terminal,
    }
