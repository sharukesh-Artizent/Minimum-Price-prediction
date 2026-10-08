"""Phase-1 hold-out metrics: decision agreement and floor-interval coverage."""
import numpy as np
import pandas as pd

from reserve.observations import Observations
from reserve.phase1 import sigmoid


def decision_metrics(floor_log: np.ndarray, obs: Observations, temperature: float) -> dict:
    if len(obs) == 0:
        return {"n": 0}
    margin = obs.log_bid - floor_log[obs.node]
    pred = margin >= 0
    y = obs.accepted == 1
    tpr = float(np.mean(pred[y])) if y.any() else float("nan")
    tnr = float(np.mean(~pred[~y])) if (~y).any() else float("nan")
    p = np.clip(sigmoid(margin / temperature), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(obs)),
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float(np.nanmean([tpr, tnr])),
        "log_loss": float(-np.mean(np.where(y, np.log(p), np.log(1 - p)))),
    }


def interval_coverage(floor_log: np.ndarray, obs: Observations) -> dict:
    """Share of node-days with both an accept and a non-accept where L <= floor <= U."""
    df = pd.DataFrame({"node": obs.node, "date": obs.date, "log_bid": obs.log_bid,
                       "accepted": obs.accepted})
    upper = df[df["accepted"] == 1].groupby(["node", "date"])["log_bid"].min().rename("U")
    lower = df[df["accepted"] == 0].groupby(["node", "date"])["log_bid"].max().rename("L")
    both = pd.concat([lower, upper], axis=1).dropna()
    if both.empty:
        return {"n_intervals": 0}
    x = floor_log[both.index.get_level_values("node").to_numpy()]
    lo, hi = both["L"].to_numpy(), both["U"].to_numpy()
    gap = np.maximum(lo - x, 0.0) + np.maximum(x - hi, 0.0)
    return {"n_intervals": int(len(both)),
            "coverage": float(np.mean(gap == 0.0)),
            "mean_log_gap": float(np.mean(gap))}


def subset(obs: Observations, mask: np.ndarray) -> Observations:
    return Observations(obs.node[mask], obs.log_bid[mask], obs.accepted[mask],
                        obs.weight[mask], obs.qty[mask], obs.date[mask])


def metrics_by_tier(floor_log: np.ndarray, obs: Observations, tiers: np.ndarray,
                    temperature: float) -> dict:
    node_tier = tiers[obs.node]
    return {str(t): decision_metrics(floor_log, subset(obs, node_tier == t), temperature)
            for t in sorted(set(node_tier))}
