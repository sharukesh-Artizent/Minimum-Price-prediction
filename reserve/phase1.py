"""Phase 1: Assurant's implied floor per node, pooled across siblings with ADMM.

Each offer is a logistic observation: P(accept | bid) = sigmoid((log_bid - x_i) / T).
A node's data term is w_i * (recency-weighted mean log-loss), w_i = n_i / (n_i + k).
"""
from dataclasses import dataclass

import numpy as np

from reserve.admm_core import ADMMResult, LocalGrad, solve_network
from reserve.config import Phase1Config
from reserve.graph import NodeGraph
from reserve.observations import Observations
from reserve.pareto import node_tiers, pareto_weight


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


@dataclass
class Phase1Output:
    floor_log: np.ndarray
    node_weight: np.ndarray
    tier: np.ndarray
    result: ADMMResult


def logistic_local_grad(obs: Observations, node_weight: np.ndarray, n_nodes: int,
                        temperature: float) -> LocalGrad:
    total = np.bincount(obs.node, weights=obs.weight, minlength=n_nodes)
    coef = np.divide(node_weight, total, out=np.zeros(n_nodes), where=total > 0)

    def grad(x: np.ndarray) -> np.ndarray:
        z = (obs.log_bid - x[obs.node]) / temperature
        g = obs.weight * (obs.accepted - sigmoid(z)) / temperature
        return coef * np.bincount(obs.node, weights=g, minlength=n_nodes)

    return grad


def price_bracket(obs: Observations, prior: np.ndarray) -> tuple[float, float]:
    lows = [prior.min()] + ([obs.log_bid.min()] if len(obs) else [])
    highs = [prior.max()] + ([obs.log_bid.max()] if len(obs) else [])
    return float(min(lows)) - 1.0, float(max(highs)) + 1.0


def estimate_floor(obs: Observations, graph: NodeGraph, prior: np.ndarray, cfg: Phase1Config,
                   lam: float | None = None, x0=None) -> Phase1Output:
    n = graph.n_nodes
    counts = np.bincount(obs.node, minlength=n)
    weight = pareto_weight(counts, cfg.k_pareto)
    grad = logistic_local_grad(obs, weight, n, cfg.temperature)
    result = solve_network(
        n, grad, graph, prior, cfg.prior_weight, cfg.lam if lam is None else lam,
        cfg.rho, cfg.max_iter, cfg.eps, cfg.bisect_iters, price_bracket(obs, prior), x0)
    return Phase1Output(result.x, weight, node_tiers(counts), result)
