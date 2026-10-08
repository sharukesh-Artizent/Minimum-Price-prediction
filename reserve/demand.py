"""Phase-2 demand inputs per node: bid level/spread, arrival rate, units per offer."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from reserve.admm_core import LocalGrad, solve_network
from reserve.config import Phase1Config, Phase2Config
from reserve.graph import NodeGraph
from reserve.observations import Observations
from reserve.pareto import pareto_weight
from reserve.phase1 import price_bracket


@dataclass
class DemandParams:
    mu: np.ndarray       # location of log bids
    sigma: np.ndarray    # spread of log bids (family level)
    rate: np.ndarray     # offers per day
    qty: np.ndarray      # units per offer
    value: np.ndarray    # mean bid = exp(mu + sigma^2 / 2)


def squared_local_grad(obs: Observations, node_weight: np.ndarray, n_nodes: int) -> LocalGrad:
    total = np.bincount(obs.node, weights=obs.weight, minlength=n_nodes)
    coef = np.divide(node_weight, total, out=np.zeros(n_nodes), where=total > 0)

    def grad(x: np.ndarray) -> np.ndarray:
        g = obs.weight * (x[obs.node] - obs.log_bid)
        return coef * np.bincount(obs.node, weights=g, minlength=n_nodes)

    return grad


def _shrink_to_family(num: np.ndarray, den: np.ndarray, family: np.ndarray, pseudo: float,
                      default: float) -> np.ndarray:
    """(num + pseudo * family_ratio) / (den + pseudo), family_ratio = sum(num) / sum(den)."""
    fam_num = pd.Series(num).groupby(family).transform("sum").to_numpy()
    fam_den = pd.Series(den).groupby(family).transform("sum").to_numpy()
    ratio = np.divide(fam_num, fam_den, out=np.full(len(num), default), where=fam_den > 0)
    return (num + pseudo * ratio) / (den + pseudo)


def fit_demand(obs: Observations, graph: NodeGraph, prior: np.ndarray, p1: Phase1Config,
               p2: Phase2Config, train_days: int) -> DemandParams:
    n = graph.n_nodes
    family = graph.nodes["family"].to_numpy()
    counts = np.bincount(obs.node, minlength=n).astype(float)
    weight = pareto_weight(counts, p1.k_pareto)
    mu = solve_network(n, squared_local_grad(obs, weight, n), graph, prior, p1.prior_weight,
                       p1.lam, p1.rho, p1.max_iter, p1.eps, p1.bisect_iters,
                       price_bracket(obs, prior)).x

    resid = obs.log_bid - mu[obs.node]
    fam_sigma = pd.Series(resid).groupby(family[obs.node]).std()
    global_sigma = float(np.std(resid)) if len(resid) > 1 else p2.min_sigma
    sigma = (graph.nodes["family"].map(fam_sigma).fillna(global_sigma)
             .clip(lower=p2.min_sigma).to_numpy(dtype=float))

    days = np.full(n, float(train_days))
    rate = _shrink_to_family(counts, days, family, p2.rate_prior_days, 0.0)
    qty_sum = np.bincount(obs.node, weights=obs.qty, minlength=n)
    qty = _shrink_to_family(qty_sum, counts, family, p2.rate_prior_days, 1.0)
    return DemandParams(mu, sigma, rate, qty, np.exp(mu + 0.5 * sigma ** 2))
