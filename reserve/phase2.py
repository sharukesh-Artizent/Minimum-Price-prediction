"""Phase 2: ROI-optimal reserve per node with ADMM + warehouse clearance duals.

maximize  sum_i  f_i(y_i)                       y_i = log reserve
s.t.      sum_{i in w} sold_i(y_i) >= T_w        warehouse clearance target (dual mu_w)
          y_src >= y_dst on ordered sibling edges (ADMM edge copies)
          y_i in [baseline_i - kappa_i, baseline_i + kappa_i]

f_i(y) = expected revenue today + unsold units * continuation value per unit.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import ndtr

from reserve.admm_core import edge_prox
from reserve.config import Phase2Config
from reserve.demand import DemandParams
from reserve.graph import NodeGraph


@dataclass
class Economics:
    mu: np.ndarray
    sigma: np.ndarray
    rate: np.ndarray
    qty: np.ndarray
    value: np.ndarray
    inventory: np.ndarray
    continuation: np.ndarray   # value of holding one unit past today
    baseline_logr: np.ndarray  # phase-1 floor
    warehouse: np.ndarray      # int code per node
    lo: np.ndarray
    hi: np.ndarray


@dataclass
class Phase2Result:
    logr: np.ndarray
    sold: np.ndarray
    revenue: np.ndarray
    objective: np.ndarray
    dual: np.ndarray
    iterations: int
    converged: bool
    ordering_violations: int
    primal_residual: list = field(default_factory=list)


def expected_outcomes(logr, mu, sigma, rate, qty, inventory):
    """Expected units sold and revenue in one day when bids >= exp(logr) are accepted.

    Log bids ~ N(mu, sigma^2), offers ~ Poisson(rate), qty units each. When stock binds,
    qualifying bids are filled at random, so revenue scales with sold / demand.
    """
    z = (logr - mu) / sigma
    survive = 1.0 - ndtr(z)
    demand = rate * qty * survive
    sold = np.minimum(inventory, demand)
    fill = np.divide(sold, demand, out=np.zeros_like(demand), where=demand > 0)
    revenue = np.exp(mu + 0.5 * sigma ** 2) * ndtr(sigma - z) * rate * qty * fill
    return sold, revenue


def _col(a: np.ndarray, like: np.ndarray) -> np.ndarray:
    return a[:, None] if like.ndim == 2 else a


def node_value(logr: np.ndarray, econ: Economics):
    """(objective, sold, revenue); logr is (N,) or (N, G)."""
    sold, revenue = expected_outcomes(
        logr, _col(econ.mu, logr), _col(econ.sigma, logr), _col(econ.rate, logr),
        _col(econ.qty, logr), _col(econ.inventory, logr))
    objective = revenue + (_col(econ.inventory, logr) - sold) * _col(econ.continuation, logr)
    return objective, sold, revenue


def build_economics(demand: DemandParams, delta: np.ndarray, inventory: np.ndarray,
                    baseline_logr: np.ndarray, node_weight: np.ndarray,
                    warehouse: np.ndarray, cfg: Phase2Config) -> Economics:
    kappa = cfg.kappa_min + (cfg.kappa_max - cfg.kappa_min) * node_weight
    survive = 1.0 - ndtr((baseline_logr - demand.mu) / demand.sigma)
    daily_units = np.maximum(demand.rate * demand.qty * survive, 1e-6)
    days = np.clip(inventory / daily_units, 1.0, cfg.max_days_to_clear)
    continuation = demand.value * np.clip(1.0 - delta - cfg.holding_frac, 0.0, 1.0) ** days
    return Economics(demand.mu, demand.sigma, demand.rate, demand.qty, demand.value,
                     inventory.astype(float), continuation, baseline_logr,
                     warehouse.astype(np.int64), baseline_logr - kappa, baseline_logr + kappa)


def baseline_targets(econ: Economics, multiplier: float) -> np.ndarray:
    _, sold, _ = node_value(econ.baseline_logr, econ)
    n_wh = int(econ.warehouse.max()) + 1
    return multiplier * np.bincount(econ.warehouse, weights=sold, minlength=n_wh)


def solve_phase2(econ: Economics, graph: NodeGraph, targets: np.ndarray,
                 cfg: Phase2Config) -> Phase2Result:
    n = len(econ.mu)
    keep = graph.ordered
    src, dst, off = graph.src[keep], graph.dst[keep], graph.offset[keep]
    ordered, theta = np.ones(len(src), dtype=bool), np.zeros(len(src))
    rows = np.arange(n)

    grid = econ.lo[:, None] + (econ.hi - econ.lo)[:, None] * np.linspace(0, 1, cfg.grid_points)
    g_obj, g_sold, _ = node_value(grid, econ)
    scale = (econ.value * np.maximum(econ.inventory, 1.0))[:, None]
    active = econ.inventory > 0
    hold_baseline = -((grid - econ.baseline_logr[:, None]) ** 2)

    n_wh = len(targets)
    dual = np.zeros(n_wh)
    y = econ.baseline_logr.copy()
    z_s, z_d = y[src].copy(), y[dst].copy()
    u_s, u_d = np.zeros(len(src)), np.zeros(len(src))
    deg = (np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)).astype(float)
    tol = cfg.eps * np.sqrt(max(2 * len(src), 1))
    res = Phase2Result(y, np.zeros(n), np.zeros(n), np.zeros(n), dual, 0, False, 0)

    for it in range(1, cfg.max_iter + 1):
        sum_v = (np.bincount(src, weights=z_s - u_s, minlength=n)
                 + np.bincount(dst, weights=z_d - u_d, minlength=n))
        score = ((g_obj + dual[econ.warehouse][:, None] * g_sold) / scale
                 - 0.5 * cfg.rho * (deg[:, None] * grid ** 2 - 2.0 * grid * sum_v[:, None]))
        score = np.where(active[:, None], score, hold_baseline)
        pick = score.argmax(axis=1)
        y = grid[rows, pick]
        sold = g_sold[rows, pick]

        wh_sold = np.bincount(econ.warehouse, weights=sold, minlength=n_wh)
        dual = np.maximum(0.0, dual + cfg.dual_step * (targets - wh_sold) / np.maximum(targets, 1.0))

        zs_old, zd_old = z_s, z_d
        z_s, z_d = edge_prox(y[src] + u_s, y[dst] + u_d, off, ordered, theta)
        u_s = u_s + y[src] - z_s
        u_d = u_d + y[dst] - z_d
        r = float(np.sqrt(np.sum((y[src] - z_s) ** 2 + (y[dst] - z_d) ** 2)))
        s = float(cfg.rho * np.sqrt(np.sum((z_s - zs_old) ** 2 + (z_d - zd_old) ** 2)))
        res.primal_residual.append(r)
        res.iterations = it
        met = np.all((wh_sold >= targets * (1 - 1e-3)) | (dual == 0))
        if r < tol and s < tol and met and it > 1:
            res.converged = True
            break

    objective, sold, revenue = node_value(y, econ)
    res.logr, res.sold, res.revenue, res.objective, res.dual = y, sold, revenue, objective, dual
    step = (econ.hi - econ.lo) / (cfg.grid_points - 1)
    res.ordering_violations = int(np.sum(y[src] < y[dst] - np.maximum(step[src], step[dst])))
    return res


def frontier(econ: Economics, graph: NodeGraph, multipliers: list[float],
             cfg: Phase2Config) -> tuple[pd.DataFrame, dict[float, Phase2Result]]:
    """Sweep the clearance target; each solve is one point of recovery vs units cleared."""
    rows, results = [], {}
    for m in multipliers:
        res = solve_phase2(econ, graph, baseline_targets(econ, m), cfg)
        results[m] = res
        rows.append({
            "multiplier": m,
            "expected_units": float(res.sold.sum()),
            "expected_revenue": float(res.revenue.sum()),
            "expected_value": float(res.objective.sum()),
            "mean_price_change_pct": float(np.mean(np.exp(res.logr - econ.baseline_logr)) - 1) * 100,
            "binding_warehouses": int(np.sum(res.dual > 0)),
            "iterations": res.iterations,
            "converged": res.converged,
            "ordering_violations": res.ordering_violations,
        })
    return pd.DataFrame(rows), results
