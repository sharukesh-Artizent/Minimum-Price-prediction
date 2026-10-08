"""Network-lasso ADMM over the SKU graph.

minimize  sum_i data_i(x_i) + (gamma/2)(x_i - prior_i)^2
        + lam * sum_e a_e |x_src - x_dst - offset_e|
s.t.      x_src >= x_dst on ordered edges

Each edge keeps its own copy of both endpoints (z_src, z_dst) with scaled duals.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from reserve.graph import NodeGraph

LocalGrad = Callable[[np.ndarray], np.ndarray]


@dataclass
class ADMMResult:
    x: np.ndarray
    iterations: int
    converged: bool
    primal_residual: list = field(default_factory=list)
    dual_residual: list = field(default_factory=list)


def edge_prox(p, q, offset, ordered, theta):
    """Closed-form prox of theta*|t| (+ t >= -offset if ordered), t = z_src - z_dst - offset.

    p, q are the points (x + u) each endpoint copy is pulled toward.
    """
    delta = p - q - offset
    t = np.sign(delta) * np.maximum(np.abs(delta) - theta, 0.0)
    t = np.where(ordered, np.maximum(t, -offset), t)
    half = (delta - t) / 2.0
    return p - half, q + half


def _x_update(local_grad, prior, prior_weight, rho, deg, sum_v, lo, hi, iters):
    """Per-node 1-D convex minimisation by vectorised bisection on the derivative."""
    a, b = lo.copy(), hi.copy()
    for _ in range(iters):
        m = 0.5 * (a + b)
        g = local_grad(m) + prior_weight * (m - prior) + rho * (deg * m - sum_v)
        up = g > 0
        b = np.where(up, m, b)
        a = np.where(up, a, m)
    return 0.5 * (a + b)


def solve_network(n_nodes: int, local_grad: LocalGrad, graph: NodeGraph, prior: np.ndarray,
                  prior_weight: float, lam: float, rho: float, max_iter: int, eps: float,
                  bisect_iters: int, bracket: tuple[float, float], x0=None) -> ADMMResult:
    src, dst = graph.src, graph.dst
    n_edges = len(src)
    x = prior.copy() if x0 is None else np.asarray(x0, dtype=float).copy()
    z_s, z_d = x[src].copy(), x[dst].copy()
    u_s, u_d = np.zeros(n_edges), np.zeros(n_edges)
    deg = (np.bincount(src, minlength=n_nodes) + np.bincount(dst, minlength=n_nodes)).astype(float)
    theta = 2.0 * lam * graph.weight / rho
    lo, hi = np.full(n_nodes, bracket[0]), np.full(n_nodes, bracket[1])
    tol = eps * np.sqrt(max(2 * n_edges, 1))
    result = ADMMResult(x=x, iterations=0, converged=False)

    for it in range(1, max_iter + 1):
        sum_v = (np.bincount(src, weights=z_s - u_s, minlength=n_nodes)
                 + np.bincount(dst, weights=z_d - u_d, minlength=n_nodes))
        x = _x_update(local_grad, prior, prior_weight, rho, deg, sum_v, lo, hi, bisect_iters)
        zs_old, zd_old = z_s, z_d
        z_s, z_d = edge_prox(x[src] + u_s, x[dst] + u_d, graph.offset, graph.ordered, theta)
        u_s = u_s + x[src] - z_s
        u_d = u_d + x[dst] - z_d
        r = float(np.sqrt(np.sum((x[src] - z_s) ** 2 + (x[dst] - z_d) ** 2)))
        s = float(rho * np.sqrt(np.sum((z_s - zs_old) ** 2 + (z_d - zd_old) ** 2)))
        result.primal_residual.append(r)
        result.dual_residual.append(s)
        result.iterations = it
        if r < tol and s < tol:
            result.converged = True
            break

    result.x = x
    return result
