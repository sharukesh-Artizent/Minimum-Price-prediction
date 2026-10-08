import numpy as np
import pandas as pd

from reserve.admm_core import edge_prox, solve_network
from reserve.graph import NodeGraph


def _graph(n, edges):
    return NodeGraph.from_edges(pd.DataFrame({"sku": [str(i) for i in range(n)]}), edges)


def _quadratic(targets):
    targets = np.asarray(targets, dtype=float)
    return lambda x: x - targets   # gradient of 0.5 * (x - target)^2


def _solve(graph, targets, lam):
    n = graph.n_nodes
    return solve_network(n, _quadratic(targets), graph, prior=np.zeros(n), prior_weight=0.0,
                         lam=lam, rho=1.0, max_iter=2000, eps=1e-7, bisect_iters=60,
                         bracket=(-5.0, 5.0))


def test_edge_prox_zero_theta_unordered_is_identity():
    zs, zd = edge_prox(np.array([0.3]), np.array([1.2]), np.array([0.0]), np.array([False]), 0.0)
    assert np.allclose([zs[0], zd[0]], [0.3, 1.2])


def test_edge_prox_large_theta_fuses_at_offset_keeping_midpoint():
    zs, zd = edge_prox(np.array([0.0]), np.array([1.0]), np.array([0.2]), np.array([False]), 10.0)
    assert np.isclose(zs[0] - zd[0], 0.2)
    assert np.isclose(zs[0] + zd[0], 1.0)


def test_edge_prox_ordered_projects_violation_to_equality():
    zs, zd = edge_prox(np.array([0.0]), np.array([1.0]), np.array([0.1]), np.array([True]), 0.0)
    assert np.allclose([zs[0], zd[0]], [0.5, 0.5])


def test_edge_prox_ordered_leaves_satisfied_pair_alone():
    zs, zd = edge_prox(np.array([2.0]), np.array([1.0]), np.array([0.1]), np.array([True]), 0.0)
    assert np.allclose([zs[0], zd[0]], [2.0, 1.0])


def test_solve_without_edges_returns_local_minimisers():
    res = _solve(_graph(2, []), [0.0, 2.0], lam=1.0)
    assert res.converged
    assert np.allclose(res.x, [0.0, 2.0], atol=1e-4)


def test_strong_fusion_pulls_nodes_together():
    res = _solve(_graph(2, [(0, 1, 0.0, False, "t")]), [0.0, 2.0], lam=50.0)
    assert np.allclose(res.x, [1.0, 1.0], atol=1e-3)


def test_ordered_edge_enforces_ranking():
    res = _solve(_graph(2, [(0, 1, 0.0, True, "t")]), [0.0, 2.0], lam=0.0)
    assert np.allclose(res.x, [1.0, 1.0], atol=1e-3)
