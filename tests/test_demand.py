import numpy as np
import pandas as pd

from reserve.config import Phase1Config, Phase2Config
from reserve.demand import DemandParams, fit_demand, squared_local_grad
from reserve.graph import NodeGraph
from reserve.observations import Observations


def test_fit_demand_rates_shrink_to_family_and_mu_tracks_bids():
    nodes = pd.DataFrame({"sku": ["1", "2"], "family": ["F", "F"]})
    graph = NodeGraph.from_edges(nodes, [])
    bids = np.array([90.0, 100.0, 110.0, 100.0])
    obs = Observations(node=np.zeros(4, dtype=np.int64), log_bid=np.log(bids),
                       accepted=np.ones(4), weight=np.ones(4), qty=np.full(4, 2.0),
                       date=np.full(4, np.datetime64("2026-01-01")))
    prior = np.log([60.0, 60.0])
    p1 = Phase1Config(max_iter=500, eps=1e-7)
    d = fit_demand(obs, graph, prior, p1, Phase2Config(), train_days=10)

    fam_rate = 4 / 20                       # 4 offers over 10 days x 2 nodes
    assert np.allclose(d.rate, [(4 + 7 * fam_rate) / 17, (0 + 7 * fam_rate) / 17])
    assert np.allclose(d.qty, [2.0, 2.0])
    assert abs(d.mu[0] - np.mean(np.log(bids))) < 0.02
    assert abs(d.mu[1] - np.log(60.0)) < 1e-3
    assert (d.sigma >= Phase2Config().min_sigma).all()
    assert np.allclose(d.value, np.exp(d.mu + d.sigma ** 2 / 2))


def test_squared_local_grad_direction_and_zeros():
    bids = np.array([100.0, 100.0])
    obs = Observations(node=np.array([0, 0], dtype=np.int64), log_bid=np.log(bids),
                       accepted=np.ones(2), weight=np.ones(2), qty=np.ones(2),
                       date=np.full(2, np.datetime64("2026-01-01")))
    node_weights = np.array([1.0, 1.0])
    grad_fn = squared_local_grad(obs, node_weights, n_nodes=2)

    # At exact mean log-bid, gradient at node 0 is zero; unobserved node 1 gradient is zero
    x_at_mean = np.array([np.log(100.0), np.log(50.0)])
    g = grad_fn(x_at_mean)
    assert np.isclose(g[0], 0.0)
    assert np.isclose(g[1], 0.0)

    # When x > log_bid, grad > 0
    x_above = np.array([np.log(110.0), np.log(50.0)])
    assert grad_fn(x_above)[0] > 0.0

    # When x < log_bid, grad < 0
    x_below = np.array([np.log(90.0), np.log(50.0)])
    assert grad_fn(x_below)[0] < 0.0


def test_fit_demand_multiple_families_and_edges():
    nodes = pd.DataFrame({"sku": ["1", "2", "3"], "family": ["F1", "F1", "F2"]})
    # Node 0 and 1 connected by warehouse edge
    graph = NodeGraph.from_edges(nodes, [(0, 1, 0.0, False, "warehouse")])
    bids = np.array([100.0, 100.0, 100.0, 100.0])
    # All 4 bids on node 0
    obs = Observations(node=np.zeros(4, dtype=np.int64), log_bid=np.log(bids),
                       accepted=np.ones(4), weight=np.ones(4), qty=np.array([1.0, 2.0, 3.0, 4.0]),
                       date=np.full(4, np.datetime64("2026-01-01")))
    prior = np.log([80.0, 80.0, 50.0])
    p1 = Phase1Config(lam=5.0, max_iter=500, eps=1e-7)
    d = fit_demand(obs, graph, prior, p1, Phase2Config(), train_days=10)

    assert isinstance(d, DemandParams)
    assert len(d.mu) == 3
    assert len(d.sigma) == 3
    assert len(d.rate) == 3
    assert len(d.qty) == 3
    assert len(d.value) == 3
    # Node 1 should be pulled toward node 0 by fusion edge
    assert d.mu[1] > prior[1]
    # Node 2 is isolated unobserved in family F2
    assert np.isclose(d.mu[2], prior[2], atol=1e-3)
    assert d.rate[2] == 0.0
