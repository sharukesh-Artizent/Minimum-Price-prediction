import numpy as np
import pandas as pd

from reserve.config import Phase1Config
from reserve.graph import NodeGraph
from reserve.observations import Observations
from reserve.phase1 import estimate_floor, logistic_local_grad, price_bracket, sigmoid


def _obs(node, bids, accepted):
    n = len(node)
    return Observations(node=np.asarray(node, dtype=np.int64), log_bid=np.log(bids),
                        accepted=np.asarray(accepted, dtype=float), weight=np.ones(n),
                        qty=np.ones(n), date=np.full(n, np.datetime64("2026-01-01")))


def _graph(n, edges):
    return NodeGraph.from_edges(pd.DataFrame({"sku": [str(i) for i in range(n)]}), edges)


# node 0: accepted at 100, countered/rejected at 80 -> floor somewhere in (80, 100)
DATA = _obs([0, 0, 0, 0, 0, 0], [100.0, 100.0, 100.0, 80.0, 80.0, 80.0], [1, 1, 1, 0, 0, 0])


def test_logistic_grad_pushes_floor_down_when_accepted_bid_below_it():
    obs = _obs([0], [100.0], [1])
    grad = logistic_local_grad(obs, np.array([1.0]), 1, temperature=0.03)
    assert grad(np.array([np.log(120.0)]))[0] > 0


def test_single_node_floor_between_rejected_and_accepted():
    prior = np.array([np.log(50.0)])
    out = estimate_floor(DATA, _graph(1, []), prior, Phase1Config())
    assert np.log(80.0) < out.floor_log[0] < np.log(100.0)
    assert out.tier.tolist() == ["A"]


def test_fusion_lends_floor_to_node_without_data():
    prior = np.array([np.log(50.0), np.log(50.0)])
    g = _graph(2, [(0, 1, 0.0, False, "warehouse")])
    out = estimate_floor(DATA, g, prior, Phase1Config(lam=5.0, max_iter=2000, eps=1e-6))
    assert abs(out.floor_log[1] - out.floor_log[0]) < 0.05
    assert out.tier.tolist() == ["A", "unseen"]


def test_without_fusion_dataless_node_stays_at_prior():
    prior = np.array([np.log(50.0), np.log(50.0)])
    g = _graph(2, [(0, 1, 0.0, False, "warehouse")])
    out = estimate_floor(DATA, g, prior, Phase1Config(max_iter=2000, eps=1e-6), lam=0.0)
    assert abs(out.floor_log[1] - np.log(50.0)) < 1e-3


def test_sigmoid():
    assert np.isclose(sigmoid(0.0), 0.5)
    assert np.isclose(sigmoid(50.0), 1.0)
    assert np.isclose(sigmoid(-50.0), 0.0)


def test_price_bracket_empty_obs():
    prior = np.array([np.log(50.0), np.log(100.0)])
    empty_obs = _obs([], [], [])
    lo, hi = price_bracket(empty_obs, prior)
    assert np.isclose(lo, np.log(50.0) - 1.0)
    assert np.isclose(hi, np.log(100.0) + 1.0)
