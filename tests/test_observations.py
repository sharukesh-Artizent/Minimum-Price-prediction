import numpy as np
import pandas as pd

from reserve.observations import attach_nodes, node_priors, observations, window


def _nodes():
    return pd.DataFrame({"sku": ["1", "2", "3"], "warehouse": ["W1", "W1", "W1"],
                         "family": ["F", "F", "G"]})


def _offers():
    return pd.DataFrame({
        "sku": ["1", "1", "2", "9"],
        "warehouse": ["W1", "W1", "W1", "W1"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-09", "2026-01-10", "2026-01-10"]),
        "log_bid": np.log([100.0, 80.0, 120.0, 50.0]),
        "accepted": [1.0, 0.0, 1.0, 1.0],
        "qty": [2.0, 1.0, 3.0, 1.0],
    })


def test_attach_nodes_marks_unknown_minus_one():
    assert attach_nodes(_offers(), _nodes()).tolist() == [0, 0, 1, -1]


def test_window_is_half_open():
    w = window(_offers(), "2026-01-09", "2026-01-10")
    assert len(w) == 1


def test_observations_drop_unknown_and_weight_by_recency():
    obs = observations(_offers(), _nodes(), as_of="2026-01-11", tau_days=10.0)
    assert obs.node.tolist() == [0, 0, 1]
    assert np.allclose(obs.weight, np.exp(-np.array([10.0, 2.0, 1.0]) / 10.0))
    assert obs.qty.tolist() == [2.0, 1.0, 3.0]
    assert len(obs) == 3


def test_observations_infinite_tau_gives_unit_weights():
    obs = observations(_offers(), _nodes(), as_of="2026-01-11", tau_days=float("inf"))
    assert np.allclose(obs.weight, [1.0, 1.0, 1.0])


def test_node_priors_family_median_of_accepted_with_global_fallback():
    obs = observations(_offers(), _nodes(), as_of="2026-01-11", tau_days=10.0)
    prior = node_priors(obs, _nodes())
    family_f = np.median(np.log([100.0, 120.0]))
    assert np.allclose(prior, [family_f, family_f, family_f])


def test_node_priors_distinct_families():
    nodes = pd.DataFrame({
        "sku": ["1", "2", "3"],
        "warehouse": ["W1", "W1", "W1"],
        "family": ["F1", "F2", "F3"],
    })
    offers = pd.DataFrame({
        "sku": ["1", "2"],
        "warehouse": ["W1", "W1"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "log_bid": [10.0, 20.0],
        "accepted": [1.0, 1.0],
        "qty": [1.0, 1.0],
    })
    obs = observations(offers, nodes, as_of="2026-01-03", tau_days=10.0)
    prior = node_priors(obs, nodes)
    # F1 has 10.0, F2 has 20.0, F3 has none so global fallback is median([10.0, 20.0]) = 15.0
    assert np.allclose(prior, [10.0, 20.0, 15.0])


def test_node_priors_none_accepted():
    nodes = pd.DataFrame({
        "sku": ["1", "2"],
        "warehouse": ["W1", "W1"],
        "family": ["F", "F"],
    })
    offers = pd.DataFrame({
        "sku": ["1"],
        "warehouse": ["W1"],
        "date": pd.to_datetime(["2026-01-01"]),
        "log_bid": [5.0],
        "accepted": [0.0],
        "qty": [1.0],
    })
    obs = observations(offers, nodes, as_of="2026-01-02", tau_days=10.0)
    prior = node_priors(obs, nodes)
    assert np.allclose(prior, [5.0, 5.0])
