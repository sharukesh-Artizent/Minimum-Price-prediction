import numpy as np
import pandas as pd

from reserve.config import Phase2Config
from reserve.demand import DemandParams
from reserve.graph import NodeGraph
from reserve.phase2 import (baseline_targets, build_economics, expected_outcomes, frontier,
                            node_value, solve_phase2)


def _econ(n=1, mu=None, inventory=None, warehouse=None, baseline=None, delta=0.002):
    mu = np.log(np.full(n, 100.0)) if mu is None else np.asarray(mu, dtype=float)
    sigma = np.full(n, 0.1)
    demand = DemandParams(mu=mu, sigma=sigma, rate=np.full(n, 3.0), qty=np.ones(n),
                          value=np.exp(mu + sigma ** 2 / 2))
    inventory = np.full(n, 5.0) if inventory is None else np.asarray(inventory, dtype=float)
    warehouse = np.zeros(n, dtype=int) if warehouse is None else np.asarray(warehouse)
    baseline = mu.copy() if baseline is None else np.asarray(baseline, dtype=float)
    return build_economics(demand, np.full(n, delta), inventory, baseline, np.ones(n),
                           warehouse, Phase2Config())


def _graph(n, edges=()):
    return NodeGraph.from_edges(pd.DataFrame({"sku": [str(i) for i in range(n)]}), list(edges))


def test_expected_outcomes_low_reserve_sells_demand_at_mean_bid():
    sold, revenue = expected_outcomes(np.log(1.0), np.log(100.0), 0.1, 2.0, 1.0, 10.0)
    assert np.isclose(sold, 2.0)
    assert np.isclose(revenue, 2.0 * 100.0 * np.exp(0.005))


def test_expected_outcomes_stock_caps_units():
    sold, revenue = expected_outcomes(np.log(1.0), np.log(100.0), 0.1, 2.0, 1.0, 1.0)
    assert np.isclose(sold, 1.0)
    assert np.isclose(revenue, 100.0 * np.exp(0.005))


def test_expected_outcomes_high_reserve_sells_nothing():
    sold, revenue = expected_outcomes(np.log(1000.0), np.log(100.0), 0.1, 2.0, 1.0, 10.0)
    assert sold < 1e-12 and revenue < 1e-9


def test_more_stock_lowers_continuation_value():
    econ = _econ(n=2, inventory=[1.0, 500.0])
    assert econ.continuation[1] < econ.continuation[0]


def test_unconstrained_solve_matches_grid_argmax():
    econ = _econ()
    res = solve_phase2(econ, _graph(1), np.zeros(1), Phase2Config())
    grid = np.linspace(econ.lo[0], econ.hi[0], Phase2Config().grid_points)
    best = grid[np.argmax(node_value(grid, econ)[0])]
    assert np.isclose(res.logr[0], best)


def test_higher_clearance_target_lowers_reserve_and_sells_more():
    econ = _econ(n=2)
    free = solve_phase2(econ, _graph(2), np.zeros(1), Phase2Config())
    push = solve_phase2(econ, _graph(2), baseline_targets(econ, 1.3), Phase2Config(max_iter=500))
    assert push.sold.sum() > free.sold.sum()
    assert np.all(push.logr <= free.logr + 1e-12)
    assert push.dual[0] > 0


def test_ordered_edge_keeps_better_sibling_priced_higher():
    # node 1 has richer bids, but node 0 is declared the more valuable sibling
    econ = _econ(n=2, mu=np.log([100.0, 115.0]), baseline=np.log([108.0, 108.0]))
    g = _graph(2, [(0, 1, 0.0, True, "grade")])
    loose = solve_phase2(econ, _graph(2), np.zeros(1), Phase2Config())
    tied = solve_phase2(econ, g, np.zeros(1), Phase2Config(max_iter=1000))
    assert loose.logr[0] < loose.logr[1]
    step = (econ.hi[0] - econ.lo[0]) / (Phase2Config().grid_points - 1)
    assert tied.logr[0] >= tied.logr[1] - step
    assert tied.ordering_violations == 0


def test_frontier_rows_per_multiplier():
    econ = _econ(n=2)
    table, results = frontier(econ, _graph(2), [0.0, 1.2], Phase2Config())
    assert table["multiplier"].tolist() == [0.0, 1.2]
    assert table.loc[1, "expected_units"] >= table.loc[0, "expected_units"]
    assert set(results) == {0.0, 1.2}
