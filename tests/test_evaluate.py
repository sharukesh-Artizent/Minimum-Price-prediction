import numpy as np

from reserve.evaluate import decision_metrics, interval_coverage, metrics_by_tier, subset
from reserve.observations import Observations


def _obs():
    # node 0 day 1: rejected 80, accepted 100 -> interval [log80, log100]
    # node 1 day 1: rejected 50, accepted 60  -> interval [log50, log60]
    return Observations(
        node=np.array([0, 0, 1, 1]),
        log_bid=np.log([80.0, 100.0, 50.0, 60.0]),
        accepted=np.array([0.0, 1.0, 0.0, 1.0]),
        weight=np.ones(4), qty=np.ones(4),
        date=np.array(["2026-01-01"] * 4, dtype="datetime64[ns]"))


def test_decision_metrics_perfect_floor():
    floor = np.log([90.0, 55.0])
    m = decision_metrics(floor, _obs(), temperature=0.03)
    assert m["n"] == 4
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["log_loss"] > 0


def test_decision_metrics_empty():
    obs = _obs()
    empty = Observations(*(a[:0] for a in (obs.node, obs.log_bid, obs.accepted,
                                           obs.weight, obs.qty, obs.date)))
    assert decision_metrics(np.zeros(2), empty, 0.03) == {"n": 0}


def test_decision_metrics_single_class():
    obs = Observations(
        node=np.array([0, 0]),
        log_bid=np.log([80.0, 100.0]),
        accepted=np.array([1.0, 1.0]),
        weight=np.ones(2), qty=np.ones(2),
        date=np.array(["2026-01-01"] * 2, dtype="datetime64[ns]"))
    floor = np.log([90.0])
    m = decision_metrics(floor, obs, 0.03)
    assert m["n"] == 2
    assert m["accuracy"] == 0.5
    # tpr is 0.5, tnr is nan -> nanmean is 0.5
    assert m["balanced_accuracy"] == 0.5


def test_interval_coverage_counts_inside_and_gap():
    floor = np.log([90.0, 70.0])          # node 1 floor above its interval
    m = interval_coverage(floor, _obs())
    assert m["n_intervals"] == 2
    assert m["coverage"] == 0.5
    assert np.isclose(m["mean_log_gap"], (np.log(70.0) - np.log(60.0)) / 2)


def test_interval_coverage_empty_or_no_both():
    obs = Observations(
        node=np.array([0, 0]),
        log_bid=np.log([80.0, 100.0]),
        accepted=np.array([1.0, 1.0]),
        weight=np.ones(2), qty=np.ones(2),
        date=np.array(["2026-01-01"] * 2, dtype="datetime64[ns]"))
    assert interval_coverage(np.zeros(1), obs) == {"n_intervals": 0}


def test_subset():
    obs = _obs()
    mask = obs.node == 0
    sub = subset(obs, mask)
    assert len(sub) == 2
    np.testing.assert_array_equal(sub.node, np.array([0, 0]))


def test_metrics_by_tier_splits_offers():
    m = metrics_by_tier(np.log([90.0, 55.0]), _obs(), np.array(["A", "B"], dtype=object), 0.03)
    assert set(m) == {"A", "B"}
    assert m["A"]["n"] == 2
    assert m["B"]["n"] == 2
