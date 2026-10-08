import numpy as np

from reserve.observations import Observations
from reserve.replay import replay


def _obs():
    return Observations(
        node=np.zeros(3, dtype=np.int64),
        log_bid=np.log([90.0, 50.0, 100.0]),
        accepted=np.array([0.0, 1.0, 1.0]),
        weight=np.ones(3),
        qty=np.array([3.0, 2.0, 3.0]),
        date=np.array(["2026-01-01"] * 3, dtype="datetime64[ns]"),
    )


def test_replay_fills_highest_bids_first_until_stock_runs_out():
    out = replay(
        np.log([80.0]),
        _obs(),
        np.array([5.0]),
        np.array([100.0]),
        np.array([0.0]),
        horizon_days=14,
    )
    assert out["units"] == 5.0
    assert np.isclose(out["revenue"], 3 * 100.0 + 2 * 90.0)
    assert out["accepted_offers"] == 2
    assert out["terminal_value"] == 0.0


def test_replay_marks_leftover_stock_with_depreciation():
    out = replay(
        np.log([95.0]),
        _obs(),
        np.array([5.0]),
        np.array([100.0]),
        np.array([0.01]),
        horizon_days=10,
    )
    assert out["units"] == 3.0
    assert np.isclose(out["terminal_value"], 2 * 100.0 * 0.99**10)


def test_replay_accept_mask_replays_actual_decisions():
    obs = _obs()
    out = replay(
        np.zeros(1),
        obs,
        np.array([10.0]),
        np.array([100.0]),
        np.array([0.0]),
        14,
        accept_mask=obs.accepted == 1,
    )
    assert np.isclose(out["revenue"], 3 * 100.0 + 2 * 50.0)
