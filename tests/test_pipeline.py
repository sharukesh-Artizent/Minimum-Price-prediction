import numpy as np
import pandas as pd
import pytest

from factories import synthetic_world
from reserve.config import Params
from reserve.pipeline import daily_reserve, prepare_from_frames, run_window


@pytest.fixture(scope="module")
def world():
    o, s, pr = synthetic_world(0)
    return prepare_from_frames(o, s, pr, Params())


def test_run_window_produces_all_policies(world):
    w = run_window(world, "2026-03-01", "2026-04-09", Params(), extras=False)
    assert set(w.results) == {"ACTUAL", "B0", "B1", "P2", "ORACLE"}
    for name in ["B0", "B1", "P2"]:
        assert np.isfinite(w.groups[f"reserve_{name.lower()}"]).all(), name
    agree = (w.results["B0"].decision == w.ev.status).mean()
    assert agree > 0.80                                   # B0 recovers the known 0.9 floor policy
    oracle = w.results["ORACLE"].value.sum()
    # Oracle is the per-group best on the price grid; P2 can sit slightly off-grid after
    # the isotonic projection, so allow 1% slack.
    assert oracle >= 0.99 * w.results["P2"].value.sum()


def test_daily_reserve_one_row_per_active_combo(world):
    out = daily_reserve(world, "2026-04-09", Params())
    assert len(out) == 20                                  # 10 SKUs x 2 WH
    assert out.reserve_p2.gt(0).all()
    assert {"sku", "wh", "reserve_p2", "reserve_b0", "m", "W", "p2_p_sell", "reason"} <= set(out.columns)
