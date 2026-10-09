import pytest

from reserve.cli_phase2 import main
from reserve.config import Phase1Config, Phase2Config
from reserve.pipeline import prepare, run_phase1, run_phase2


def test_phase2_cli_writes_frontier_replay_and_reserves(data_dir, tmp_path):
    out = tmp_path / "out"
    table, replays, reserves = main([
        "--data-dir", str(data_dir), "--out-dir", str(out), "--train-end", "2025-10-23",
        "--train-days", "5", "--test-days", "3", "--multipliers", "0", "1.2"])
    for name in ("phase2_frontier.csv", "phase2_replay.csv", "phase2_replay_fair.csv", "phase2_reserves.csv"):
        assert (out / name).exists()
    assert table["multiplier"].tolist() == [0.0, 1.2]
    assert replays["policy"].tolist() == [
        "assurant_actual", "phase1_floor", "phase2_x0.0", "phase2_x1.2"]
    assert len(reserves) == 1


def test_run_phase2_empty_multipliers_raises_value_error(data_dir):
    p1, p2 = Phase1Config(), Phase2Config()
    ctx = prepare(data_dir, "2025-10-23", 5, 3, p1)
    _, _, phase1 = run_phase1(ctx, p1)
    with pytest.raises(ValueError, match="multipliers list must not be empty"):
        run_phase2(ctx, phase1, p1, p2, [])


def test_phase2_cli_fair_replay(data_dir, tmp_path):
    out = tmp_path / "out"
    table, replays, reserves = main([
        "--data-dir", str(data_dir), "--out-dir", str(out), "--train-end", "2025-10-23",
        "--train-days", "5", "--test-days", "3", "--multipliers", "0", "1.2",
        "--fair-replay"])
    for name in ("phase2_frontier.csv", "phase2_replay.csv", "phase2_replay_fair.csv", "phase2_reserves.csv"):
        assert (out / name).exists()
    assert replays["policy"].tolist() == [
        "assurant_actual", "phase1_floor", "phase2_x0.0", "phase2_x1.2"]


