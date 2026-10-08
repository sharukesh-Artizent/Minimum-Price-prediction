from reserve.cli_phase2 import main


def test_phase2_cli_writes_frontier_replay_and_reserves(data_dir, tmp_path):
    out = tmp_path / "out"
    table, replays, reserves = main([
        "--data-dir", str(data_dir), "--out-dir", str(out), "--train-end", "2025-10-23",
        "--train-days", "5", "--test-days", "3", "--multipliers", "0", "1.2"])
    for name in ("phase2_frontier.csv", "phase2_replay.csv", "phase2_reserves.csv"):
        assert (out / name).exists()
    assert table["multiplier"].tolist() == [0.0, 1.2]
    assert replays["policy"].tolist() == [
        "assurant_actual", "phase1_floor", "phase2_x0.0", "phase2_x1.2"]
    assert len(reserves) == 1
