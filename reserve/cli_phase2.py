"""python -m reserve.cli_phase2 --train-end 2026-03-26"""
import argparse
import time
from pathlib import Path

from reserve.config import DATA_DIR, OUTPUT_DIR, Phase1Config, Phase2Config
from reserve.pipeline import prepare, run_phase1, run_phase2


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 2: ROI-optimal reserve via ADMM")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--train-end", default="2026-03-26")
    ap.add_argument("--train-days", type=int, default=56)
    ap.add_argument("--test-days", type=int, default=14)
    ap.add_argument("--multipliers", type=float, nargs="+", default=[0.0, 1.0, 1.1, 1.2, 1.3])
    ap.add_argument("--fair-replay", action="store_true", help="Run fair replay with redeemed counters and scaled stock")
    args = ap.parse_args(argv)

    started = time.perf_counter()
    p1, p2 = Phase1Config(), Phase2Config()
    ctx = prepare(args.data_dir, args.train_end, args.train_days, args.test_days, p1)
    _, _, phase1 = run_phase1(ctx, p1)
    table, replays, reserves, fair_replays = run_phase2(
        ctx, phase1, p1, p2, args.multipliers, return_both=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "phase2_frontier.csv", index=False)
    replays.to_csv(args.out_dir / "phase2_replay.csv", index=False)
    fair_replays.to_csv(args.out_dir / "phase2_replay_fair.csv", index=False)
    reserves.to_csv(args.out_dir / "phase2_reserves.csv", index=False)
    print("--- Frontier ---")
    print(table.to_string(index=False))
    print("\n--- Baseline Replay (Unscaled) ---")
    print(replays.to_string(index=False))
    print("\n--- Fair Replay (Counters Credited & Scaled Stock) ---")
    print(fair_replays.to_string(index=False))
    print(f"\nseconds: {time.perf_counter() - started:.1f}")
    return table, replays, reserves


if __name__ == "__main__":
    main()
