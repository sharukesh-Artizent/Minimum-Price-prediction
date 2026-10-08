"""python -m reserve.cli_phase1 --train-end 2026-03-26"""
import argparse
import json
import time
from pathlib import Path

from reserve.config import DATA_DIR, OUTPUT_DIR, Phase1Config
from reserve.pipeline import prepare, run_phase1


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Phase 1: ADMM floor baseline")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--train-end", default="2026-03-26")
    ap.add_argument("--train-days", type=int, default=56)
    ap.add_argument("--test-days", type=int, default=14)
    ap.add_argument("--lam", type=float, default=Phase1Config.lam)
    args = ap.parse_args(argv)

    started = time.perf_counter()
    cfg = Phase1Config(lam=args.lam)
    ctx = prepare(args.data_dir, args.train_end, args.train_days, args.test_days, cfg)
    metrics, floors, _ = run_phase1(ctx, cfg)
    metrics["seconds"] = round(time.perf_counter() - started, 1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    floors.to_csv(args.out_dir / "phase1_floors.csv", index=False)
    (args.out_dir / "phase1_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
