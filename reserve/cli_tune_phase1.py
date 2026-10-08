"""python -m reserve.cli_tune_phase1 --train-end 2026-03-12

Grid over lam x rho on a validation window that ends before the phase-1 test window,
so the chosen hyperparameters never see the final hold-out.
"""
import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import pandas as pd

from reserve.config import DATA_DIR, OUTPUT_DIR, Phase1Config
from reserve.evaluate import decision_metrics, interval_coverage
from reserve.phase1 import estimate_floor
from reserve.pipeline import prepare


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description="Phase 1 hyperparameter sweep")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--train-end", default="2026-03-12")
    ap.add_argument("--train-days", type=int, default=56)
    ap.add_argument("--test-days", type=int, default=14)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--max-iter", type=int, default=600)
    args = ap.parse_args(argv)

    base = Phase1Config(max_iter=args.max_iter)
    ctx = prepare(args.data_dir, args.train_end, args.train_days, args.test_days, base)
    rows = []
    for lam, rho in itertools.product(args.lams, args.rhos):
        cfg = replace(base, lam=lam, rho=rho)
        out = estimate_floor(ctx.train_obs, ctx.graph, ctx.prior, cfg)
        metrics = decision_metrics(out.floor_log, ctx.test_obs, cfg.temperature)
        coverage = interval_coverage(out.floor_log, ctx.test_obs)
        rows.append({"lam": lam, "rho": rho, "iterations": out.result.iterations,
                     "converged": out.result.converged,
                     **{f"val_{k}": v for k, v in metrics.items()},
                     **{f"interval_{k}": v for k, v in coverage.items()}})
        print(rows[-1], flush=True)

    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "phase1_tuning.csv", index=False)
    return table


if __name__ == "__main__":
    main()
