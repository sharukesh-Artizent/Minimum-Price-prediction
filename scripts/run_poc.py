"""CLI: `backtest [--walk-forward]` or `daily --date YYYY-MM-DD`."""
import argparse
import pickle

from reserve import config
from reserve.config import Params
from reserve.pipeline import daily_reserve, prepare, run_backtest


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    bt = sub.add_parser("backtest")
    bt.add_argument("--walk-forward", action="store_true")
    bt.add_argument("--no-cache", action="store_true")
    dy = sub.add_parser("daily")
    dy.add_argument("--date", required=True)
    args = ap.parse_args()

    p = Params()
    config.OUTPUT.mkdir(parents=True, exist_ok=True)
    D = prepare(p, use_cache=not getattr(args, "no_cache", False))
    if args.cmd == "backtest":
        w = run_backtest(D, p, walk_forward=args.walk_forward)
        tag = "walkforward" if args.walk_forward else "single"
        with open(config.OUTPUT / f"window_{tag}.pkl", "wb") as f:
            pickle.dump(w, f)
        from reserve.report import build_report
        path = build_report(w, config.OUTPUT, tag)
        print(f"report: {path}")
    else:
        out = daily_reserve(D, args.date, p)
        path = config.OUTPUT / f"reserve_{args.date}.csv"
        out.to_csv(path, index=False)
        print(f"{len(out)} reserves -> {path}")


if __name__ == "__main__":
    main()
