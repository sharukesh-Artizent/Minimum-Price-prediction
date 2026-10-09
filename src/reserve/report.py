"""Markdown report + charts for a backtest window, including the spec's success criteria."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reserve.config import F1_MIN, F2_TOL, F3_TOL, S2_MAX_DROP
from reserve.metrics import bootstrap_uplift, deviation_bucket, segment_table
from reserve.replay import summarize

POLICIES = ["ACTUAL", "B0", "B1", "P2", "ORACLE"]


def explain(row) -> str:
    return (f"Market ${row.m:,.0f}; ~{row.tau:.0f}d to sell, holding value ${row.W:,.0f}; "
            f"reserve ${row.reserve_p2:,.0f} ({row.reserve_p2 / row.m:.0%} of market, "
            f"{row.reserve_p2 / row.reserve_b0 - 1:+.0%} vs current floor); P(sell) ~{row.p2_p_sell:.0%}")


def md_table(df: pd.DataFrame, floatfmt: str = "{:,.3f}") -> str:
    def fmt(v):
        return floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |",
             "|" + "|".join(["---"] * len(df.columns)) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def criteria(w) -> pd.DataFrame:
    r, ev = w.results, w.ev
    f1 = float((r["B0"].decision == ev.status).mean())
    f2 = float(r["B0"].revenue.sum() / r["ACTUAL"].revenue.sum() - 1)
    matched_rev = float((ev.qty * ev.ref_price)[ev.sale_match].sum())
    f3 = float(r["ACTUAL"].revenue.sum() * w.fitted.recall["__all__"] / matched_rev - 1)
    a = ev.segment == "A"
    s1 = bootstrap_uplift(r["P2"][a], r["B0"][a])
    s2 = summarize(r["P2"])["sell_through"] - summarize(r["B0"])["sell_through"]
    s3 = w.market_test_mape
    s4 = None if w.sensitivity is None else bool((w.sensitivity.uplift_vs_b0 > 0).all())
    rows = [
        ("F1", "B0 decision agreement", f1, f">= {F1_MIN}", f1 >= F1_MIN),
        ("F2", "B0 / ACTUAL revenue - 1", f2, f"within +/-{F2_TOL}", abs(f2) <= F2_TOL),
        ("F3", "ACTUAL x recall / matched revenue - 1", f3, f"within +/-{F3_TOL}", abs(f3) <= F3_TOL),
        ("S1", "P2 - B0 total value, segment A, 95% CI low", s1["ci_lo"], "> 0", s1["ci_lo"] > 0),
        ("S2", "P2 - B0 sell-through", s2, f">= -{S2_MAX_DROP}", s2 >= -S2_MAX_DROP),
        ("S3", "M1 test MAPE model vs naive", s3["model"] - s3["naive"], "<= 0", s3["model"] <= s3["naive"] + 1e-12),
        ("S4", "uplift sign stable under sensitivity", float(s4) if s4 is not None else float("nan"),
         "all > 0", bool(s4)),
    ]
    return pd.DataFrame(rows, columns=["id", "check", "value", "threshold", "pass"]).astype({"pass": bool})


def _frontier_png(w, path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    if w.frontier is not None:
        ax.plot(w.frontier.sell_through, w.frontier.revenue, "o-", label="P2 frontier (vary h)")
    for name in ["ACTUAL", "B0", "B1", "P2"]:
        s = summarize(w.results[name])
        ax.scatter(s["sell_through"], s["revenue"], label=name, zorder=3)
    ax.set_xlabel("sell-through (units sold / units bid)")
    ax.set_ylabel("revenue ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _curve_png(w, path: Path):
    c = w.fitted.curve
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, v in c.curves.items():
        ax.plot(c.grid, v, lw=2 if k == "__all__" else 0.8, label="Overall" if k == "__all__" else None)
    ax.set_xlabel("counter gap (c - b) / b")
    ax.set_ylabel("P(counter taken), recall-corrected")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_report(w, out_dir: Path, tag: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _frontier_png(w, out_dir / f"frontier_{tag}.png")
    _curve_png(w, out_dir / f"counter_curve_{tag}.png")
    F = w.fitted
    summary = pd.DataFrame([{"policy": k, **summarize(w.results[k])} for k in POLICIES])
    boots = pd.DataFrame([{"comparison": f"{a} vs {b}", **bootstrap_uplift(w.results[a], w.results[b])}
                          for a, b in [("P2", "B0"), ("P2", "ACTUAL"), ("B1", "B0"), ("ORACLE", "P2")]])
    seg = {k: w.results[k].assign(segment=w.ev.segment) for k in ["B0", "B1", "P2"]}
    grade = {k: w.results[k].assign(grade=w.ev.grade) for k in ["B0", "P2"]}
    dev = {"P2": w.results["P2"].assign(dev=deviation_bucket(w.ev.reserve_p2, w.ev.reserve_b0)),
           "B0": w.results["B0"].assign(dev=deviation_bucket(w.ev.reserve_p2, w.ev.reserve_b0))}
    sample = (w.groups.merge(F.segments, on=["sku", "wh"], how="left").query("segment == 'A'")
              .sort_values("nb", ascending=False).head(20))
    sample = sample.assign(reason=sample.apply(explain, axis=1))[
        ["sku", "wh", "day", "model", "capacity", "grade", "reserve_b0", "reserve_p2", "reason"]]

    parts = [
        f"# Reserve price POC — backtest ({tag})",
        f"Test offers: {len(w.ev):,} | groups: {len(w.groups):,} | delta/day: {F.delta:.5f} | "
        f"rho: {F.rho:.3f} | match recall (all): {F.recall['__all__']:.3f} | "
        f"counter max gap: {F.curve.max_gap:.3f}",
        "## Success criteria", md_table(criteria(w)),
        "## Policy comparison", md_table(summary),
        "### Bootstrap uplift (total value, day blocks)", md_table(boots),
        "## M1 market value MAPE",
        md_table(pd.DataFrame([{"split": "inner valid", **F.market_inner_mape},
                               {"split": "test", **w.market_test_mape}])),
        "## Pareto segments", md_table(segment_table(seg, "segment")),
        "### By grade", md_table(segment_table(grade, "grade")),
        "## Deviation from current floor", md_table(segment_table(dev, "dev")),
        "## Frontier", f"![frontier](frontier_{tag}.png)",
        md_table(w.frontier) if w.frontier is not None else "_not computed in walk-forward mode_",
        "## Counter take-up curve", f"![curve](counter_curve_{tag}.png)",
        "## Sensitivity",
        md_table(w.sensitivity) if w.sensitivity is not None else "_not computed in walk-forward mode_",
        "## Sample reserves (segment A, busiest groups)", md_table(sample, "{:,.2f}"),
    ]
    path = out_dir / f"report_{tag}.md"
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return path
