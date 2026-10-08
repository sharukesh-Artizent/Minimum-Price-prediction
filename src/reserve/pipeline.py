"""Orchestration: prepare data, fit models at a cutoff, score groups, replay, backtest, daily output."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from reserve import config
from reserve.bids import BidDist
from reserve.config import (DATA_START, DECISION_STATUSES, H_GRID, INNER_VALID_DAYS, KEYS,
                            SENSITIVITY, TEST_END, TEST_START, WARMUP_DAYS, Params)
from reserve.counter_curve import CounterCurve, fit_counter_curve
from reserve.economics import continuation_value, estimate_delta, estimate_rho
from reserve.features import attach_product, build_features
from reserve.floor_model import FloorModel, XGB_PARAMS
from reserve.labels import floor_bounds, match_offers_to_sales, match_recall

XGB_PARAMS["aft_loss_distribution_scale"] = 0.01

from reserve.loaders import load_all
from reserve.market import MarketModel, mape, realized_targets
from reserve.metrics import pareto_segment
from reserve.optimizer import optimize
from reserve.replay import hindsight_reserve, replay_actual, replay_policy, summarize
from reserve.rule_b1 import apply_rule, tau_edges_from, tune_rule
from reserve.siblings import enforce_monotone


@dataclass
class Data:
    offers: pd.DataFrame
    sales: pd.DataFrame
    product: pd.DataFrame
    matched: pd.DataFrame
    feats: pd.DataFrame


@dataclass
class Fitted:
    market: MarketModel
    delta: float
    rho: float
    recall: pd.Series
    curve: CounterCurve
    bids: BidDist
    floor: FloorModel
    rule: dict
    tau_edges: np.ndarray
    segments: pd.DataFrame
    market_inner_mape: dict


@dataclass
class WindowResult:
    results: dict
    ev: pd.DataFrame
    groups: pd.DataFrame
    fitted: Fitted
    market_test_mape: dict
    frontier: pd.DataFrame | None
    sensitivity: pd.DataFrame | None


def prepare_from_frames(offers, sales, product, p: Params) -> Data:
    matched = match_offers_to_sales(offers, sales, p.match_window_days, p.match_price_tol)
    queries = pd.concat([offers[KEYS], sales[KEYS]]).drop_duplicates()
    feats = build_features(queries, offers, sales, product, p)
    return Data(offers, sales, product, matched, feats)


def prepare(p: Params, use_cache: bool = True) -> Data:
    offers, sales, product = load_all(use_cache)
    f = config.INTERIM / "feats.parquet"
    m = config.INTERIM / "matched.parquet"
    if use_cache and f.exists() and m.exists():
        return Data(offers, sales, product, pd.read_parquet(m), pd.read_parquet(f))
    D = prepare_from_frames(offers, sales, product, p)
    D.feats.to_parquet(f, index=False)
    D.matched.to_parquet(m, index=False)
    return D


def enrich(feats: pd.DataFrame, F: Fitted, p: Params, h_pct=None, delta_mult: float = 1.0) -> pd.DataFrame:
    out = feats.copy()
    out["log_m"] = F.market.predict(out)
    out["m"] = np.exp(out.log_m)
    h = p.holding_cost_pct_per_day if h_pct is None else h_pct
    out["W"] = continuation_value(out.m, out.tau, F.delta * delta_mult, F.rho, h, p.salvage_floor_pct)
    return out


def fit_models(D: Data, cutoff, p: Params) -> Fitted:
    cutoff = pd.Timestamp(cutoff)
    inner = cutoff - pd.Timedelta(days=INNER_VALID_DAYS)
    warm = pd.Timestamp(DATA_START) + pd.Timedelta(days=WARMUP_DAYS)

    tgt = D.feats.merge(realized_targets(D.sales), on=KEYS)
    tr = tgt[(tgt.day >= warm) & (tgt.day < inner)]
    va = tgt[(tgt.day >= inner) & (tgt.day < cutoff)]
    market = MarketModel().fit(tr, va)
    inner_mape = {"naive": mape(va.y, va.log_m_naive), "model": mape(va.y, market.predict(va)),
                  "use_lgb": market.use_lgb}

    rho = estimate_rho(pd.DataFrame({"y": tr.y, "log_m": market.predict(tr)}))
    delta = estimate_delta(attach_product(D.sales, D.product), cutoff)
    recall = match_recall(D.matched, cutoff)
    curve = fit_counter_curve(D.matched, cutoff, recall)
    F = Fitted(market, delta, rho, recall, curve, None, None, {}, np.array([]),
               pareto_segment(D.offers[D.offers.day < cutoff]), inner_mape)

    feats = enrich(D.feats, F, p)
    offers_f = D.matched.merge(feats, on=KEYS)
    F.bids = BidDist.fit(offers_f, cutoff, p)

    bounds = floor_bounds(D.offers).merge(feats, on=KEYS)
    F.floor = FloorModel().fit(bounds[(bounds.day >= warm) & (bounds.day < inner)],
                               bounds[(bounds.day >= inner) & (bounds.day < cutoff)])

    F.tau_edges = tau_edges_from(feats[feats.day < cutoff])
    tune = offers_f[(offers_f.day >= inner) & (offers_f.day < cutoff) & offers_f.status.isin(DECISION_STATUSES)]
    F.rule = tune_rule(tune, curve, F.tau_edges)
    return F


def score_groups(F: Fitted, rows: pd.DataFrame, p: Params, h_pct=None, delta_mult: float = 1.0) -> pd.DataFrame:
    g = enrich(rows.reset_index(drop=True), F, p, h_pct, delta_mult)
    g["b0_ratio"] = F.floor.predict_ratio(g)
    g["reserve_b0"] = g.b0_ratio * g.m
    g["reserve_b1"] = apply_rule(g, F.rule, F.tau_edges)
    opt = optimize(g, F.bids.lookup(g), F.curve, p, b0_ratio=g.b0_ratio.to_numpy())
    g = pd.concat([g, opt.add_prefix("p2_")], axis=1)
    g = enforce_monotone(g.assign(reserve=g.p2_reserve))
    g["reserve_p2"] = g.reserve_mono
    return g.drop(columns=["reserve", "reserve_mono"])


GROUP_COLS = ["m", "W", "reserve_b0", "reserve_b1", "reserve_p2", "category", "model", "grade", "capacity"]


def _eval_offers(D: Data, groups: pd.DataFrame, F: Fitted, cutoff, eval_end) -> pd.DataFrame:
    ev = D.matched[(D.matched.day >= cutoff) & (D.matched.day <= eval_end)
                   & D.matched.status.isin(DECISION_STATUSES)]
    ev = ev.merge(groups[KEYS + GROUP_COLS], on=KEYS)
    ev = ev.merge(F.segments, on=["sku", "wh"], how="left")
    ev["segment"] = ev.segment.fillna("BC")
    return ev.reset_index(drop=True)


def run_window(D: Data, cutoff, eval_end, p: Params, extras: bool = True) -> WindowResult:
    cutoff, eval_end = pd.Timestamp(cutoff), pd.Timestamp(eval_end)
    F = fit_models(D, cutoff, p)
    keys_ev = D.matched[(D.matched.day >= cutoff) & (D.matched.day <= eval_end)][KEYS].drop_duplicates()
    rows = keys_ev.merge(D.feats, on=KEYS)
    groups = score_groups(F, rows, p)
    ev = _eval_offers(D, groups, F, cutoff, eval_end)

    results = {"ACTUAL": replay_actual(ev, F.curve)}
    for name in ["B0", "B1", "P2"]:
        results[name] = replay_policy(ev, ev[f"reserve_{name.lower()}"], F.curve)
    oracle = ev.merge(hindsight_reserve(ev, F.curve, p), on=KEYS, how="left")
    results["ORACLE"] = replay_policy(ev, oracle.reserve_oracle.to_numpy(), F.curve)

    tgt = groups.merge(realized_targets(D.sales), on=KEYS)
    market_test = {"naive": mape(tgt.y, tgt.log_m_naive), "model": mape(tgt.y, tgt.log_m)}

    frontier = sensitivity = None
    if extras:
        fr = []
        for h in H_GRID:
            gh = score_groups(F, rows, p, h_pct=h)
            r = replay_policy(ev, ev[KEYS].merge(gh[KEYS + ["reserve_p2"]], on=KEYS, how="left").reserve_p2, F.curve)
            fr.append({"h": h, **summarize(r)})
        frontier = pd.DataFrame(fr)
        se = []
        for h_mult, d_mult in SENSITIVITY:
            gs = score_groups(F, rows, p, h_pct=p.holding_cost_pct_per_day * h_mult, delta_mult=d_mult)
            r_new = ev[KEYS].merge(gs[KEYS + ["reserve_p2"]], on=KEYS, how="left").reserve_p2
            res = replay_policy(ev, r_new, F.curve)
            base_b0 = results["B0"].value.sum()
            se.append({"h_mult": h_mult, "delta_mult": d_mult,
                       "median_abs_reserve_change": float(np.median(np.abs(r_new / ev.reserve_p2 - 1))),
                       "uplift_vs_b0": float(res.value.sum() - base_b0),
                       "uplift_vs_b0_pct": float(res.value.sum() / base_b0 - 1)})
        sensitivity = pd.DataFrame(se)
    return WindowResult(results, ev, groups, F, market_test, frontier, sensitivity)


def run_backtest(D: Data, p: Params, walk_forward: bool = False) -> WindowResult:
    start, end = pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)
    if not walk_forward:
        return run_window(D, start, end, p, extras=True)
    parts = []
    cut = start
    while cut <= end:
        stop = min(cut + pd.Timedelta(days=6), end)
        parts.append(run_window(D, cut, stop, p, extras=False))
        cut = stop + pd.Timedelta(days=1)
    results = {k: pd.concat([w.results[k] for w in parts], ignore_index=True) for k in parts[0].results}
    mt = {k: float(np.mean([w.market_test_mape[k] for w in parts])) for k in parts[0].market_test_mape}
    return WindowResult(results, pd.concat([w.ev for w in parts], ignore_index=True),
                        pd.concat([w.groups for w in parts], ignore_index=True),
                        parts[-1].fitted, mt, None, None)


def daily_reserve(D: Data, date, p: Params) -> pd.DataFrame:
    from reserve.report import explain
    date = pd.Timestamp(date).normalize()
    F = fit_models(D, date, p)
    recent = D.offers[(D.offers.day < date) & (D.offers.day >= date - pd.Timedelta(days=p.bid_window_days))]
    q = recent[["sku", "wh"]].drop_duplicates().assign(day=date)
    rows = build_features(q, D.offers, D.sales, D.product, p)
    g = score_groups(F, rows, p)
    g["reason"] = g.apply(explain, axis=1)
    cols = ["sku", "wh", "day", "model", "capacity", "grade", "m", "W", "tau", "nb",
            "reserve_b0", "reserve_b1", "reserve_p2", "p2_p_sell", "p2_exp_value", "reason"]
    return g[cols].sort_values("nb", ascending=False).reset_index(drop=True)
