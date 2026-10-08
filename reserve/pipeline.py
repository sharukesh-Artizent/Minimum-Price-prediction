"""End-to-end wiring: load data, build graph, run phase 1 (and phase 2)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from reserve.config import Phase1Config
from reserve.evaluate import decision_metrics, interval_coverage, metrics_by_tier
from reserve.graph import NodeGraph, build_graph, build_nodes
from reserve.loaders import load_offers, load_products, load_sales
from reserve.observations import Observations, node_priors, observations, window
from reserve.phase1 import Phase1Output, estimate_floor


@dataclass
class Context:
    offers: pd.DataFrame
    sales: pd.DataFrame
    nodes: pd.DataFrame
    graph: NodeGraph
    train_obs: Observations
    test_obs: Observations
    prior: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_end: pd.Timestamp


def prepare(data_dir: Path, train_end: str, train_days: int, test_days: int,
            cfg: Phase1Config) -> Context:
    offers = load_offers(data_dir)
    nodes = build_nodes(offers, load_products(data_dir))
    graph = build_graph(nodes, cfg)
    end = pd.Timestamp(train_end)
    start = end - pd.Timedelta(days=train_days)
    test_end = end + pd.Timedelta(days=test_days)
    train_obs = observations(window(offers, start, end), nodes, end, cfg.recency_tau_days)
    test_obs = observations(window(offers, end, test_end), nodes, end, float("inf"))
    return Context(offers, load_sales(data_dir), nodes, graph, train_obs, test_obs,
                   node_priors(train_obs, nodes), start, end, test_end)


def _report(floor_log: np.ndarray, ctx: Context, tiers: np.ndarray, cfg: Phase1Config) -> dict:
    return {
        "overall": decision_metrics(floor_log, ctx.test_obs, cfg.temperature),
        "by_tier": metrics_by_tier(floor_log, ctx.test_obs, tiers, cfg.temperature),
        "intervals": interval_coverage(floor_log, ctx.test_obs),
    }


def run_phase1(ctx: Context, cfg: Phase1Config) -> tuple[dict, pd.DataFrame, Phase1Output]:
    pooled = estimate_floor(ctx.train_obs, ctx.graph, ctx.prior, cfg)
    alone = estimate_floor(ctx.train_obs, ctx.graph, ctx.prior, cfg, lam=0.0)
    metrics = {
        "graph": {"nodes": ctx.graph.n_nodes, "edges": int(len(ctx.graph.src)),
                  "edges_by_kind": pd.Series(ctx.graph.kind).value_counts().to_dict()},
        "admm": {"iterations": pooled.result.iterations, "converged": pooled.result.converged},
        "family_prior": _report(ctx.prior, ctx, pooled.tier, cfg),
        "admm_no_fusion": _report(alone.floor_log, ctx, pooled.tier, cfg),
        "admm_fused": _report(pooled.floor_log, ctx, pooled.tier, cfg),
    }
    floors = ctx.nodes[["sku", "warehouse", "family", "grade", "capacity_gb"]].copy()
    floors["tier"] = pooled.tier
    floors["node_weight"] = pooled.node_weight
    floors["floor_price"] = np.exp(pooled.floor_log)
    floors["floor_price_no_fusion"] = np.exp(alone.floor_log)
    floors["family_prior_price"] = np.exp(ctx.prior)
    return metrics, floors, pooled
