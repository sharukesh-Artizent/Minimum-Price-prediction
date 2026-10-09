"""Offer-level arrays keyed by node, plus family-median priors."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Observations:
    node: np.ndarray       # int64 node id per offer
    log_bid: np.ndarray
    accepted: np.ndarray   # 1.0 accepted, 0.0 countered or rejected
    weight: np.ndarray     # recency weight
    qty: np.ndarray
    date: np.ndarray       # datetime64[ns]
    counter_price: np.ndarray | None = None
    order_status: np.ndarray | None = None
    status: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.node)


def attach_nodes(frame: pd.DataFrame, nodes: pd.DataFrame) -> np.ndarray:
    """Node id for each (sku, warehouse) row of frame, -1 when not a node."""
    index = pd.MultiIndex.from_frame(nodes[["sku", "warehouse"]])
    return index.get_indexer(pd.MultiIndex.from_frame(frame[["sku", "warehouse"]]))


def window(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    """Rows with start <= date < end."""
    return frame[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] < pd.Timestamp(end))]


def observations(offers: pd.DataFrame, nodes: pd.DataFrame, as_of, tau_days: float) -> Observations:
    node = attach_nodes(offers, nodes)
    keep = node >= 0
    o = offers[keep]
    age = (pd.Timestamp(as_of) - o["date"]).dt.days.to_numpy(dtype=float)
    cp = o["counter_price"].to_numpy(dtype=float) if "counter_price" in o.columns else None
    os = o["order_status"].to_numpy() if "order_status" in o.columns else None
    st = o["status"].to_numpy() if "status" in o.columns else None
    return Observations(
        node=node[keep].astype(np.int64),
        log_bid=o["log_bid"].to_numpy(dtype=float),
        accepted=o["accepted"].to_numpy(dtype=float),
        weight=np.exp(-age / tau_days),
        qty=o["qty"].to_numpy(dtype=float),
        date=o["date"].to_numpy(),
        counter_price=cp,
        order_status=os,
        status=st,
    )


def node_priors(obs: Observations, nodes: pd.DataFrame) -> np.ndarray:
    """Family median log price of accepted bids; global median when a family has none."""
    family = nodes["family"].to_numpy()[obs.node]
    acc = obs.accepted == 1
    if acc.any():
        by_family = pd.Series(obs.log_bid[acc]).groupby(family[acc]).median()
        fallback = float(np.median(obs.log_bid[acc]))
    else:
        by_family = pd.Series(dtype=float)
        fallback = float(np.median(obs.log_bid)) if len(obs) else 0.0
    return nodes["family"].map(by_family).fillna(fallback).to_numpy(dtype=float)
