"""SKU x Warehouse nodes and the sibling edges the ADMM solvers share."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from reserve.config import GRADE_RANK, Phase1Config


@dataclass
class NodeGraph:
    nodes: pd.DataFrame      # one row per node, index 0..N-1
    src: np.ndarray          # edge endpoint; the more valuable node on ordered edges
    dst: np.ndarray
    offset: np.ndarray       # expected x[src] - x[dst] in log price
    ordered: np.ndarray      # True => enforce x[src] >= x[dst]
    weight: np.ndarray
    kind: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @classmethod
    def from_edges(cls, nodes: pd.DataFrame, edges: list[tuple]) -> "NodeGraph":
        """edges: (src, dst, offset, ordered, kind) tuples."""
        if edges:
            src, dst, offset, ordered, kind = zip(*edges)
        else:
            src = dst = offset = ordered = kind = ()
        return cls(
            nodes=nodes.reset_index(drop=True),
            src=np.asarray(src, dtype=np.int64),
            dst=np.asarray(dst, dtype=np.int64),
            offset=np.asarray(offset, dtype=float),
            ordered=np.asarray(ordered, dtype=bool),
            weight=np.ones(len(src)),
            kind=np.asarray(kind, dtype=object),
        )


def parse_capacity_gb(text) -> float:
    t = str(text).strip().upper()
    try:
        if t.endswith("TB"):
            return float(t[:-2]) * 1024.0
        if t.endswith("GB"):
            value = float(t[:-2])
            return 1024.0 if value in (1000.0, 1024.0) else value
    except ValueError:
        pass
    return float("nan")


def build_nodes(offers: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    keys = (offers[["sku", "warehouse"]].drop_duplicates()
            .sort_values(["sku", "warehouse"]).reset_index(drop=True))
    nodes = keys.merge(products, on="sku", how="left")
    for col in ("manufacturer", "model", "carrier", "lock_status", "capacity", "grade"):
        nodes[col] = nodes[col].fillna("")
    nodes["capacity_gb"] = nodes["capacity"].map(parse_capacity_gb)
    nodes["grade_rank"] = nodes["grade"].map(GRADE_RANK)
    nodes["family"] = (nodes["manufacturer"] + "|" + nodes["model"] + "|"
                       + nodes["carrier"] + "|" + nodes["lock_status"])
    return nodes


def _consecutive_pairs(df: pd.DataFrame, group_cols: list[str], sort_cols: list[str]):
    """(first, second) node ids of neighbours in sort order within each group."""
    s = df.sort_values(group_cols + sort_cols, kind="mergesort")
    ids = s.index.to_numpy()
    if len(ids) < 2:
        return ids[:0], ids[:0]
    same = np.ones(len(ids) - 1, dtype=bool)
    for col in group_cols:
        v = s[col].to_numpy()
        same &= v[1:] == v[:-1]
    return ids[:-1][same], ids[1:][same]


def build_graph(nodes: pd.DataFrame, cfg: Phase1Config) -> NodeGraph:
    edges: list[tuple] = []

    a, b = _consecutive_pairs(nodes, ["sku"], ["warehouse"])
    edges += [(i, j, 0.0, False, "warehouse") for i, j in zip(a, b)]

    known = nodes[nodes["model"] != ""]
    a, b = _consecutive_pairs(known[known["capacity_gb"].notna()],
                              ["family", "grade", "warehouse", "capacity_gb"], ["sku"])
    edges += [(i, j, 0.0, False, "variant") for i, j in zip(a, b)]

    reps = known[known["capacity_gb"].notna()].drop_duplicates(
        ["family", "grade", "warehouse", "capacity_gb"])
    lo, hi = _consecutive_pairs(reps, ["family", "grade", "warehouse"], ["capacity_gb"])
    cap = nodes["capacity_gb"].to_numpy()
    edges += [(h, l, cfg.capacity_step_log * float(np.log2(cap[h] / cap[l])), True, "capacity")
              for l, h in zip(lo, hi)]

    reps = known[known["capacity_gb"].notna() & known["grade_rank"].notna()].drop_duplicates(
        ["family", "capacity_gb", "warehouse", "grade_rank"])
    lo, hi = _consecutive_pairs(reps, ["family", "capacity_gb", "warehouse"], ["grade_rank"])
    rank = nodes["grade_rank"].to_numpy()
    edges += [(h, l, cfg.grade_step_log * float(rank[h] - rank[l]), True, "grade")
              for l, h in zip(lo, hi)]

    return NodeGraph.from_edges(nodes, edges)
