import math

import numpy as np
import pandas as pd

from reserve.config import Phase1Config
from reserve.graph import build_graph, build_nodes, parse_capacity_gb


def _products():
    rows = [("1", "64GB", "A+"), ("2", "128GB", "A+"), ("3", "64GB", "B"), ("4", "64GB", "A+")]
    return pd.DataFrame([
        {"sku": s, "manufacturer": "Apple", "model": "iPhone 12", "carrier": "OTH",
         "lock_status": "LOCKED", "capacity": c, "grade": g} for s, c, g in rows])


def _offers():
    return pd.DataFrame({"sku": ["1", "1", "2", "3", "4"],
                         "warehouse": ["W1", "W2", "W1", "W1", "W1"]})


def test_parse_capacity_normalises_terabyte_spellings():
    assert parse_capacity_gb("1TB") == 1024.0
    assert parse_capacity_gb("1000GB") == 1024.0
    assert parse_capacity_gb("128GB") == 128.0
    assert math.isnan(parse_capacity_gb("*"))


def test_build_nodes_sorted_and_enriched():
    nodes = build_nodes(_offers(), _products())
    assert list(zip(nodes["sku"], nodes["warehouse"])) == [
        ("1", "W1"), ("1", "W2"), ("2", "W1"), ("3", "W1"), ("4", "W1")]
    assert nodes.loc[0, "family"] == "Apple|iPhone 12|OTH|LOCKED"
    assert nodes.loc[3, "grade_rank"] == 6


def test_build_graph_edges():
    cfg = Phase1Config()
    g = build_graph(build_nodes(_offers(), _products()), cfg)
    edges = {(k, int(s), int(d)): (o, bool(r))
             for k, s, d, o, r in zip(g.kind, g.src, g.dst, g.offset, g.ordered)}
    assert set(edges) == {("warehouse", 0, 1), ("variant", 0, 4),
                          ("capacity", 2, 0), ("grade", 0, 3)}
    assert edges[("warehouse", 0, 1)] == (0.0, False)
    assert np.isclose(edges[("capacity", 2, 0)][0], math.log(1.10))
    assert np.isclose(edges[("grade", 0, 3)][0], 4 * math.log(1.08))
    assert edges[("grade", 0, 3)][1] is True


def test_empty_graph():
    nodes = pd.DataFrame(columns=["sku", "warehouse", "manufacturer", "model",
                                 "carrier", "lock_status", "capacity", "grade",
                                 "capacity_gb", "grade_rank", "family"])
    cfg = Phase1Config()
    g = build_graph(nodes, cfg)
    assert g.n_nodes == 0
    assert len(g.src) == 0
    assert len(g.dst) == 0
    assert len(g.offset) == 0
    assert len(g.ordered) == 0
    assert len(g.weight) == 0
    assert len(g.kind) == 0


def test_parse_capacity_edge_cases():
    assert parse_capacity_gb("2TB") == 2048.0
    assert parse_capacity_gb("1024GB") == 1024.0
    assert parse_capacity_gb("512GB") == 512.0
    assert math.isnan(parse_capacity_gb(""))
    assert math.isnan(parse_capacity_gb(None))
    assert math.isnan(parse_capacity_gb("ABC"))
    assert math.isnan(parse_capacity_gb("128MB"))


def test_unranked_grade_and_unknown_model():
    products = pd.DataFrame([
        {"sku": "1", "manufacturer": "Apple", "model": "", "carrier": "OTH",
         "lock_status": "LOCKED", "capacity": "64GB", "grade": "FTP"},
        {"sku": "2", "manufacturer": "Apple", "model": "", "carrier": "OTH",
         "lock_status": "LOCKED", "capacity": "64GB", "grade": "FTP"},
    ])
    offers = pd.DataFrame({"sku": ["1", "2"], "warehouse": ["W1", "W1"]})
    nodes = build_nodes(offers, products)
    assert math.isnan(nodes.loc[0, "grade_rank"])
    assert math.isnan(nodes.loc[1, "grade_rank"])
    cfg = Phase1Config()
    g = build_graph(nodes, cfg)
    # model is empty, so no variant, capacity, or grade edges
    assert len(g.src) == 0

