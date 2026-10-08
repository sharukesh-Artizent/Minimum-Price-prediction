# ADMM Reserve Price POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python POC that predicts a daily reserve (minimum) price per SKU × Warehouse in two phases. Phase 1 is a baseline floor learned from Assurant's decisions and pooled with network-lasso ADMM. Phase 2 is an ROI-optimal reserve solved with ADMM under warehouse clearance and sibling-ordering constraints. Both are evaluated on a time-based hold-out.

**Architecture:** A package `reserve/` with one file per job: loaders, Pareto weights, SKU graph, offer observations, a generic ADMM core (per-node bisection plus closed-form edge proximal step), the Phase-1 floor estimator, evaluation, Phase-2 demand and market inputs, the Phase-2 grid-ADMM solver, a replay backtest, and pipeline/CLI entry points. All prices are in log space. Data is read from the shared CSV extracts and is never copied into the repo.

**Tech Stack:** Python 3.13, numpy, pandas, scipy (`scipy.special.ndtr` only), pytest. No sklearn, no solver libraries — the ADMM is hand-written.

**Spec:** [docs/03-admm-design.md](../../03-admm-design.md). Background: [docs/01-project-context.md](../../01-project-context.md), [docs/02-data-findings.md](../../02-data-findings.md).

## Global Constraints

- Repo root: `C:\Artizent\Offer Clearance\POC\ADMM`. Run every command from there in Git Bash.
- Python: use the repo venv only — `.venv/Scripts/python`. Never install into the global interpreter.
- Dependencies: `numpy>=2.0`, `pandas>=2.2`, `scipy>=1.13`, `pytest>=8.0`. Nothing else.
- Data directory: `../../Data/Data Files` relative to the repo root, overridable with the `RESERVE_DATA_DIR` env var. Read-only. Never commit data or `outputs/`.
- Prices are modelled in natural-log space. Node = (sku, warehouse), with node ids `0..N-1` in `(sku, warehouse)` sort order.
- Validation is time-based only (train window before `train_end`, test window after). Never use random k-fold.
- Default windows: `train_end=2026-03-26`, `train_days=56`, `test_days=14`. Tuning uses `train_end=2026-03-12`.
- Placeholder business factors (to be replaced by Assurant values): capacity `log(1.10)` per doubling, grade `log(1.08)` per rank, holding `0.001` per day, default depreciation `0.002` per day.
- Commit messages follow `feat:` / `test:` / `docs:` prefixes and end with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## File Structure

| Path | Responsibility |
|---|---|
| `requirements.txt`, `pyproject.toml` | dependencies; pytest config (`pythonpath = ["."]`) |
| `reserve/config.py` | paths, grade ranks, `Phase1Config`, `Phase2Config` |
| `reserve/loaders.py` | `load_offers`, `load_products`, `load_sales` → clean DataFrames |
| `reserve/pareto.py` | A/B/C tiers and flexible weight `n/(n+k)` |
| `reserve/graph.py` | `NodeGraph`, `build_nodes`, `build_graph` (warehouse / variant / capacity / grade edges) |
| `reserve/observations.py` | `Observations` arrays, `attach_nodes`, `window`, `observations`, `node_priors` |
| `reserve/admm_core.py` | `edge_prox`, `solve_network` (network-lasso ADMM) |
| `reserve/phase1.py` | logistic data term and `estimate_floor` |
| `reserve/evaluate.py` | decision metrics, interval coverage, metrics by tier |
| `reserve/pipeline.py` | `prepare` context, `run_phase1`, `run_phase2` |
| `reserve/cli_phase1.py`, `reserve/cli_tune_phase1.py`, `reserve/cli_phase2.py` | command-line entry points |
| `reserve/demand.py` | `fit_demand` → μ, σ, rate, qty, value |
| `reserve/market.py` | `daily_depreciation`, `proxy_inventory` |
| `reserve/phase2.py` | `Economics`, `expected_outcomes`, `solve_phase2`, `frontier` |
| `reserve/replay.py` | replay backtest |
| `tests/` | one test file per module; `conftest.py` writes tiny CSV fixtures |
| `docs/04-results.md` | written up from real runs in Tasks 8, 9 and 14 |

---

### Task 1: Scaffold, config and data loaders

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `reserve/__init__.py` (empty), `tests/__init__.py` (empty)
- Create: `reserve/config.py`, `reserve/loaders.py`
- Test: `tests/conftest.py`, `tests/test_loaders.py`

**Interfaces:**
- Produces: `reserve.config.DATA_DIR: Path`, `OUTPUT_DIR: Path`, `GRADE_RANK: dict[str, int]`, frozen dataclasses `Phase1Config`, `Phase2Config` (fields as in the code below).
- Produces: `load_offers(data_dir) -> DataFrame[offer_id, buyer_id, sku, warehouse, qty, bid, status, created, date, accepted, log_bid]` (only ACCEPTED/COUNTERED/NOT_ACCEPTED, bid>0, qty>0; `accepted` is 1.0/0.0; `date` is midnight Timestamp).
- Produces: `load_products(data_dir) -> DataFrame[sku, manufacturer, model, carrier, lock_status, capacity, grade]` (one row per sku).
- Produces: `load_sales(data_dir) -> DataFrame[sku, warehouse, unit_price, qty, created, date]` (price>0, qty>0).
- Produces: pytest fixture `data_dir` (tmp dir with tiny Offers / Product / Sales CSVs).

- [ ] **Step 1: Create the venv and project files**

`requirements.txt`:

```text
numpy>=2.0
pandas>=2.2
scipy>=1.13
pytest>=8.0
```

`pyproject.toml`:

```toml
[project]
name = "reserve-admm-poc"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
mkdir -p reserve tests && touch reserve/__init__.py tests/__init__.py
```

Expected: pip finishes without errors; `.venv/Scripts/python -c "import numpy, pandas, scipy"` prints nothing.

- [ ] **Step 2: Write the fixture and the failing test**

`tests/conftest.py`:

```python
import pytest

OFFERS = """OFFER_ID,BUYER_ID,SKU,WAREHOUSE,OFFER_QUANTITY,OFFER_PRICE,COUNTER_PRICE,OFFER_STATUS,ORDER_STATUS,OFFER_CREATE_DATETIME,TIMESTAMP,SUBMITTED_SHOPPING_CART_ID
1,70,100,W1,2,100,NULL,ACCEPTED,REDEEMED,10/20/2025 13:08,10/20/2025 16:32,1
2,70,100,W1,1,80,85,COUNTERED,REDEEMED,10/20/2025 14:08,10/20/2025 16:32,1
3,71,100,W1,1,60,NULL,NOT_ACCEPTED,NULL,10/21/2025 09:00,10/21/2025 10:00,NULL
4,71,200,W1,1,0,NULL,ACCEPTED,NULL,10/21/2025 09:00,10/21/2025 10:00,NULL
5,71,200,W1,1,50,NULL,PENDING_ACCEPTANCE,NULL,10/21/2025 09:00,10/21/2025 10:00,NULL
6,72,100,W1,1,90,NULL,ACCEPTED,REDEEMED,10/23/2025 10:00,10/23/2025 11:00,2
"""

PRODUCTS = """SKU,WAREHOUSE,MANUFACTURER,MODEL,CARRIER,LOCK_STATUS,CAPACITY,GRADE
100,W1,Apple,iPhone 12,OTH,LOCKED,64GB,A+
100,W2,Apple,iPhone 12,OTH,LOCKED,64GB,A+
200,W1,Apple,iPhone 12,OTH,LOCKED,128GB,A+
"""

SALES = """SALE_ID,BUYER_ID,WAREHOUSE_CODE,DATE,PRODUCT_ID,UNIT_PRICE,SALES_QTY
1,70,W1,10/22/2025 11:26,100,95,2
2,70,W1,10/23/2025 11:26,100,0,1
3,70,W1,10/24/2025 11:26,200,150,1
"""


@pytest.fixture
def data_dir(tmp_path):
    files = {
        "Offers/Offers.csv": OFFERS,
        "Product/Product Information.csv": PRODUCTS,
        "Sales/Historical Sales/Sales Order&Line items.csv": SALES,
    }
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path
```

`tests/test_loaders.py`:

```python
import numpy as np

from reserve.loaders import load_offers, load_products, load_sales


def test_load_offers_keeps_decided_positive_offers(data_dir):
    offers = load_offers(data_dir)
    assert sorted(offers["offer_id"]) == ["1", "2", "3", "6"]
    assert set(offers["status"]) == {"ACCEPTED", "COUNTERED", "NOT_ACCEPTED"}


def test_load_offers_derives_label_date_and_log_bid(data_dir):
    offers = load_offers(data_dir).set_index("offer_id")
    assert offers.loc["1", "accepted"] == 1.0
    assert offers.loc["2", "accepted"] == 0.0
    assert offers.loc["1", "date"].strftime("%Y-%m-%d") == "2025-10-20"
    assert np.isclose(offers.loc["1", "log_bid"], np.log(100))


def test_load_products_one_row_per_sku(data_dir):
    products = load_products(data_dir)
    assert products["sku"].is_unique
    assert list(products.columns) == [
        "sku", "manufacturer", "model", "carrier", "lock_status", "capacity", "grade"]


def test_load_sales_drops_zero_price(data_dir):
    sales = load_sales(data_dir)
    assert len(sales) == 2
    assert set(sales["sku"]) == {"100", "200"}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_loaders.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reserve.loaders'`

- [ ] **Step 4: Write the implementation**

`reserve/config.py`:

```python
"""Paths and tunable parameters for the ADMM reserve-price POC."""
import math
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("RESERVE_DATA_DIR", REPO_ROOT.parents[1] / "Data" / "Data Files"))
OUTPUT_DIR = REPO_ROOT / "outputs"

# Higher rank = more valuable. Grades not listed here (R, L, S, FTP) get no grade
# edges until Assurant confirms where they sit in the scale.
GRADE_RANK = {
    "AA+": 11, "A+": 10, "A": 9, "A-": 8, "B+": 7, "B": 6,
    "B-": 5, "C+": 4, "C": 3, "C-": 2, "D": 1,
}


@dataclass(frozen=True)
class Phase1Config:
    lam: float = 1.0                 # strength of sibling fusion
    rho: float = 1.0                 # ADMM penalty
    k_pareto: float = 10.0           # node weight w_i = n_i / (n_i + k)
    temperature: float = 0.03        # logistic softness in log-price units (~3%)
    prior_weight: float = 0.01       # pull toward family median when data and edges are silent
    recency_tau_days: float = 14.0   # offer weight = exp(-age_days / tau)
    max_iter: int = 300
    eps: float = 1e-4
    bisect_iters: int = 40
    capacity_step_log: float = math.log(1.10)  # placeholder until Assurant sets storage factors
    grade_step_log: float = math.log(1.08)     # placeholder until Assurant sets grade factors


@dataclass(frozen=True)
class Phase2Config:
    grid_points: int = 41
    kappa_min: float = 0.02          # trust region (log) for tail nodes
    kappa_max: float = 0.15          # trust region (log) for head nodes
    rho: float = 1.0
    max_iter: int = 200
    eps: float = 1e-3
    dual_step: float = 5.0           # $/unit added to warehouse dual per 100% shortfall
    holding_frac: float = 0.001      # holding cost per unit-day as fraction of value
    default_delta: float = 0.002     # daily depreciation when a family has too few sales
    rate_prior_days: float = 7.0     # pseudo-days shrinking node rates to family rate
    min_sigma: float = 0.02
    max_days_to_clear: float = 365.0
```

`reserve/loaders.py`:

```python
"""Read the Assurant CSV extracts into clean DataFrames."""
from pathlib import Path

import numpy as np
import pandas as pd

DATE_FMT = "%m/%d/%Y %H:%M"
DECISION_STATUSES = ("ACCEPTED", "COUNTERED", "NOT_ACCEPTED")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, na_values=["NULL", ""], keep_default_na=False)


def load_offers(data_dir: Path) -> pd.DataFrame:
    df = _read(Path(data_dir) / "Offers" / "Offers.csv")
    out = pd.DataFrame({
        "offer_id": df["OFFER_ID"],
        "buyer_id": df["BUYER_ID"],
        "sku": df["SKU"].str.strip(),
        "warehouse": df["WAREHOUSE"].str.strip(),
        "qty": pd.to_numeric(df["OFFER_QUANTITY"], errors="coerce"),
        "bid": pd.to_numeric(df["OFFER_PRICE"], errors="coerce"),
        "status": df["OFFER_STATUS"],
        "created": pd.to_datetime(df["OFFER_CREATE_DATETIME"], format=DATE_FMT, errors="coerce"),
    })
    keep = (
        out["status"].isin(DECISION_STATUSES)
        & (out["bid"] > 0)
        & (out["qty"] > 0)
        & out["created"].notna()
    )
    out = out[keep].copy()
    out["date"] = out["created"].dt.normalize()
    out["accepted"] = (out["status"] == "ACCEPTED").astype(float)
    out["log_bid"] = np.log(out["bid"])
    return out.reset_index(drop=True)


def load_products(data_dir: Path) -> pd.DataFrame:
    df = _read(Path(data_dir) / "Product" / "Product Information.csv")
    out = pd.DataFrame({
        "sku": df["SKU"].str.strip(),
        "manufacturer": df["MANUFACTURER"].fillna(""),
        "model": df["MODEL"].fillna(""),
        "carrier": df["CARRIER"].fillna(""),
        "lock_status": df["LOCK_STATUS"].fillna(""),
        "capacity": df["CAPACITY"].fillna(""),
        "grade": df["GRADE"].fillna("").str.strip(),
    })
    return out.drop_duplicates("sku").reset_index(drop=True)


def load_sales(data_dir: Path) -> pd.DataFrame:
    df = _read(Path(data_dir) / "Sales" / "Historical Sales" / "Sales Order&Line items.csv")
    out = pd.DataFrame({
        "sku": df["PRODUCT_ID"].str.strip(),
        "warehouse": df["WAREHOUSE_CODE"].str.strip(),
        "unit_price": pd.to_numeric(df["UNIT_PRICE"], errors="coerce"),
        "qty": pd.to_numeric(df["SALES_QTY"], errors="coerce"),
        "created": pd.to_datetime(df["DATE"], format=DATE_FMT, errors="coerce"),
    })
    keep = (out["unit_price"] > 0) & (out["qty"] > 0) & out["created"].notna()
    out = out[keep].copy()
    out["date"] = out["created"].dt.normalize()
    return out.reset_index(drop=True)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_loaders.py -v`
Expected: 4 PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pyproject.toml reserve tests
git commit -m "feat: scaffold reserve package with config and data loaders"
```

### Task 2: Pareto tiers and flexible weight

**Files:**
- Create: `reserve/pareto.py`
- Test: `tests/test_pareto.py`

**Interfaces:**
- Produces: `pareto_tiers(counts: pd.Series, a_share=0.80, b_share=0.95) -> DataFrame[n, cum_share, tier]` (an item is in A if the cumulative share *before* it is < 80%).
- Produces: `node_tiers(counts: np.ndarray, ...) -> np.ndarray[object]` with values `A`/`B`/`C`/`unseen` (count 0).
- Produces: `pareto_weight(n, k) -> np.ndarray` = `n / (n + k)`.

- [ ] **Step 1: Write the failing test** — `tests/test_pareto.py`

```python
import numpy as np
import pandas as pd

from reserve.pareto import node_tiers, pareto_tiers, pareto_weight


def test_pareto_tiers_include_item_that_crosses_threshold():
    counts = pd.Series([5, 50, 30, 5, 10], index=["e", "a", "b", "d", "c"])
    tiers = pareto_tiers(counts)
    assert tiers.loc[["a", "b"], "tier"].tolist() == ["A", "A"]
    assert tiers.loc[["c", "e"], "tier"].tolist() == ["B", "B"]
    assert tiers.loc["d", "tier"] == "C"


def test_node_tiers_marks_zero_count_nodes_unseen():
    tiers = node_tiers(np.array([90, 0, 10]))
    assert tiers.tolist() == ["A", "unseen", "B"]


def test_pareto_weight_saturates():
    w = pareto_weight([0, 10, 1000], k=10)
    assert np.allclose(w, [0.0, 0.5, 1000 / 1010])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_pareto.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reserve.pareto'`

- [ ] **Step 3: Write the implementation**

`reserve/pareto.py`:

```python
"""Pareto (A/B/C) tiers by offer volume and the flexible per-node weight."""
import numpy as np
import pandas as pd


def pareto_tiers(counts: pd.Series, a_share: float = 0.80, b_share: float = 0.95) -> pd.DataFrame:
    s = counts.sort_values(ascending=False, kind="mergesort")
    cum = s.cumsum() / s.sum()
    before = cum.shift(fill_value=0.0)
    tier = np.where(before < a_share, "A", np.where(before < b_share, "B", "C"))
    return pd.DataFrame({"n": s, "cum_share": cum, "tier": tier}, index=s.index)


def node_tiers(counts: np.ndarray, a_share: float = 0.80, b_share: float = 0.95) -> np.ndarray:
    tiers = np.full(len(counts), "unseen", dtype=object)
    seen = np.flatnonzero(counts > 0)
    if seen.size:
        t = pareto_tiers(pd.Series(counts[seen], index=seen), a_share, b_share)
        tiers[t.index.to_numpy()] = t["tier"].to_numpy()
    return tiers


def pareto_weight(n, k: float) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    return n / (n + k)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_pareto.py -v`
Expected: all PASS. Then run the whole suite: `.venv/Scripts/python -m pytest -q` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add reserve/pareto.py tests/test_pareto.py
git commit -m "feat: pareto tiers and flexible node weight"
```

### Task 3: SKU graph (nodes and sibling edges)

**Files:**
- Create: `reserve/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `GRADE_RANK`, `Phase1Config.capacity_step_log`, `Phase1Config.grade_step_log` (Task 1).
- Produces: dataclass `NodeGraph(nodes, src, dst, offset, ordered, weight, kind)` with property `n_nodes` and classmethod `NodeGraph.from_edges(nodes, edges: list[(src, dst, offset, ordered, kind)])`.
- Produces: `parse_capacity_gb(text) -> float` (1TB/1000GB/1024GB → 1024.0; unknown → nan).
- Produces: `build_nodes(offers, products) -> DataFrame` sorted by (sku, warehouse), index 0..N-1, adding `capacity_gb`, `grade_rank` (NaN for unranked grades), `family`.
- Produces: `build_graph(nodes, cfg: Phase1Config) -> NodeGraph`. Ordered edges always have `src` = the more valuable node.

- [ ] **Step 1: Write the failing test** — `tests/test_graph.py`

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reserve.graph'`

- [ ] **Step 3: Write the implementation**

`reserve/graph.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_graph.py -v`
Expected: all PASS. Then run the whole suite: `.venv/Scripts/python -m pytest -q` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add reserve/graph.py tests/test_graph.py
git commit -m "feat: SKU x warehouse graph with sibling edges"
```

### Task 4: Offer observations and family priors

**Files:**
- Create: `reserve/observations.py`
- Test: `tests/test_observations.py`

**Interfaces:**
- Consumes: DataFrames shaped like `load_offers` / `load_sales` output (columns `sku`, `warehouse`, `date`, plus `log_bid`, `accepted`, `qty` for offers) and nodes with a `family` column.
- Produces: dataclass `Observations(node, log_bid, accepted, weight, qty, date)` (numpy arrays, `len()` supported).
- Produces: `attach_nodes(frame, nodes) -> np.ndarray` (node id, or -1); `window(frame, start, end)` (half-open `[start, end)` on `date`).
- Produces: `observations(offers, nodes, as_of, tau_days) -> Observations` with weight `exp(-(as_of - date).days / tau)`; pass `tau_days=float('inf')` for weight 1.
- Produces: `node_priors(obs, nodes) -> np.ndarray` (family median log bid of accepted offers, global median fallback).

- [ ] **Step 1: Write the failing test** — `tests/test_observations.py`

```python
import numpy as np
import pandas as pd

from reserve.observations import attach_nodes, node_priors, observations, window


def _nodes():
    return pd.DataFrame({"sku": ["1", "2", "3"], "warehouse": ["W1", "W1", "W1"],
                         "family": ["F", "F", "G"]})


def _offers():
    return pd.DataFrame({
        "sku": ["1", "1", "2", "9"],
        "warehouse": ["W1", "W1", "W1", "W1"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-09", "2026-01-10", "2026-01-10"]),
        "log_bid": np.log([100.0, 80.0, 120.0, 50.0]),
        "accepted": [1.0, 0.0, 1.0, 1.0],
        "qty": [2.0, 1.0, 3.0, 1.0],
    })


def test_attach_nodes_marks_unknown_minus_one():
    assert attach_nodes(_offers(), _nodes()).tolist() == [0, 0, 1, -1]


def test_window_is_half_open():
    w = window(_offers(), "2026-01-09", "2026-01-10")
    assert len(w) == 1


def test_observations_drop_unknown_and_weight_by_recency():
    obs = observations(_offers(), _nodes(), as_of="2026-01-11", tau_days=10.0)
    assert obs.node.tolist() == [0, 0, 1]
    assert np.allclose(obs.weight, np.exp(-np.array([10.0, 2.0, 1.0]) / 10.0))
    assert obs.qty.tolist() == [2.0, 1.0, 3.0]


def test_node_priors_family_median_of_accepted_with_global_fallback():
    obs = observations(_offers(), _nodes(), as_of="2026-01-11", tau_days=10.0)
    prior = node_priors(obs, _nodes())
    family_f = np.median(np.log([100.0, 120.0]))
    assert np.allclose(prior, [family_f, family_f, family_f])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_observations.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reserve.observations'`

- [ ] **Step 3: Write the implementation**

`reserve/observations.py`:

```python
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
    return Observations(
        node=node[keep].astype(np.int64),
        log_bid=o["log_bid"].to_numpy(dtype=float),
        accepted=o["accepted"].to_numpy(dtype=float),
        weight=np.exp(-age / tau_days),
        qty=o["qty"].to_numpy(dtype=float),
        date=o["date"].to_numpy(),
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_observations.py -v`
Expected: all PASS. Then run the whole suite: `.venv/Scripts/python -m pytest -q` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add reserve/observations.py tests/test_observations.py
git commit -m "feat: offer observations, windows and family priors"
```

### Task 5: Network-lasso ADMM core

**Files:**
- Create: `reserve/admm_core.py`
- Test: `tests/test_admm_core.py`

**Interfaces:**
- Consumes: `NodeGraph` (Task 3).
- Produces: `LocalGrad = Callable[[np.ndarray], np.ndarray]`. It returns the per-node derivative of the data term at x.
- Produces: `edge_prox(p, q, offset, ordered, theta) -> (z_src, z_dst)` (closed form; see spec section 3).
- Produces: `solve_network(n_nodes, local_grad, graph, prior, prior_weight, lam, rho, max_iter, eps, bisect_iters, bracket, x0=None) -> ADMMResult(x, iterations, converged, primal_residual, dual_residual)`.
- Note: the x-update uses bisection, so `local_grad` must be nondecreasing in x (the data term must be convex).

- [ ] **Step 1: Write the failing test** — `tests/test_admm_core.py`

```python
import numpy as np
import pandas as pd

from reserve.admm_core import edge_prox, solve_network
from reserve.graph import NodeGraph


def _graph(n, edges):
    return NodeGraph.from_edges(pd.DataFrame({"sku": [str(i) for i in range(n)]}), edges)


def _quadratic(targets):
    targets = np.asarray(targets, dtype=float)
    return lambda x: x - targets   # gradient of 0.5 * (x - target)^2


def _solve(graph, targets, lam):
    n = graph.n_nodes
    return solve_network(n, _quadratic(targets), graph, prior=np.zeros(n), prior_weight=0.0,
                         lam=lam, rho=1.0, max_iter=2000, eps=1e-7, bisect_iters=60,
                         bracket=(-5.0, 5.0))


def test_edge_prox_zero_theta_unordered_is_identity():
    zs, zd = edge_prox(np.array([0.3]), np.array([1.2]), np.array([0.0]), np.array([False]), 0.0)
    assert np.allclose([zs[0], zd[0]], [0.3, 1.2])


def test_edge_prox_large_theta_fuses_at_offset_keeping_midpoint():
    zs, zd = edge_prox(np.array([0.0]), np.array([1.0]), np.array([0.2]), np.array([False]), 10.0)
    assert np.isclose(zs[0] - zd[0], 0.2)
    assert np.isclose(zs[0] + zd[0], 1.0)


def test_edge_prox_ordered_projects_violation_to_equality():
    zs, zd = edge_prox(np.array([0.0]), np.array([1.0]), np.array([0.1]), np.array([True]), 0.0)
    assert np.allclose([zs[0], zd[0]], [0.5, 0.5])


def test_edge_prox_ordered_leaves_satisfied_pair_alone():
    zs, zd = edge_prox(np.array([2.0]), np.array([1.0]), np.array([0.1]), np.array([True]), 0.0)
    assert np.allclose([zs[0], zd[0]], [2.0, 1.0])


def test_solve_without_edges_returns_local_minimisers():
    res = _solve(_graph(2, []), [0.0, 2.0], lam=1.0)
    assert res.converged
    assert np.allclose(res.x, [0.0, 2.0], atol=1e-4)


def test_strong_fusion_pulls_nodes_together():
    res = _solve(_graph(2, [(0, 1, 0.0, False, "t")]), [0.0, 2.0], lam=50.0)
    assert np.allclose(res.x, [1.0, 1.0], atol=1e-3)


def test_ordered_edge_enforces_ranking():
    res = _solve(_graph(2, [(0, 1, 0.0, True, "t")]), [0.0, 2.0], lam=0.0)
    assert np.allclose(res.x, [1.0, 1.0], atol=1e-3)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_admm_core.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reserve.admm_core'`

- [ ] **Step 3: Write the implementation**

`reserve/admm_core.py`:

```python
"""Network-lasso ADMM over the SKU graph.

minimize  sum_i data_i(x_i) + (gamma/2)(x_i - prior_i)^2
        + lam * sum_e a_e |x_src - x_dst - offset_e|
s.t.      x_src >= x_dst on ordered edges

Each edge keeps its own copy of both endpoints (z_src, z_dst) with scaled duals.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from reserve.graph import NodeGraph

LocalGrad = Callable[[np.ndarray], np.ndarray]


@dataclass
class ADMMResult:
    x: np.ndarray
    iterations: int
    converged: bool
    primal_residual: list = field(default_factory=list)
    dual_residual: list = field(default_factory=list)


def edge_prox(p, q, offset, ordered, theta):
    """Closed-form prox of theta*|t| (+ t >= -offset if ordered), t = z_src - z_dst - offset.

    p, q are the points (x + u) each endpoint copy is pulled toward.
    """
    delta = p - q - offset
    t = np.sign(delta) * np.maximum(np.abs(delta) - theta, 0.0)
    t = np.where(ordered, np.maximum(t, -offset), t)
    half = (delta - t) / 2.0
    return p - half, q + half


def _x_update(local_grad, prior, prior_weight, rho, deg, sum_v, lo, hi, iters):
    """Per-node 1-D convex minimisation by vectorised bisection on the derivative."""
    a, b = lo.copy(), hi.copy()
    for _ in range(iters):
        m = 0.5 * (a + b)
        g = local_grad(m) + prior_weight * (m - prior) + rho * (deg * m - sum_v)
        up = g > 0
        b = np.where(up, m, b)
        a = np.where(up, a, m)
    return 0.5 * (a + b)


def solve_network(n_nodes: int, local_grad: LocalGrad, graph: NodeGraph, prior: np.ndarray,
                  prior_weight: float, lam: float, rho: float, max_iter: int, eps: float,
                  bisect_iters: int, bracket: tuple[float, float], x0=None) -> ADMMResult:
    src, dst = graph.src, graph.dst
    n_edges = len(src)
    x = prior.copy() if x0 is None else np.asarray(x0, dtype=float).copy()
    z_s, z_d = x[src].copy(), x[dst].copy()
    u_s, u_d = np.zeros(n_edges), np.zeros(n_edges)
    deg = (np.bincount(src, minlength=n_nodes) + np.bincount(dst, minlength=n_nodes)).astype(float)
    theta = 2.0 * lam * graph.weight / rho
    lo, hi = np.full(n_nodes, bracket[0]), np.full(n_nodes, bracket[1])
    tol = eps * np.sqrt(max(2 * n_edges, 1))
    result = ADMMResult(x=x, iterations=0, converged=False)

    for it in range(1, max_iter + 1):
        sum_v = (np.bincount(src, weights=z_s - u_s, minlength=n_nodes)
                 + np.bincount(dst, weights=z_d - u_d, minlength=n_nodes))
        x = _x_update(local_grad, prior, prior_weight, rho, deg, sum_v, lo, hi, bisect_iters)
        zs_old, zd_old = z_s, z_d
        z_s, z_d = edge_prox(x[src] + u_s, x[dst] + u_d, graph.offset, graph.ordered, theta)
        u_s = u_s + x[src] - z_s
        u_d = u_d + x[dst] - z_d
        r = float(np.sqrt(np.sum((x[src] - z_s) ** 2 + (x[dst] - z_d) ** 2)))
        s = float(rho * np.sqrt(np.sum((z_s - zs_old) ** 2 + (z_d - zd_old) ** 2)))
        result.primal_residual.append(r)
        result.dual_residual.append(s)
        result.iterations = it
        if r < tol and s < tol:
            result.converged = True
            break

    result.x = x
    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_admm_core.py -v`
Expected: all PASS. Then run the whole suite: `.venv/Scripts/python -m pytest -q` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add reserve/admm_core.py tests/test_admm_core.py
git commit -m "feat: network-lasso ADMM core with ordered edges"
```

### Task 6: Phase-1 floor estimator

**Files:**
- Create: `reserve/phase1.py`
- Test: `tests/test_phase1.py`

**Interfaces:**
- Consumes: `Observations` (Task 4), `solve_network` (Task 5), `pareto_weight`, `node_tiers` (Task 2), `Phase1Config`.
- Produces: `sigmoid(z)`, `logistic_local_grad(obs, node_weight, n_nodes, temperature) -> LocalGrad`, `price_bracket(obs, prior) -> (lo, hi)`.
- Produces: `estimate_floor(obs, graph, prior, cfg, lam=None, x0=None) -> Phase1Output(floor_log, node_weight, tier, result)`. `lam=None` uses `cfg.lam`.

- [ ] **Step 1: Write the failing test** — `tests/test_phase1.py`

```python
import numpy as np
import pandas as pd

from reserve.config import Phase1Config
from reserve.graph import NodeGraph
from reserve.observations import Observations
from reserve.phase1 import estimate_floor, logistic_local_grad


def _obs(node, bids, accepted):
    n = len(node)
    return Observations(node=np.asarray(node, dtype=np.int64), log_bid=np.log(bids),
                        accepted=np.asarray(accepted, dtype=float), weight=np.ones(n),
                        qty=np.ones(n), date=np.full(n, np.datetime64("2026-01-01")))


def _graph(n, edges):
    return NodeGraph.from_edges(pd.DataFrame({"sku": [str(i) for i in range(n)]}), edges)


# node 0: accepted at 100, countered/rejected at 80 -> floor somewhere in (80, 100)
DATA = _obs([0, 0, 0, 0, 0, 0], [100.0, 100.0, 100.0, 80.0, 80.0, 80.0], [1, 1, 1, 0, 0, 0])


def test_logistic_grad_pushes_floor_down_when_accepted_bid_below_it():
    obs = _obs([0], [100.0], [1])
    grad = logistic_local_grad(obs, np.array([1.0]), 1, temperature=0.03)
    assert grad(np.array([np.log(120.0)]))[0] > 0


def test_single_node_floor_between_rejected_and_accepted():
    prior = np.array([np.log(50.0)])
    out = estimate_floor(DATA, _graph(1, []), prior, Phase1Config())
    assert np.log(80.0) < out.floor_log[0] < np.log(100.0)
    assert out.tier.tolist() == ["A"]


def test_fusion_lends_floor_to_node_without_data():
    prior = np.array([np.log(50.0), np.log(50.0)])
    g = _graph(2, [(0, 1, 0.0, False, "warehouse")])
    out = estimate_floor(DATA, g, prior, Phase1Config(lam=5.0, max_iter=2000, eps=1e-6))
    assert abs(out.floor_log[1] - out.floor_log[0]) < 0.05
    assert out.tier.tolist() == ["A", "unseen"]


def test_without_fusion_dataless_node_stays_at_prior():
    prior = np.array([np.log(50.0), np.log(50.0)])
    g = _graph(2, [(0, 1, 0.0, False, "warehouse")])
    out = estimate_floor(DATA, g, prior, Phase1Config(max_iter=2000, eps=1e-6), lam=0.0)
    assert abs(out.floor_log[1] - np.log(50.0)) < 1e-3
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_phase1.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reserve.phase1'`

- [ ] **Step 3: Write the implementation**

`reserve/phase1.py`:

```python
"""Phase 1: Assurant's implied floor per node, pooled across siblings with ADMM.

Each offer is a logistic observation: P(accept | bid) = sigmoid((log_bid - x_i) / T).
A node's data term is w_i * (recency-weighted mean log-loss), w_i = n_i / (n_i + k).
"""
from dataclasses import dataclass

import numpy as np

from reserve.admm_core import ADMMResult, LocalGrad, solve_network
from reserve.config import Phase1Config
from reserve.graph import NodeGraph
from reserve.observations import Observations
from reserve.pareto import node_tiers, pareto_weight


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


@dataclass
class Phase1Output:
    floor_log: np.ndarray
    node_weight: np.ndarray
    tier: np.ndarray
    result: ADMMResult


def logistic_local_grad(obs: Observations, node_weight: np.ndarray, n_nodes: int,
                        temperature: float) -> LocalGrad:
    total = np.bincount(obs.node, weights=obs.weight, minlength=n_nodes)
    coef = np.divide(node_weight, total, out=np.zeros(n_nodes), where=total > 0)

    def grad(x: np.ndarray) -> np.ndarray:
        z = (obs.log_bid - x[obs.node]) / temperature
        g = obs.weight * (obs.accepted - sigmoid(z)) / temperature
        return coef * np.bincount(obs.node, weights=g, minlength=n_nodes)

    return grad


def price_bracket(obs: Observations, prior: np.ndarray) -> tuple[float, float]:
    lows = [prior.min()] + ([obs.log_bid.min()] if len(obs) else [])
    highs = [prior.max()] + ([obs.log_bid.max()] if len(obs) else [])
    return float(min(lows)) - 1.0, float(max(highs)) + 1.0


def estimate_floor(obs: Observations, graph: NodeGraph, prior: np.ndarray, cfg: Phase1Config,
                   lam: float | None = None, x0=None) -> Phase1Output:
    n = graph.n_nodes
    counts = np.bincount(obs.node, minlength=n)
    weight = pareto_weight(counts, cfg.k_pareto)
    grad = logistic_local_grad(obs, weight, n, cfg.temperature)
    result = solve_network(
        n, grad, graph, prior, cfg.prior_weight, cfg.lam if lam is None else lam,
        cfg.rho, cfg.max_iter, cfg.eps, cfg.bisect_iters, price_bracket(obs, prior), x0)
    return Phase1Output(result.x, weight, node_tiers(counts), result)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_phase1.py -v`
Expected: all PASS. Then run the whole suite: `.venv/Scripts/python -m pytest -q` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add reserve/phase1.py tests/test_phase1.py
git commit -m "feat: phase-1 floor estimator (logistic network lasso)"
```

### Task 7: Phase-1 evaluation metrics

**Files:**
- Create: `reserve/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `Observations` (Task 4), `sigmoid` (Task 6).
- Produces: `decision_metrics(floor_log, obs, temperature) -> dict(n, accuracy, balanced_accuracy, log_loss)` (`{'n': 0}` when empty).
- Produces: `interval_coverage(floor_log, obs) -> dict(n_intervals, coverage, mean_log_gap)` (`{'n_intervals': 0}` when none).
- Produces: `subset(obs, mask) -> Observations`, `metrics_by_tier(floor_log, obs, tiers, temperature) -> dict[tier, metrics]`.

- [ ] **Step 1: Write the failing test** — `tests/test_evaluate.py`

```python
import numpy as np

from reserve.evaluate import decision_metrics, interval_coverage, metrics_by_tier
from reserve.observations import Observations


def _obs():
    # node 0 day 1: rejected 80, accepted 100 -> interval [log80, log100]
    # node 1 day 1: rejected 50, accepted 60  -> interval [log50, log60]
    return Observations(
        node=np.array([0, 0, 1, 1]),
        log_bid=np.log([80.0, 100.0, 50.0, 60.0]),
        accepted=np.array([0.0, 1.0, 0.0, 1.0]),
        weight=np.ones(4), qty=np.ones(4),
        date=np.array(["2026-01-01"] * 4, dtype="datetime64[ns]"))


def test_decision_metrics_perfect_floor():
    floor = np.log([90.0, 55.0])
    m = decision_metrics(floor, _obs(), temperature=0.03)
    assert m["n"] == 4
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0


def test_decision_metrics_empty():
    obs = _obs()
    empty = Observations(*(a[:0] for a in (obs.node, obs.log_bid, obs.accepted,
                                           obs.weight, obs.qty, obs.date)))
    assert decision_metrics(np.zeros(2), empty, 0.03) == {"n": 0}


def test_interval_coverage_counts_inside_and_gap():
    floor = np.log([90.0, 70.0])          # node 1 floor above its interval
    m = interval_coverage(floor, _obs())
    assert m["n_intervals"] == 2
    assert m["coverage"] == 0.5
    assert np.isclose(m["mean_log_gap"], (np.log(70.0) - np.log(60.0)) / 2)


def test_metrics_by_tier_splits_offers():
    m = metrics_by_tier(np.log([90.0, 55.0]), _obs(), np.array(["A", "B"], dtype=object), 0.03)
    assert set(m) == {"A", "B"}
    assert m["A"]["n"] == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reserve.evaluate'`

- [ ] **Step 3: Write the implementation**

`reserve/evaluate.py`:

```python
"""Phase-1 hold-out metrics: decision agreement and floor-interval coverage."""
import numpy as np
import pandas as pd

from reserve.observations import Observations
from reserve.phase1 import sigmoid


def decision_metrics(floor_log: np.ndarray, obs: Observations, temperature: float) -> dict:
    if len(obs) == 0:
        return {"n": 0}
    margin = obs.log_bid - floor_log[obs.node]
    pred = margin >= 0
    y = obs.accepted == 1
    tpr = float(np.mean(pred[y])) if y.any() else float("nan")
    tnr = float(np.mean(~pred[~y])) if (~y).any() else float("nan")
    p = np.clip(sigmoid(margin / temperature), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(obs)),
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float(np.nanmean([tpr, tnr])),
        "log_loss": float(-np.mean(np.where(y, np.log(p), np.log(1 - p)))),
    }


def interval_coverage(floor_log: np.ndarray, obs: Observations) -> dict:
    """Share of node-days with both an accept and a non-accept where L <= floor <= U."""
    df = pd.DataFrame({"node": obs.node, "date": obs.date, "log_bid": obs.log_bid,
                       "accepted": obs.accepted})
    upper = df[df["accepted"] == 1].groupby(["node", "date"])["log_bid"].min().rename("U")
    lower = df[df["accepted"] == 0].groupby(["node", "date"])["log_bid"].max().rename("L")
    both = pd.concat([lower, upper], axis=1).dropna()
    if both.empty:
        return {"n_intervals": 0}
    x = floor_log[both.index.get_level_values("node").to_numpy()]
    lo, hi = both["L"].to_numpy(), both["U"].to_numpy()
    gap = np.maximum(lo - x, 0.0) + np.maximum(x - hi, 0.0)
    return {"n_intervals": int(len(both)),
            "coverage": float(np.mean(gap == 0.0)),
            "mean_log_gap": float(np.mean(gap))}


def subset(obs: Observations, mask: np.ndarray) -> Observations:
    return Observations(obs.node[mask], obs.log_bid[mask], obs.accepted[mask],
                        obs.weight[mask], obs.qty[mask], obs.date[mask])


def metrics_by_tier(floor_log: np.ndarray, obs: Observations, tiers: np.ndarray,
                    temperature: float) -> dict:
    node_tier = tiers[obs.node]
    return {str(t): decision_metrics(floor_log, subset(obs, node_tier == t), temperature)
            for t in sorted(set(node_tier))}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_evaluate.py -v`
Expected: all PASS. Then run the whole suite: `.venv/Scripts/python -m pytest -q` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add reserve/evaluate.py tests/test_evaluate.py
git commit -m "feat: phase-1 decision and interval metrics"
```

### Task 8: Phase-1 pipeline and CLI (real-data checkpoint)

**Files:**
- Create: `reserve/pipeline.py`, `reserve/cli_phase1.py`
- Test: `tests/test_cli_phase1.py`

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces: dataclass `Context(offers, sales, nodes, graph, train_obs, test_obs, prior, train_start, train_end, test_end)`.
- Produces: `prepare(data_dir, train_end: str, train_days: int, test_days: int, cfg: Phase1Config) -> Context`.
- Produces: `run_phase1(ctx, cfg) -> (metrics: dict, floors: DataFrame, pooled: Phase1Output)`. Metrics keys: `graph`, `admm`, `family_prior`, `admm_no_fusion`, `admm_fused`; each policy has `overall`, `by_tier`, `intervals`.
- Produces: `reserve.cli_phase1.main(argv=None) -> dict`, which writes `phase1_floors.csv` and `phase1_metrics.json` to `--out-dir`.

- [ ] **Step 1: Write the failing test** — `tests/test_cli_phase1.py`

```python
import json

from reserve.cli_phase1 import main


def test_phase1_cli_writes_outputs(data_dir, tmp_path):
    out = tmp_path / "out"
    metrics = main(["--data-dir", str(data_dir), "--out-dir", str(out),
                    "--train-end", "2025-10-23", "--train-days", "5", "--test-days", "3"])
    assert (out / "phase1_floors.csv").exists()
    saved = json.loads((out / "phase1_metrics.json").read_text())
    assert saved["graph"]["nodes"] == 1
    assert metrics["admm_fused"]["overall"]["n"] == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_cli_phase1.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reserve.cli_phase1'`

- [ ] **Step 3: Write the implementation**

`reserve/pipeline.py`:

```python
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
```

`reserve/cli_phase1.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_cli_phase1.py -v`
Expected: all PASS. Then run the whole suite: `.venv/Scripts/python -m pytest -q` — all PASS.

- [ ] **Step 5: Run Phase 1 on the real data**

Run: `.venv/Scripts/python -m reserve.cli_phase1`
Expected: finishes in about 1.5–3 minutes and prints JSON. Sanity checks (the prototype got these values; yours should be close):
- `graph.nodes` ≈ 55,070; `graph.edges` ≈ 68,357
- `admm_fused.overall.accuracy` ≈ 0.76, higher than `admm_no_fusion` (≈ 0.73), which is higher than `family_prior` (≈ 0.60)
- the fused improvement is largest on tiers `B`, `C` and `unseen`
- `outputs/phase1_floors.csv` and `outputs/phase1_metrics.json` exist

If fused is *not* better than no-fusion on tiers B, C and unseen, stop and report. It means the graph edges are wrong (check `edges_by_kind`).

- [ ] **Step 6: Commit**

```bash
git add reserve/pipeline.py reserve/cli_phase1.py tests/test_cli_phase1.py
git commit -m "feat: phase-1 pipeline and CLI"
```

### Task 9: Phase-1 hyperparameter sweep on a validation window

**Files:**
- Create: `reserve/cli_tune_phase1.py`
- Modify: `reserve/config.py` (only `Phase1Config.lam` / `rho` defaults, after the sweep)
- Test: `tests/test_cli_tune_phase1.py`

**Interfaces:**
- Consumes: `prepare` (Task 8), `estimate_floor` (Task 6), `decision_metrics`, `interval_coverage` (Task 7).
- Produces: `reserve.cli_tune_phase1.main(argv=None) -> DataFrame` with columns `lam, rho, iterations, converged, val_*, interval_*`; writes `phase1_tuning.csv`.

- [ ] **Step 1: Write the failing test** — `tests/test_cli_tune_phase1.py`

```python
from reserve.cli_tune_phase1 import main


def test_tuning_cli_one_row_per_combination(data_dir, tmp_path):
    table = main(["--data-dir", str(data_dir), "--out-dir", str(tmp_path / "out"),
                  "--train-end", "2025-10-23", "--train-days", "5", "--test-days", "3",
                  "--lams", "0", "1", "--rhos", "1", "--max-iter", "50"])
    assert len(table) == 2
    assert {"lam", "rho", "converged", "val_n"} <= set(table.columns)
    assert (tmp_path / "out" / "phase1_tuning.csv").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_cli_tune_phase1.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reserve.cli_tune_phase1'`

- [ ] **Step 3: Write the implementation**

`reserve/cli_tune_phase1.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_cli_tune_phase1.py -v`
Expected: all PASS. Then run the whole suite: `.venv/Scripts/python -m pytest -q` — all PASS.

- [ ] **Step 5: Run the sweep and choose hyperparameters**

Run: `.venv/Scripts/python -m reserve.cli_tune_phase1`
Expected: 9 rows (3 λ × 3 ρ) printed as they finish, about 2–4 minutes each; `outputs/phase1_tuning.csv` is written.
Choose the (λ, ρ) with the highest `val_balanced_accuracy`. Among ties within 0.005, prefer `converged=True` and then lower `val_log_loss`. Set the defaults to the chosen values by editing `lam` and `rho` in `Phase1Config` in `reserve/config.py`. Re-run `.venv/Scripts/python -m pytest -q` (all PASS) and `.venv/Scripts/python -m reserve.cli_phase1`, and confirm the test-window accuracy did not drop.

- [ ] **Step 6: Commit**

```bash
git add reserve/cli_tune_phase1.py reserve/config.py tests/test_cli_tune_phase1.py
git commit -m "feat: phase-1 hyperparameter sweep"
```

### Task 10: Phase-2 demand model

**Files:**
- Create: `reserve/demand.py`
- Test: `tests/test_demand.py`

**Interfaces:**
- Consumes: `solve_network` (Task 5), `pareto_weight` (Task 2), `price_bracket` (Task 6), `Phase1Config`, `Phase2Config`.
- Produces: dataclass `DemandParams(mu, sigma, rate, qty, value)` (per-node arrays; `value = exp(mu + sigma²/2)`).
- Produces: `squared_local_grad(obs, node_weight, n_nodes) -> LocalGrad`.
- Produces: `fit_demand(obs, graph, prior, p1, p2, train_days) -> DemandParams`.

- [ ] **Step 1: Write the failing test** — `tests/test_demand.py`

```python
import numpy as np
import pandas as pd

from reserve.config import Phase1Config, Phase2Config
from reserve.demand import fit_demand
from reserve.graph import NodeGraph
from reserve.observations import Observations


def test_fit_demand_rates_shrink_to_family_and_mu_tracks_bids():
    nodes = pd.DataFrame({"sku": ["1", "2"], "family": ["F", "F"]})
    graph = NodeGraph.from_edges(nodes, [])
    bids = np.array([90.0, 100.0, 110.0, 100.0])
    obs = Observations(node=np.zeros(4, dtype=np.int64), log_bid=np.log(bids),
                       accepted=np.ones(4), weight=np.ones(4), qty=np.full(4, 2.0),
                       date=np.full(4, np.datetime64("2026-01-01")))
    prior = np.log([60.0, 60.0])
    p1 = Phase1Config(max_iter=500, eps=1e-7)
    d = fit_demand(obs, graph, prior, p1, Phase2Config(), train_days=10)

    fam_rate = 4 / 20                       # 4 offers over 10 days x 2 nodes
    assert np.allclose(d.rate, [(4 + 7 * fam_rate) / 17, (0 + 7 * fam_rate) / 17])
    assert np.allclose(d.qty, [2.0, 2.0])
    assert abs(d.mu[0] - np.mean(np.log(bids))) < 0.02
    assert abs(d.mu[1] - np.log(60.0)) < 1e-3
    assert (d.sigma >= Phase2Config().min_sigma).all()
    assert np.allclose(d.value, np.exp(d.mu + d.sigma ** 2 / 2))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_demand.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reserve.demand'`

- [ ] **Step 3: Write the implementation**

`reserve/demand.py`:

```python
"""Phase-2 demand inputs per node: bid level/spread, arrival rate, units per offer."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from reserve.admm_core import LocalGrad, solve_network
from reserve.config import Phase1Config, Phase2Config
from reserve.graph import NodeGraph
from reserve.observations import Observations
from reserve.pareto import pareto_weight
from reserve.phase1 import price_bracket


@dataclass
class DemandParams:
    mu: np.ndarray       # location of log bids
    sigma: np.ndarray    # spread of log bids (family level)
    rate: np.ndarray     # offers per day
    qty: np.ndarray      # units per offer
    value: np.ndarray    # mean bid = exp(mu + sigma^2 / 2)


def squared_local_grad(obs: Observations, node_weight: np.ndarray, n_nodes: int) -> LocalGrad:
    total = np.bincount(obs.node, weights=obs.weight, minlength=n_nodes)
    coef = np.divide(node_weight, total, out=np.zeros(n_nodes), where=total > 0)

    def grad(x: np.ndarray) -> np.ndarray:
        g = obs.weight * (x[obs.node] - obs.log_bid)
        return coef * np.bincount(obs.node, weights=g, minlength=n_nodes)

    return grad


def _shrink_to_family(num: np.ndarray, den: np.ndarray, family: np.ndarray, pseudo: float,
                      default: float) -> np.ndarray:
    """(num + pseudo * family_ratio) / (den + pseudo), family_ratio = sum(num) / sum(den)."""
    fam_num = pd.Series(num).groupby(family).transform("sum").to_numpy()
    fam_den = pd.Series(den).groupby(family).transform("sum").to_numpy()
    ratio = np.divide(fam_num, fam_den, out=np.full(len(num), default), where=fam_den > 0)
    return (num + pseudo * ratio) / (den + pseudo)


def fit_demand(obs: Observations, graph: NodeGraph, prior: np.ndarray, p1: Phase1Config,
               p2: Phase2Config, train_days: int) -> DemandParams:
    n = graph.n_nodes
    family = graph.nodes["family"].to_numpy()
    counts = np.bincount(obs.node, minlength=n).astype(float)
    weight = pareto_weight(counts, p1.k_pareto)
    mu = solve_network(n, squared_local_grad(obs, weight, n), graph, prior, p1.prior_weight,
                       p1.lam, p1.rho, p1.max_iter, p1.eps, p1.bisect_iters,
                       price_bracket(obs, prior)).x

    resid = obs.log_bid - mu[obs.node]
    fam_sigma = pd.Series(resid).groupby(family[obs.node]).std()
    global_sigma = float(np.std(resid)) if len(resid) > 1 else p2.min_sigma
    sigma = (graph.nodes["family"].map(fam_sigma).fillna(global_sigma)
             .clip(lower=p2.min_sigma).to_numpy(dtype=float))

    days = np.full(n, float(train_days))
    rate = _shrink_to_family(counts, days, family, p2.rate_prior_days, 0.0)
    qty_sum = np.bincount(obs.node, weights=obs.qty, minlength=n)
    qty = _shrink_to_family(qty_sum, counts, family, p2.rate_prior_days, 1.0)
    return DemandParams(mu, sigma, rate, qty, np.exp(mu + 0.5 * sigma ** 2))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_demand.py -v`
Expected: all PASS. Then run the whole suite: `.venv/Scripts/python -m pytest -q` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add reserve/demand.py tests/test_demand.py
git commit -m "feat: phase-2 demand parameters pooled with ADMM"
```

### Task 11: Depreciation and inventory proxy

**Files:**
- Create: `reserve/market.py`
- Test: `tests/test_market.py`

**Interfaces:**
- Consumes: `attach_nodes`, `window` (Task 4); sales shaped like `load_sales` output.
- Produces: `daily_depreciation(sales, nodes, default, min_rows=30, cap=0.01) -> np.ndarray` (per node, from a family-level within-node slope).
- Produces: `proxy_inventory(sales, nodes, start, end) -> np.ndarray` (units sold per node in `[start, end)`).

- [ ] **Step 1: Write the failing test** — `tests/test_market.py`

```python
import numpy as np
import pandas as pd

from reserve.market import daily_depreciation, proxy_inventory


def _nodes():
    return pd.DataFrame({"sku": ["1", "2", "3"], "warehouse": ["W1", "W1", "W1"],
                         "family": ["F", "F", "G"]})


def _sales():
    days = np.arange(40)
    f = pd.DataFrame({"sku": "1", "warehouse": "W1",
                      "date": pd.Timestamp("2026-01-01") + pd.to_timedelta(days, unit="D"),
                      "unit_price": 100.0 * np.exp(-0.005 * days), "qty": 1.0})
    g = pd.DataFrame({"sku": "3", "warehouse": "W1",
                      "date": pd.to_datetime(["2026-01-01", "2026-01-05"]),
                      "unit_price": [50.0, 40.0], "qty": [2.0, 3.0]})
    return pd.concat([f, g], ignore_index=True)


def test_daily_depreciation_recovers_slope_and_falls_back_for_thin_family():
    delta = daily_depreciation(_sales(), _nodes(), default=0.002)
    assert np.allclose(delta[:2], 0.005, atol=1e-9)
    assert delta[2] == 0.002


def test_proxy_inventory_sums_units_in_window():
    inv = proxy_inventory(_sales(), _nodes(), "2026-01-01", "2026-01-03")
    assert inv.tolist() == [2.0, 0.0, 2.0]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_market.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reserve.market'`

- [ ] **Step 3: Write the implementation**

`reserve/market.py`:

```python
"""Per-node depreciation and stock inputs taken from historical sales."""
import numpy as np
import pandas as pd

from reserve.observations import attach_nodes, window


def daily_depreciation(sales: pd.DataFrame, nodes: pd.DataFrame, default: float,
                       min_rows: int = 30, cap: float = 0.01) -> np.ndarray:
    """Family-level daily price decline from a within-node regression of log price on day."""
    idx = attach_nodes(sales, nodes)
    s = sales[idx >= 0].copy()
    if s.empty:
        return np.full(len(nodes), default)
    s["node"] = idx[idx >= 0]
    s["family"] = nodes["family"].to_numpy()[s["node"]]
    s["t"] = (s["date"] - s["date"].min()).dt.days.astype(float)
    s["lp"] = np.log(s["unit_price"])
    s["t_c"] = s["t"] - s.groupby("node")["t"].transform("mean")
    s["lp_c"] = s["lp"] - s.groupby("node")["lp"].transform("mean")
    s["xy"] = s["t_c"] * s["lp_c"]
    s["xx"] = s["t_c"] ** 2
    agg = s.groupby("family")[["xy", "xx"]].sum()
    rows = s.groupby("family").size()
    slope = agg["xy"] / agg["xx"].where(agg["xx"] > 0)
    delta = (-slope).clip(lower=0.0, upper=cap).where(rows >= min_rows)
    return nodes["family"].map(delta).fillna(default).to_numpy(dtype=float)


def proxy_inventory(sales: pd.DataFrame, nodes: pd.DataFrame, start, end) -> np.ndarray:
    """Units Assurant actually sold per node in [start, end): stand-in for stock to clear.

    The Inventory extract (Jul-Aug 2025) does not overlap the offer history, so this
    proxy is used until a full inventory extract arrives.
    """
    w = window(sales, start, end)
    idx = attach_nodes(w, nodes)
    keep = idx >= 0
    return np.bincount(idx[keep], weights=w["qty"].to_numpy(dtype=float)[keep],
                       minlength=len(nodes))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_market.py -v`
Expected: all PASS. Then run the whole suite: `.venv/Scripts/python -m pytest -q` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add reserve/market.py tests/test_market.py
git commit -m "feat: depreciation estimate and inventory proxy"
```

### Task 12: Phase-2 economics, ADMM solver and Pareto frontier

**Files:**
- Create: `reserve/phase2.py`
- Test: `tests/test_phase2.py`

**Interfaces:**
- Consumes: `edge_prox` (Task 5), `DemandParams` (Task 10), `NodeGraph` (Task 3), `Phase2Config`.
- Produces: dataclass `Economics(mu, sigma, rate, qty, value, inventory, continuation, baseline_logr, warehouse, lo, hi)`.
- Produces: `expected_outcomes(logr, mu, sigma, rate, qty, inventory) -> (sold, revenue)` and `node_value(logr, econ) -> (objective, sold, revenue)`. `logr` may be shape (N,) or (N, G).
- Produces: `build_economics(demand, delta, inventory, baseline_logr, node_weight, warehouse, cfg) -> Economics`, `baseline_targets(econ, multiplier) -> np.ndarray[n_warehouses]`.
- Produces: `solve_phase2(econ, graph, targets, cfg) -> Phase2Result(logr, sold, revenue, objective, dual, iterations, converged, ordering_violations, primal_residual)`.
- Produces: `frontier(econ, graph, multipliers, cfg) -> (DataFrame, dict[multiplier, Phase2Result])`.

- [ ] **Step 1: Write the failing test** — `tests/test_phase2.py`

```python
import numpy as np
import pandas as pd

from reserve.config import Phase2Config
from reserve.demand import DemandParams
from reserve.graph import NodeGraph
from reserve.phase2 import (baseline_targets, build_economics, expected_outcomes, frontier,
                            node_value, solve_phase2)


def _econ(n=1, mu=None, inventory=None, warehouse=None, baseline=None, delta=0.002):
    mu = np.log(np.full(n, 100.0)) if mu is None else np.asarray(mu, dtype=float)
    sigma = np.full(n, 0.1)
    demand = DemandParams(mu=mu, sigma=sigma, rate=np.full(n, 3.0), qty=np.ones(n),
                          value=np.exp(mu + sigma ** 2 / 2))
    inventory = np.full(n, 5.0) if inventory is None else np.asarray(inventory, dtype=float)
    warehouse = np.zeros(n, dtype=int) if warehouse is None else np.asarray(warehouse)
    baseline = mu.copy() if baseline is None else np.asarray(baseline, dtype=float)
    return build_economics(demand, np.full(n, delta), inventory, baseline, np.ones(n),
                           warehouse, Phase2Config())


def _graph(n, edges=()):
    return NodeGraph.from_edges(pd.DataFrame({"sku": [str(i) for i in range(n)]}), list(edges))


def test_expected_outcomes_low_reserve_sells_demand_at_mean_bid():
    sold, revenue = expected_outcomes(np.log(1.0), np.log(100.0), 0.1, 2.0, 1.0, 10.0)
    assert np.isclose(sold, 2.0)
    assert np.isclose(revenue, 2.0 * 100.0 * np.exp(0.005))


def test_expected_outcomes_stock_caps_units():
    sold, revenue = expected_outcomes(np.log(1.0), np.log(100.0), 0.1, 2.0, 1.0, 1.0)
    assert np.isclose(sold, 1.0)
    assert np.isclose(revenue, 100.0 * np.exp(0.005))


def test_expected_outcomes_high_reserve_sells_nothing():
    sold, revenue = expected_outcomes(np.log(1000.0), np.log(100.0), 0.1, 2.0, 1.0, 10.0)
    assert sold < 1e-12 and revenue < 1e-9


def test_more_stock_lowers_continuation_value():
    econ = _econ(n=2, inventory=[1.0, 500.0])
    assert econ.continuation[1] < econ.continuation[0]


def test_unconstrained_solve_matches_grid_argmax():
    econ = _econ()
    res = solve_phase2(econ, _graph(1), np.zeros(1), Phase2Config())
    grid = np.linspace(econ.lo[0], econ.hi[0], Phase2Config().grid_points)
    best = grid[np.argmax(node_value(grid, econ)[0])]
    assert np.isclose(res.logr[0], best)


def test_higher_clearance_target_lowers_reserve_and_sells_more():
    econ = _econ(n=2)
    free = solve_phase2(econ, _graph(2), np.zeros(1), Phase2Config())
    push = solve_phase2(econ, _graph(2), baseline_targets(econ, 1.3), Phase2Config(max_iter=500))
    assert push.sold.sum() > free.sold.sum()
    assert np.all(push.logr <= free.logr + 1e-12)
    assert push.dual[0] > 0


def test_ordered_edge_keeps_better_sibling_priced_higher():
    # node 1 has richer bids, but node 0 is declared the more valuable sibling
    econ = _econ(n=2, mu=np.log([100.0, 115.0]), baseline=np.log([108.0, 108.0]))
    g = _graph(2, [(0, 1, 0.0, True, "grade")])
    loose = solve_phase2(econ, _graph(2), np.zeros(1), Phase2Config())
    tied = solve_phase2(econ, g, np.zeros(1), Phase2Config(max_iter=1000))
    assert loose.logr[0] < loose.logr[1]
    step = (econ.hi[0] - econ.lo[0]) / (Phase2Config().grid_points - 1)
    assert tied.logr[0] >= tied.logr[1] - step
    assert tied.ordering_violations == 0


def test_frontier_rows_per_multiplier():
    econ = _econ(n=2)
    table, results = frontier(econ, _graph(2), [0.0, 1.2], Phase2Config())
    assert table["multiplier"].tolist() == [0.0, 1.2]
    assert table.loc[1, "expected_units"] >= table.loc[0, "expected_units"]
    assert set(results) == {0.0, 1.2}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_phase2.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reserve.phase2'`

- [ ] **Step 3: Write the implementation**

`reserve/phase2.py`:

```python
"""Phase 2: ROI-optimal reserve per node with ADMM + warehouse clearance duals.

maximize  sum_i  f_i(y_i)                       y_i = log reserve
s.t.      sum_{i in w} sold_i(y_i) >= T_w        warehouse clearance target (dual mu_w)
          y_src >= y_dst on ordered sibling edges (ADMM edge copies)
          y_i in [baseline_i - kappa_i, baseline_i + kappa_i]

f_i(y) = expected revenue today + unsold units * continuation value per unit.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import ndtr

from reserve.admm_core import edge_prox
from reserve.config import Phase2Config
from reserve.demand import DemandParams
from reserve.graph import NodeGraph


@dataclass
class Economics:
    mu: np.ndarray
    sigma: np.ndarray
    rate: np.ndarray
    qty: np.ndarray
    value: np.ndarray
    inventory: np.ndarray
    continuation: np.ndarray   # value of holding one unit past today
    baseline_logr: np.ndarray  # phase-1 floor
    warehouse: np.ndarray      # int code per node
    lo: np.ndarray
    hi: np.ndarray


@dataclass
class Phase2Result:
    logr: np.ndarray
    sold: np.ndarray
    revenue: np.ndarray
    objective: np.ndarray
    dual: np.ndarray
    iterations: int
    converged: bool
    ordering_violations: int
    primal_residual: list = field(default_factory=list)


def expected_outcomes(logr, mu, sigma, rate, qty, inventory):
    """Expected units sold and revenue in one day when bids >= exp(logr) are accepted.

    Log bids ~ N(mu, sigma^2), offers ~ Poisson(rate), qty units each. When stock binds,
    qualifying bids are filled at random, so revenue scales with sold / demand.
    """
    z = (logr - mu) / sigma
    survive = 1.0 - ndtr(z)
    demand = rate * qty * survive
    sold = np.minimum(inventory, demand)
    fill = np.divide(sold, demand, out=np.zeros_like(demand), where=demand > 0)
    revenue = np.exp(mu + 0.5 * sigma ** 2) * ndtr(sigma - z) * rate * qty * fill
    return sold, revenue


def _col(a: np.ndarray, like: np.ndarray) -> np.ndarray:
    return a[:, None] if like.ndim == 2 else a


def node_value(logr: np.ndarray, econ: Economics):
    """(objective, sold, revenue); logr is (N,) or (N, G)."""
    sold, revenue = expected_outcomes(
        logr, _col(econ.mu, logr), _col(econ.sigma, logr), _col(econ.rate, logr),
        _col(econ.qty, logr), _col(econ.inventory, logr))
    objective = revenue + (_col(econ.inventory, logr) - sold) * _col(econ.continuation, logr)
    return objective, sold, revenue


def build_economics(demand: DemandParams, delta: np.ndarray, inventory: np.ndarray,
                    baseline_logr: np.ndarray, node_weight: np.ndarray,
                    warehouse: np.ndarray, cfg: Phase2Config) -> Economics:
    kappa = cfg.kappa_min + (cfg.kappa_max - cfg.kappa_min) * node_weight
    survive = 1.0 - ndtr((baseline_logr - demand.mu) / demand.sigma)
    daily_units = np.maximum(demand.rate * demand.qty * survive, 1e-6)
    days = np.clip(inventory / daily_units, 1.0, cfg.max_days_to_clear)
    continuation = demand.value * np.clip(1.0 - delta - cfg.holding_frac, 0.0, 1.0) ** days
    return Economics(demand.mu, demand.sigma, demand.rate, demand.qty, demand.value,
                     inventory.astype(float), continuation, baseline_logr,
                     warehouse.astype(np.int64), baseline_logr - kappa, baseline_logr + kappa)


def baseline_targets(econ: Economics, multiplier: float) -> np.ndarray:
    _, sold, _ = node_value(econ.baseline_logr, econ)
    n_wh = int(econ.warehouse.max()) + 1
    return multiplier * np.bincount(econ.warehouse, weights=sold, minlength=n_wh)


def solve_phase2(econ: Economics, graph: NodeGraph, targets: np.ndarray,
                 cfg: Phase2Config) -> Phase2Result:
    n = len(econ.mu)
    keep = graph.ordered
    src, dst, off = graph.src[keep], graph.dst[keep], graph.offset[keep]
    ordered, theta = np.ones(len(src), dtype=bool), np.zeros(len(src))
    rows = np.arange(n)

    grid = econ.lo[:, None] + (econ.hi - econ.lo)[:, None] * np.linspace(0, 1, cfg.grid_points)
    g_obj, g_sold, _ = node_value(grid, econ)
    scale = (econ.value * np.maximum(econ.inventory, 1.0))[:, None]
    active = econ.inventory > 0
    hold_baseline = -((grid - econ.baseline_logr[:, None]) ** 2)

    n_wh = len(targets)
    dual = np.zeros(n_wh)
    y = econ.baseline_logr.copy()
    z_s, z_d = y[src].copy(), y[dst].copy()
    u_s, u_d = np.zeros(len(src)), np.zeros(len(src))
    deg = (np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)).astype(float)
    tol = cfg.eps * np.sqrt(max(2 * len(src), 1))
    res = Phase2Result(y, np.zeros(n), np.zeros(n), np.zeros(n), dual, 0, False, 0)

    for it in range(1, cfg.max_iter + 1):
        sum_v = (np.bincount(src, weights=z_s - u_s, minlength=n)
                 + np.bincount(dst, weights=z_d - u_d, minlength=n))
        score = ((g_obj + dual[econ.warehouse][:, None] * g_sold) / scale
                 - 0.5 * cfg.rho * (deg[:, None] * grid ** 2 - 2.0 * grid * sum_v[:, None]))
        score = np.where(active[:, None], score, hold_baseline)
        pick = score.argmax(axis=1)
        y = grid[rows, pick]
        sold = g_sold[rows, pick]

        wh_sold = np.bincount(econ.warehouse, weights=sold, minlength=n_wh)
        dual = np.maximum(0.0, dual + cfg.dual_step * (targets - wh_sold) / np.maximum(targets, 1.0))

        zs_old, zd_old = z_s, z_d
        z_s, z_d = edge_prox(y[src] + u_s, y[dst] + u_d, off, ordered, theta)
        u_s = u_s + y[src] - z_s
        u_d = u_d + y[dst] - z_d
        r = float(np.sqrt(np.sum((y[src] - z_s) ** 2 + (y[dst] - z_d) ** 2)))
        s = float(cfg.rho * np.sqrt(np.sum((z_s - zs_old) ** 2 + (z_d - zd_old) ** 2)))
        res.primal_residual.append(r)
        res.iterations = it
        met = np.all((wh_sold >= targets * (1 - 1e-3)) | (dual == 0))
        if r < tol and s < tol and met and it > 1:
            res.converged = True
            break

    objective, sold, revenue = node_value(y, econ)
    res.logr, res.sold, res.revenue, res.objective, res.dual = y, sold, revenue, objective, dual
    step = (econ.hi - econ.lo) / (cfg.grid_points - 1)
    res.ordering_violations = int(np.sum(y[src] < y[dst] - np.maximum(step[src], step[dst])))
    return res


def frontier(econ: Economics, graph: NodeGraph, multipliers: list[float],
             cfg: Phase2Config) -> tuple[pd.DataFrame, dict[float, Phase2Result]]:
    """Sweep the clearance target; each solve is one point of recovery vs units cleared."""
    rows, results = [], {}
    for m in multipliers:
        res = solve_phase2(econ, graph, baseline_targets(econ, m), cfg)
        results[m] = res
        rows.append({
            "multiplier": m,
            "expected_units": float(res.sold.sum()),
            "expected_revenue": float(res.revenue.sum()),
            "expected_value": float(res.objective.sum()),
            "mean_price_change_pct": float(np.mean(np.exp(res.logr - econ.baseline_logr)) - 1) * 100,
            "binding_warehouses": int(np.sum(res.dual > 0)),
            "iterations": res.iterations,
            "converged": res.converged,
            "ordering_violations": res.ordering_violations,
        })
    return pd.DataFrame(rows), results
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_phase2.py -v`
Expected: all PASS. Then run the whole suite: `.venv/Scripts/python -m pytest -q` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add reserve/phase2.py tests/test_phase2.py
git commit -m "feat: phase-2 ROI ADMM with warehouse duals and frontier"
```

### Task 13: Replay backtest

**Files:**
- Create: `reserve/replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Consumes: `Observations` (Task 4).
- Produces: `replay(reserve_log, obs, inventory, value, delta, horizon_days, accept_mask=None) -> dict(revenue, units, accepted_offers, terminal_value, total_value)`.

- [ ] **Step 1: Write the failing test** — `tests/test_replay.py`

```python
import numpy as np

from reserve.observations import Observations
from reserve.replay import replay


def _obs():
    return Observations(
        node=np.zeros(3, dtype=np.int64), log_bid=np.log([90.0, 50.0, 100.0]),
        accepted=np.array([0.0, 1.0, 1.0]), weight=np.ones(3), qty=np.array([3.0, 2.0, 3.0]),
        date=np.array(["2026-01-01"] * 3, dtype="datetime64[ns]"))


def test_replay_fills_highest_bids_first_until_stock_runs_out():
    out = replay(np.log([80.0]), _obs(), np.array([5.0]), np.array([100.0]),
                 np.array([0.0]), horizon_days=14)
    assert out["units"] == 5.0
    assert np.isclose(out["revenue"], 3 * 100.0 + 2 * 90.0)
    assert out["accepted_offers"] == 2
    assert out["terminal_value"] == 0.0


def test_replay_marks_leftover_stock_with_depreciation():
    out = replay(np.log([95.0]), _obs(), np.array([5.0]), np.array([100.0]),
                 np.array([0.01]), horizon_days=10)
    assert out["units"] == 3.0
    assert np.isclose(out["terminal_value"], 2 * 100.0 * 0.99 ** 10)


def test_replay_accept_mask_replays_actual_decisions():
    obs = _obs()
    out = replay(np.zeros(1), obs, np.array([10.0]), np.array([100.0]), np.array([0.0]), 14,
                 accept_mask=obs.accepted == 1)
    assert np.isclose(out["revenue"], 3 * 100.0 + 2 * 50.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_replay.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reserve.replay'`

- [ ] **Step 3: Write the implementation**

`reserve/replay.py`:

```python
"""Replay hold-out offers against a reserve policy with a stock cap per node."""
import numpy as np

from reserve.observations import Observations


def replay(reserve_log: np.ndarray, obs: Observations, inventory: np.ndarray,
           value: np.ndarray, delta: np.ndarray, horizon_days: int,
           accept_mask: np.ndarray | None = None) -> dict:
    """Walk offers by date, node, highest bid first; accept if bid >= reserve and stock left.

    accept_mask overrides the reserve rule (used to replay Assurant's actual decisions).
    Unsold stock is marked at value * (1 - delta) ** horizon_days.
    """
    order = np.lexsort((-obs.log_bid, obs.node, obs.date))
    remaining = inventory.astype(float).copy()
    revenue = units = 0.0
    accepted = 0
    for k in order:
        i = obs.node[k]
        ok = accept_mask[k] if accept_mask is not None else obs.log_bid[k] >= reserve_log[i]
        if ok and remaining[i] > 0:
            take = min(obs.qty[k], remaining[i])
            remaining[i] -= take
            units += take
            revenue += take * float(np.exp(obs.log_bid[k]))
            accepted += 1
    terminal = float(np.sum(remaining * value * (1.0 - delta) ** horizon_days))
    return {"revenue": revenue, "units": units, "accepted_offers": accepted,
            "terminal_value": terminal, "total_value": revenue + terminal}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_replay.py -v`
Expected: all PASS. Then run the whole suite: `.venv/Scripts/python -m pytest -q` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add reserve/replay.py tests/test_replay.py
git commit -m "feat: replay backtest with stock cap and terminal value"
```

### Task 14: Phase-2 pipeline and CLI (real-data checkpoint)

**Files:**
- Modify: `reserve/pipeline.py` (replace the whole file with the version below — it adds Phase-2 imports and `run_phase2`)
- Create: `reserve/cli_phase2.py`
- Test: `tests/test_cli_phase2.py`

**Interfaces:**
- Consumes: everything from Tasks 1–13.
- Produces: `run_phase2(ctx, phase1: Phase1Output, p1, p2, multipliers) -> (frontier_table, replay_table, reserves)`. The replay policies are `assurant_actual`, `phase1_floor`, and `phase2_x{m}` for each multiplier. `reserves` holds per-node detail at the first multiplier.
- Produces: `reserve.cli_phase2.main(argv=None)`, which writes `phase2_frontier.csv`, `phase2_replay.csv` and `phase2_reserves.csv`.

- [ ] **Step 1: Write the failing test** — `tests/test_cli_phase2.py`

```python
from reserve.cli_phase2 import main


def test_phase2_cli_writes_frontier_replay_and_reserves(data_dir, tmp_path):
    out = tmp_path / "out"
    table, replays, reserves = main([
        "--data-dir", str(data_dir), "--out-dir", str(out), "--train-end", "2025-10-23",
        "--train-days", "5", "--test-days", "3", "--multipliers", "0", "1.2"])
    for name in ("phase2_frontier.csv", "phase2_replay.csv", "phase2_reserves.csv"):
        assert (out / name).exists()
    assert table["multiplier"].tolist() == [0.0, 1.2]
    assert replays["policy"].tolist() == [
        "assurant_actual", "phase1_floor", "phase2_x0.0", "phase2_x1.2"]
    assert len(reserves) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_cli_phase2.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reserve.cli_phase2'`

- [ ] **Step 3: Write the implementation**

`reserve/pipeline.py`:

```python
"""End-to-end wiring: load data, build graph, run phase 1 (and phase 2)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from reserve.config import Phase1Config, Phase2Config
from reserve.demand import fit_demand
from reserve.evaluate import decision_metrics, interval_coverage, metrics_by_tier
from reserve.graph import NodeGraph, build_graph, build_nodes
from reserve.loaders import load_offers, load_products, load_sales
from reserve.market import daily_depreciation, proxy_inventory
from reserve.observations import Observations, node_priors, observations, window
from reserve.phase1 import Phase1Output, estimate_floor
from reserve.phase2 import build_economics, frontier
from reserve.replay import replay


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


def run_phase2(ctx: Context, phase1: Phase1Output, p1: Phase1Config, p2: Phase2Config,
               multipliers: list[float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (frontier table, replay table, per-node reserves at multiplier 0)."""
    train_days = (ctx.train_end - ctx.train_start).days
    horizon = (ctx.test_end - ctx.train_end).days
    demand = fit_demand(ctx.train_obs, ctx.graph, ctx.prior, p1, p2, train_days)
    delta = daily_depreciation(ctx.sales[ctx.sales["date"] < ctx.train_end], ctx.nodes,
                               p2.default_delta)
    inventory = proxy_inventory(ctx.sales, ctx.nodes, ctx.train_end, ctx.test_end)
    warehouse = pd.factorize(ctx.nodes["warehouse"])[0]
    econ = build_economics(demand, delta, inventory, phase1.floor_log, phase1.node_weight,
                           warehouse, p2)
    table, results = frontier(econ, ctx.graph, multipliers, p2)

    def _replay(name, reserve_log, mask=None):
        out = replay(reserve_log, ctx.test_obs, inventory, demand.value, delta, horizon, mask)
        return {"policy": name, **out}

    rows = [_replay("assurant_actual", phase1.floor_log, ctx.test_obs.accepted == 1),
            _replay("phase1_floor", phase1.floor_log)]
    rows += [_replay(f"phase2_x{m}", res.logr) for m, res in results.items()]
    replays = pd.DataFrame(rows)

    first = results[multipliers[0]]
    reserves = ctx.nodes[["sku", "warehouse", "family", "grade", "capacity_gb"]].copy()
    reserves["inventory_proxy"] = inventory
    reserves["phase1_floor"] = np.exp(phase1.floor_log)
    reserves["phase2_reserve"] = np.exp(first.logr)
    reserves["continuation_value"] = econ.continuation
    reserves["expected_units"] = first.sold
    reserves["warehouse_dual"] = first.dual[warehouse]
    return table, replays, reserves
```

`reserve/cli_phase2.py`:

```python
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
    args = ap.parse_args(argv)

    started = time.perf_counter()
    p1, p2 = Phase1Config(), Phase2Config()
    ctx = prepare(args.data_dir, args.train_end, args.train_days, args.test_days, p1)
    _, _, phase1 = run_phase1(ctx, p1)
    table, replays, reserves = run_phase2(ctx, phase1, p1, p2, args.multipliers)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "phase2_frontier.csv", index=False)
    replays.to_csv(args.out_dir / "phase2_replay.csv", index=False)
    reserves.to_csv(args.out_dir / "phase2_reserves.csv", index=False)
    print(table.to_string(index=False))
    print()
    print(replays.to_string(index=False))
    print(f"\nseconds: {time.perf_counter() - started:.1f}")
    return table, replays, reserves


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_cli_phase2.py -v`
Expected: all PASS. Then run the whole suite: `.venv/Scripts/python -m pytest -q` — all PASS.

- [ ] **Step 5: Run Phase 2 on the real data**

Run: `.venv/Scripts/python -m reserve.cli_phase2`
Expected: finishes in about 3–5 minutes and prints the frontier table, then the replay table. Sanity checks (the prototype got these values):
- frontier `expected_units` does not decrease as the multiplier grows; `binding_warehouses` grows from 0
- `mean_price_change_pct` is within ±15% (the trust-box bound)
- replay `phase2_x0.0` total_value ≈ 91M vs `assurant_actual` ≈ 88M
- `outputs/phase2_frontier.csv`, `phase2_replay.csv` and `phase2_reserves.csv` exist

The prototype did not converge in 200 iterations and had about 365 ordering violations (out of about 20.6K ordered edges). Record the numbers you get; do not tune here.

- [ ] **Step 6: Commit**

```bash
git add reserve/pipeline.py reserve/cli_phase2.py tests/test_cli_phase2.py
git commit -m "feat: phase-2 pipeline, frontier and replay CLI"
```

### Task 15: Results write-up for the mentor

**Files:**
- Create: `docs/04-results.md`
- Modify: `README.md` (add a row for `docs/04-results.md` to the Docs table)

**Interfaces:**
- Consumes: `outputs/phase1_metrics.json`, `outputs/phase1_tuning.csv`, `outputs/phase2_frontier.csv`, `outputs/phase2_replay.csv`, `outputs/phase2_reserves.csv` from Tasks 8, 9 and 14.

- [ ] **Step 1: Pull the headline numbers**

```bash
.venv/Scripts/python -c "
import json, pandas as pd
m = json.load(open('outputs/phase1_metrics.json'))
for k in ('family_prior', 'admm_no_fusion', 'admm_fused'):
    t = m[k]['by_tier']
    print(k, round(m[k]['overall']['accuracy'], 3), {x: round(t[x]['accuracy'], 3) for x in t}, round(m[k]['intervals']['coverage'], 3))
print(pd.read_csv('outputs/phase1_tuning.csv').sort_values('val_balanced_accuracy').tail(3).to_string(index=False))
print(pd.read_csv('outputs/phase2_frontier.csv').to_string(index=False))
print(pd.read_csv('outputs/phase2_replay.csv').to_string(index=False))
r = pd.read_csv('outputs/phase2_reserves.csv'); a = r[r.inventory_proxy > 0]
chg = a.phase2_reserve / a.phase1_floor - 1
print('active', len(a), 'inventory-weighted change %', round((chg * a.inventory_proxy).sum() / a.inventory_proxy.sum() * 100, 2))
"
```

- [ ] **Step 2: Write `docs/04-results.md`** with exactly these sections, filled with the numbers printed in Step 1:
  1. **Setup:** windows (train / validation / test dates), chosen λ and ρ, graph size by edge kind.
  2. **Phase 1:** a table of accuracy, balanced accuracy, accuracy by tier (A, B, C, unseen) and interval coverage for family prior, ADMM λ = 0 and ADMM fused. Add one sentence on whether pooling helped the tail.
  3. **Phase 2 frontier:** the frontier table, plus one sentence on the recovery vs clearance trade-off.
  4. **Replay:** the replay table, plus the inventory-weighted price change.
  5. **Caveats:** copy the four limitations from `docs/02-data-findings.md` ("Known limitations"), plus convergence status and ordering violations.
  6. **Asks for Assurant:** copy the "Open items" list from `docs/01-project-context.md`.

- [ ] **Step 3: Verify the suite is still green**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all PASS (46 tests).

- [ ] **Step 4: Commit**

```bash
git add docs/04-results.md README.md
git commit -m "docs: POC results for phase 1 and phase 2"
```

---

## Notes for the executor

- Tasks 8, 9 and 14 are **checkpoints**. After each one, report the printed metrics before continuing.
- Do not "fix" non-convergence by raising `max_iter` past 1000 in Phase 2. Grid-based x-updates plus dual ascent can oscillate. Record it, and let Task 15 report it.
- The inventory proxy, placeholder factors and replay limitations are deliberate POC simplifications (see the spec, section 4). Do not try to remove them.
