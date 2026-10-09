# Reserve Price CLM Fine-Tuning POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and locally verify a self-contained Colab/Kaggle notebook that fine-tunes CLM-8B to predict a daily reserve-price band per SKU × Warehouse and evaluates it against Assurant's historical floors.

**Architecture:** Small tested Python modules in `reserve_clm/` (load → label → features → text → embed → train → evaluate). `tools/build_notebook.py` inlines every module into the notebook with `%%writefile` cells and adds thin driver cells, so the notebook runs exactly the tested code. The notebook is executed locally in smoke mode (Qwen3-0.6B on CPU, subsampled data) before hand-off.

**Tech Stack:** Python 3.11–3.13, pandas (2.2 and 3.x compatible), numpy, PyTorch, transformers ≥ 4.51 (Qwen3), `contrastive-lm` (CLM repo, pinned commit `d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6`), pytest, nbformat, nbclient, ipykernel.

**Spec:** `poc/CML/docs/2026-10-09-reserve-price-clm-design.md`

**Working directory:** every path and command in this plan is relative to `poc/CML/` (run all commands from there). The data lives at `../../Data` (repo root `Data/`).

## Global Constraints

- No new data: only `Data/Data Files/Offers/Offers.csv`, `Data/Data Files/Product/Product Information.csv`, `Data/Data Files/Sales/Historical Sales/Sales Order&Line items.csv`.
- `OFFER_STATUS` ∈ {`ACCEPTED`, `COUNTERED`, `NOT_ACCEPTED`} is the only outcome used; `PENDING_ACCEPTANCE` rows and `ORDER_STATUS` are dropped/ignored.
- Inventory and Buyer Profile files are not used (no date overlap / label leakage).
- Every feature for target day `d` uses only data from days `< d`.
- Splits: train `day < 2026-02-01`, val `2026-02-01 ≤ day < 2026-03-01`, test `day ≥ 2026-03-01`.
- 14 bands: edges `0.70 | 0.80, 0.825, …, 1.10 | 1.20` (outer edges only bound the tails for overlap maths).
- Reference price needs ≥ 3 accepted offers in the 30-day window (`MIN_REF_ACCEPTS = 3`).
- Main set = `ref_source == "sku"`; fallback set = `ref_source == "family"` (evaluated separately, never trained on).
- CLM fine-tuning only — no zero-shot evaluation anywhere.
- Notebook must be runnable end to end with no edits other than pointing at the data folder.
- Workspace is not a git repository: "Checkpoint" steps run the full test suite instead of committing. If the user enables git, commit at each checkpoint.

## File Structure

| Path | Responsibility |
|---|---|
| `reserve_clm/__init__.py` | package marker |
| `reserve_clm/config.py` | dates, windows, thresholds, CLM commit |
| `reserve_clm/bands.py` | band edges, labels, midpoints, `band_of`, `soft_targets` |
| `reserve_clm/data.py` | locate + load + clean offers, sales, products |
| `reserve_clm/labels.py` | floor intervals from `OFFER_STATUS`, split assignment |
| `reserve_clm/features.py` | as-of features (reference price, bids, sales, last floor) |
| `reserve_clm/text.py` | state text + band descriptions |
| `reserve_clm/baselines.py` | B0/B1/B2 rule baselines |
| `reserve_clm/evaluate.py` | metric primitives (interval, band, replay, calibration) |
| `reserve_clm/report.py` | method comparison tables, breakdowns, examples |
| `reserve_clm/embed.py` | Qwen last-token embedder with disk cache |
| `reserve_clm/train.py` | CLM head model, soft InfoNCE training, prediction, checkpoint |
| `reserve_clm/pipeline.py` | `RunConfig`, `prepare`, `views`, `embed_all`, `run_training`, `evaluate_all` |
| `tests/…` | one test file per module + real-data integration test |
| `tools/build_notebook.py` | generate `notebooks/reserve_price_clm.ipynb` |
| `tools/run_smoke.py` | execute the notebook locally in smoke mode |
| `requirements-dev.txt`, `pytest.ini`, `tests/conftest.py` | local tooling |

---

### Task 1: Local environment, config and bands

**Files:**
- Create: `requirements-dev.txt`, `pytest.ini`, `tests/conftest.py`, `reserve_clm/__init__.py`, `reserve_clm/config.py`, `reserve_clm/bands.py`
- Test: `tests/test_bands.py`

**Interfaces:**
- Produces: `config.TRAIN_END`, `config.VAL_END`, `config.REF_WINDOW_DAYS=30`, `config.TREND_WINDOW_DAYS=7`, `config.MIN_REF_ACCEPTS=3`, `config.OFFER_STATUSES`, `config.CLM_REPO_COMMIT`; `bands.EDGES (np.ndarray, 15)`, `bands.N_BANDS=14`, `bands.BAND_KEYS (list[str] "b00".."b13")`, `bands.BAND_LABELS (list[str])`, `bands.BAND_MIDS (np.ndarray, 14)`, `bands.band_of(ratio) -> np.ndarray[int]`, `bands.soft_targets(lo, hi) -> np.ndarray[n,14]`.

- [ ] **Step 1: Create tooling files**

`requirements-dev.txt`:
```text
pandas
numpy
pyarrow
pytest
transformers>=4.51
accelerate
huggingface_hub
nbformat
nbclient
ipykernel
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
markers =
    slow: needs a model download or the real data files
```

`tests/conftest.py`:
```python
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT.parents[1] / "Data"          # repo-root Data/ (ROOT = poc/CML)
```

- [ ] **Step 2: Install local dependencies**

Run:
```bash
python -m pip install -q -r requirements-dev.txt
python -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -q --no-deps "contrastive-lm @ git+https://github.com/Contrastive-LM/CLM@d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6"
python -c "import torch, transformers, clm.heads; print(torch.__version__, transformers.__version__)"
```
Expected: versions print, no ImportError.

- [ ] **Step 3: Write the failing test**

`tests/test_bands.py`:
```python
import numpy as np

from reserve_clm.bands import BAND_KEYS, BAND_LABELS, BAND_MIDS, EDGES, N_BANDS, band_of, soft_targets


def test_band_layout():
    assert N_BANDS == 14 and len(BAND_KEYS) == 14 and len(BAND_LABELS) == 14 and len(BAND_MIDS) == 14
    assert EDGES[0] == 0.70 and EDGES[1] == 0.80 and EDGES[-2] == 1.10 and EDGES[-1] == 1.20
    assert BAND_KEYS[0] == "b00" and BAND_KEYS[-1] == "b13"
    assert BAND_LABELS[0] == "<0.80x" and BAND_LABELS[-1] == ">=1.10x"


def test_band_of_edges():
    assert band_of([0.5, 0.799, 0.80, 0.824, 0.825, 1.099, 1.10, 3.0]).tolist() == [0, 0, 1, 1, 2, 12, 13, 13]


def test_soft_targets_point_mass():
    t = soft_targets([0.96], [0.96])
    assert t.shape == (1, 14) and t[0, band_of([0.96])[0]] == 1.0 and t.sum() == 1.0


def test_soft_targets_overlap_is_proportional():
    # 0.94..0.99 overlaps band [0.925,0.95) by 0.01, [0.95,0.975) by 0.025, [0.975,1.0) by 0.015
    t = soft_targets([0.94], [0.99])[0]
    np.testing.assert_allclose(t[band_of([0.94])[0]], 0.2)
    np.testing.assert_allclose(t[band_of([0.96])[0]], 0.5)
    np.testing.assert_allclose(t[band_of([0.98])[0]], 0.3)
    np.testing.assert_allclose(t.sum(), 1.0)


def test_soft_targets_clip_tails():
    t = soft_targets([1.5], [2.0])[0]
    assert t[13] == 1.0
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_bands.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve_clm'`

- [ ] **Step 5: Write minimal implementation**

`reserve_clm/__init__.py`:
```python
"""Reserve-price band prediction with CLM-8B fine-tuning (POC)."""
```

`reserve_clm/config.py`:
```python
"""Project-wide constants for the reserve-price CLM POC."""
import pandas as pd

OFFER_STATUSES = ("ACCEPTED", "COUNTERED", "NOT_ACCEPTED")
TRAIN_END = pd.Timestamp("2026-02-01")   # train: day < TRAIN_END
VAL_END = pd.Timestamp("2026-03-01")     # val: TRAIN_END <= day < VAL_END; test: day >= VAL_END
REF_WINDOW_DAYS = 30
TREND_WINDOW_DAYS = 7
MIN_REF_ACCEPTS = 3                      # accepted offers needed in the window for a reference price
CLM_REPO_COMMIT = "d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6"
```

`reserve_clm/bands.py`:
```python
"""Reserve bands: multiples of the reference (market) price that CLM chooses between."""
import numpy as np

INNER_EDGES = np.round(np.arange(0.80, 1.10001, 0.025), 3)       # 0.800 ... 1.100 (13 edges)
EDGES = np.concatenate([[0.70], INNER_EDGES, [1.20]])             # outer edges bound the tails
N_BANDS = len(EDGES) - 1                                          # 14
BAND_KEYS = [f"b{i:02d}" for i in range(N_BANDS)]
BAND_LABELS = (["<0.80x"] + [f"{EDGES[i]:.3f}-{EDGES[i + 1]:.3f}x" for i in range(1, N_BANDS - 1)]
               + [">=1.10x"])
BAND_MIDS = (EDGES[:-1] + EDGES[1:]) / 2


def band_of(ratio) -> np.ndarray:
    """Band index (0..13) of each ratio; anything below 0.80 is band 0, from 1.10 up band 13."""
    r = np.asarray(ratio, dtype=float)
    return np.searchsorted(EDGES[1:-1], r, side="right")


def soft_targets(lo, hi) -> np.ndarray:
    """Distribution over bands proportional to the overlap with [lo, hi] (ratios).

    A zero-width interval puts all mass on the band containing it.
    """
    lo = np.clip(np.asarray(lo, dtype=float), EDGES[0], EDGES[-1])
    hi = np.clip(np.asarray(hi, dtype=float), EDGES[0], EDGES[-1])
    a, b = EDGES[:-1][None, :], EDGES[1:][None, :]
    overlap = np.clip(np.minimum(hi[:, None], b) - np.maximum(lo[:, None], a), 0.0, None)
    width = overlap.sum(1)
    out = np.zeros_like(overlap)
    wide = width > 1e-12
    out[wide] = overlap[wide] / width[wide, None]
    point = ~wide
    out[point, band_of((lo[point] + hi[point]) / 2)] = 1.0
    return out
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_bands.py -v`
Expected: 5 passed

- [ ] **Step 7: Checkpoint** — Run: `python -m pytest -q` → all pass.

---

### Task 2: Data loading

**Files:**
- Create: `reserve_clm/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `config.OFFER_STATUSES`
- Produces:
  - `find_data_files(roots: list[str]) -> dict[str, str]` with keys `offers`, `products`, `sales`
  - `load_offers(path) -> DataFrame[offer_id, buyer_id, sku, wh, qty(float), price(float), status, ts(datetime64[ns]), day(datetime64[ns]), accepted(bool)]`
  - `load_sales(path) -> DataFrame[sku, wh, status, order_type, units(float), unit_price(float), day]`
  - `load_products(path) -> DataFrame[sku, wh, manufacturer, model, capacity, grade, color, carrier, lock_status, category, family]`

- [ ] **Step 1: Write the failing test**

`tests/test_data.py`:
```python
import pandas as pd
import pytest

from reserve_clm.data import find_data_files, load_offers, load_products, load_sales

OFFERS = """OFFER_ID,BUYER_ID,SKU,WAREHOUSE,OFFER_QUANTITY,OFFER_PRICE,COUNTER_PRICE,OFFER_STATUS,ORDER_STATUS,OFFER_CREATE_DATETIME,TIMESTAMP,SUBMITTED_SHOPPING_CART_ID
1,70,A,W1,6,239,NULL,ACCEPTED,REDEEMED,10/20/2025 13:08,10/20/2025 16:32,1
2,71,A,W1,2,200,210,COUNTERED,NULL,10/20/2025 14:00,10/20/2025 16:32,2
3,72,A,W1,1,150,NULL,NOT_ACCEPTED,NULL,10/21/2025 09:00,x,3
4,73,A,W1,1,150,NULL,PENDING_ACCEPTANCE,NULL,10/21/2025 09:00,x,4
5,74,A,W1,1,0,NULL,ACCEPTED,NULL,10/21/2025 09:00,x,5
"""
SALES = """SALE_ID,BUYER_ID,WAREHOUSE_CODE,CURRENCY_CODE,STATUS,ORDER_TYPE,DATE,SALE_COMPLETION_DATE,SHOPPING_CART_ID,PAYMENT_METHOD,SHIPPING_CARRIER,CPR_ORDER,PX_CONNECT_ORDER,PXN_BUYER_ID,HYLA_ORDER_NUMBER,HYLA_ORDER_STATUS,LINE_ITEM_ID,PRODUCT_ID,LPN,UNIT_PRICE,SALES_QTY,QUANTITY_SHIPPED,QUANTITY_BACKORDERED,QUANTITY_CANCELLED,TOTAL_PRICE
1,418,W1,USD,INVOICED,STOCKLIST,10/19/2025 11:26,NULL,NULL,WIRE,FEDEX,0,0,NULL,1,NULL,1,A,,433,6,6,0,2,2598
2,418,W1,USD,CANCELED,STOCKLIST,10/19/2025 11:26,NULL,NULL,WIRE,FEDEX,0,0,NULL,1,NULL,2,A,,433,6,6,0,0,2598
3,418,W1,USD,INVOICED,AUCTION,10/20/2025 11:26,NULL,NULL,WIRE,FEDEX,0,0,NULL,1,NULL,3,A,,400,1,1,0,1,400
"""
PRODUCTS = """SKU,WAREHOUSE,ITEM_DESCRIPTION,CATEGORY,PRODUCT_FAMILY,MANUFACTURER,MODEL,MODEL_NUMBER,CAPACITY,COLOR,CARRIER,LOCK_STATUS,LAUNCH_DATE,PROTOCOL,CHANNEL,INACTIVE,DEACTIVATED,UPDATE_DATETIME,GRADE_STANDARD,GRADE
A,W1,desc,Phones,fam,Apple,iPhone 13,X1,128GB,Blue,OTH,UNLOCKED,,GSM,UNASSIGNED,0,0,2024-04-10,DLS,B
B,W1,desc,Phones,fam,Apple,iPhone 13,X1,128GB,,OTH,UNLOCKED,,GSM,UNASSIGNED,0,0,2024-04-10,DLS,B
"""


@pytest.fixture
def files(tmp_path):
    (tmp_path / "x" / "Offers").mkdir(parents=True)
    (tmp_path / "x" / "Offers" / "Offers.csv").write_text(OFFERS)
    (tmp_path / "x" / "Sales Order&Line items.csv").write_text(SALES)
    (tmp_path / "Product Information.csv").write_text(PRODUCTS)
    return tmp_path


def test_find_data_files(files):
    found = find_data_files(["", str(files / "missing"), str(files)])
    assert found["offers"].endswith("Offers.csv") and found["sales"].endswith("items.csv")
    with pytest.raises(FileNotFoundError):
        find_data_files([str(files / "x" / "Offers")])


def test_load_offers_filters_and_types(files):
    o = load_offers(str(files / "x" / "Offers" / "Offers.csv"))
    assert o.offer_id.tolist() == ["1", "2", "3"]           # pending and zero price dropped
    assert o.accepted.tolist() == [True, False, False]
    assert o.day.iloc[2] == pd.Timestamp("2025-10-21") and str(o.day.dtype) == "datetime64[ns]"
    assert o.price.tolist() == [239.0, 200.0, 150.0]


def test_load_sales_nets_cancelled(files):
    s = load_sales(str(files / "x" / "Sales Order&Line items.csv"))
    assert len(s) == 1                                      # CANCELED row and 1-1=0 unit row dropped
    assert s.units.iloc[0] == 4 and s.order_type.iloc[0] == "STOCKLIST"


def test_load_products_family(files):
    p = load_products(str(files / "Product Information.csv"))
    assert p.family.tolist() == ["Apple|iPhone 13|128GB|B"] * 2
    assert p.color.tolist() == ["Blue", "unknown"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve_clm.data'`

- [ ] **Step 3: Write minimal implementation**

`reserve_clm/data.py`:
```python
"""Locate and load the three source files, keeping only what the POC trusts."""
from __future__ import annotations

import glob
import os

import pandas as pd

from .config import OFFER_STATUSES

FILES = {"offers": "Offers.csv", "products": "Product Information.csv",
         "sales": "Sales Order&Line items.csv"}
_TS = "%m/%d/%Y %H:%M"


def find_data_files(roots: list[str]) -> dict[str, str]:
    """First match of each required file under the given roots (searched recursively, in order)."""
    out: dict[str, str] = {}
    for name, fname in FILES.items():
        for root in roots:
            if not root or not os.path.isdir(root):
                continue
            hits = sorted(glob.glob(os.path.join(glob.escape(root), "**", glob.escape(fname)), recursive=True))
            if hits:
                out[name] = hits[0]
                break
    missing = [FILES[k] for k in FILES if k not in out]
    if missing:
        raise FileNotFoundError(f"could not find {missing} under {roots}")
    return out


def _day(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format=_TS, errors="coerce").dt.normalize().astype("datetime64[ns]")


def load_offers(path: str) -> pd.DataFrame:
    cols = ["OFFER_ID", "BUYER_ID", "SKU", "WAREHOUSE", "OFFER_QUANTITY", "OFFER_PRICE",
            "OFFER_STATUS", "OFFER_CREATE_DATETIME"]
    df = pd.read_csv(path, dtype=str, usecols=cols, encoding_errors="ignore")
    out = pd.DataFrame({
        "offer_id": df.OFFER_ID, "buyer_id": df.BUYER_ID, "sku": df.SKU, "wh": df.WAREHOUSE,
        "qty": pd.to_numeric(df.OFFER_QUANTITY, errors="coerce"),
        "price": pd.to_numeric(df.OFFER_PRICE, errors="coerce"),
        "status": df.OFFER_STATUS,
        "ts": pd.to_datetime(df.OFFER_CREATE_DATETIME, format=_TS, errors="coerce").astype("datetime64[ns]"),
    })
    out = out[out.status.isin(OFFER_STATUSES) & (out.price > 0) & out.ts.notna()].copy()
    out["day"] = out.ts.dt.normalize()
    out["accepted"] = out.status.eq("ACCEPTED")
    out["qty"] = out.qty.fillna(1.0)
    return out.reset_index(drop=True)


def load_sales(path: str) -> pd.DataFrame:
    cols = ["WAREHOUSE_CODE", "STATUS", "ORDER_TYPE", "DATE", "PRODUCT_ID", "UNIT_PRICE",
            "SALES_QTY", "QUANTITY_CANCELLED"]
    df = pd.read_csv(path, dtype=str, usecols=cols, encoding_errors="ignore")
    units = (pd.to_numeric(df.SALES_QTY, errors="coerce").fillna(0)
             - pd.to_numeric(df.QUANTITY_CANCELLED, errors="coerce").fillna(0))
    out = pd.DataFrame({
        "sku": df.PRODUCT_ID, "wh": df.WAREHOUSE_CODE, "status": df.STATUS, "order_type": df.ORDER_TYPE,
        "units": units, "unit_price": pd.to_numeric(df.UNIT_PRICE, errors="coerce"), "day": _day(df.DATE),
    })
    return out[(out.status != "CANCELED") & (out.units > 0) & out.day.notna()].reset_index(drop=True)


def load_products(path: str) -> pd.DataFrame:
    cols = {"SKU": "sku", "WAREHOUSE": "wh", "MANUFACTURER": "manufacturer", "MODEL": "model",
            "CAPACITY": "capacity", "GRADE": "grade", "COLOR": "color", "CARRIER": "carrier",
            "LOCK_STATUS": "lock_status", "CATEGORY": "category"}
    df = pd.read_csv(path, dtype=str, usecols=list(cols), encoding_errors="ignore")
    out = df.rename(columns=cols)[list(cols.values())].copy()
    for c in list(cols.values())[2:]:
        out[c] = out[c].fillna("unknown").str.strip().replace("", "unknown")
    out["family"] = out.manufacturer + "|" + out.model + "|" + out.capacity + "|" + out.grade
    return out.drop_duplicates(["sku", "wh"]).reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data.py -v`
Expected: 4 passed

- [ ] **Step 5: Checkpoint** — Run: `python -m pytest -q` → all pass.

---

### Task 3: Floor-interval labels

**Files:**
- Create: `reserve_clm/labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: offers frame from `load_offers`; `config.TRAIN_END`, `config.VAL_END`
- Produces: `floor_intervals(offers) -> DataFrame[sku, wh, day, lo, hi, n_offers_day, mid]`; `assign_split(day: Series) -> np.ndarray[str]` with values `train|val|test`.

- [ ] **Step 1: Write the failing test**

`tests/test_labels.py`:
```python
import pandas as pd

from reserve_clm.labels import assign_split, floor_intervals


def _offers(rows):
    df = pd.DataFrame(rows, columns=["sku", "wh", "day", "price", "status"])
    df["day"] = pd.to_datetime(df.day)
    df["accepted"] = df.status.eq("ACCEPTED")
    return df


def test_floor_intervals():
    o = _offers([
        ("A", "W1", "2026-01-05", 90, "NOT_ACCEPTED"), ("A", "W1", "2026-01-05", 95, "COUNTERED"),
        ("A", "W1", "2026-01-05", 100, "ACCEPTED"), ("A", "W1", "2026-01-05", 105, "ACCEPTED"),
        ("A", "W1", "2026-01-06", 100, "ACCEPTED"),                                   # one-sided
        ("A", "W2", "2026-01-05", 110, "NOT_ACCEPTED"), ("A", "W2", "2026-01-05", 100, "ACCEPTED"),  # inconsistent
    ])
    f = floor_intervals(o)
    assert len(f) == 1
    r = f.iloc[0]
    assert (r.sku, r.wh, r.lo, r.hi, r.mid, r.n_offers_day) == ("A", "W1", 95, 100, 97.5, 4)


def test_assign_split():
    days = pd.Series(pd.to_datetime(["2026-01-31", "2026-02-01", "2026-02-28", "2026-03-01"]))
    assert assign_split(days).tolist() == ["train", "val", "val", "test"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve_clm.labels'`

- [ ] **Step 3: Write minimal implementation**

`reserve_clm/labels.py`:
```python
"""Assurant's implicit daily floor, read from OFFER_STATUS.

On a SKU x warehouse x day with both outcomes, the floor sits between the highest bid
Assurant did not accept (countered or rejected) and the lowest bid it accepted.
"""
import numpy as np
import pandas as pd

from .config import TRAIN_END, VAL_END

GKEYS = ["sku", "wh", "day"]


def floor_intervals(offers: pd.DataFrame) -> pd.DataFrame:
    lo = offers[~offers.accepted].groupby(GKEYS).price.max().rename("lo")
    hi = offers[offers.accepted].groupby(GKEYS).price.min().rename("hi")
    n = offers.groupby(GKEYS).size().rename("n_offers_day")
    df = pd.concat([lo, hi], axis=1, join="inner").join(n).reset_index()
    df = df[df.lo <= df.hi].copy()
    df["mid"] = (df.lo + df.hi) / 2
    return df.reset_index(drop=True)


def assign_split(day: pd.Series) -> np.ndarray:
    return np.where(day < TRAIN_END, "train", np.where(day < VAL_END, "val", "test"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_labels.py -v`
Expected: 2 passed

- [ ] **Step 5: Checkpoint** — Run: `python -m pytest -q` → all pass.

---

### Task 4: As-of features

**Files:**
- Create: `reserve_clm/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: frames from `load_offers`, `load_sales`, `load_products`, `floor_intervals`; config windows.
- Produces:
  - `lagged_join(events, targets, keys, window) -> DataFrame` (event rows from days `d-window … d-1`, with `day` = target day and `lag`)
  - `reference_prices(offers, products, targets) -> DataFrame[sku, wh, day, family, ref_source, ref_price, ref_n, ref_trend]`
  - `build_features(offers, sales, products, floors, targets) -> DataFrame` keyed by `sku, wh, day` with columns: `family, ref_source, ref_price, ref_n, ref_trend, n_offers_30, units_req_30, acc_30, bid_med_30, bid_top_30, n_offers_7, n_bidders_30, units_sold_30, units_sold_7, sale_price_med_30, days_since_last_sale, last_lo, last_hi, days_since_last_floor, manufacturer, model, capacity, grade, color, carrier, lock_status, category, bid_med_ratio_30, bid_top_ratio_30, sale_price_ratio_30, last_lo_ratio, last_hi_ratio, accept_share_30`

- [ ] **Step 1: Write the failing test**

`tests/test_features.py`:
```python
import numpy as np
import pandas as pd

from reserve_clm.features import build_features, lagged_join, reference_prices

D0 = pd.Timestamp("2026-01-10")
FAM = "Apple|iPhone 13|128GB|B"


def day(k):
    return D0 - pd.Timedelta(days=k)


def products():
    rows = [("A", "W1"), ("B", "W1")]
    p = pd.DataFrame(rows, columns=["sku", "wh"])
    for c, v in [("manufacturer", "Apple"), ("model", "iPhone 13"), ("capacity", "128GB"), ("grade", "B"),
                 ("color", "Blue"), ("carrier", "OTH"), ("lock_status", "UNLOCKED"), ("category", "Phones")]:
        p[c] = v
    p["family"] = FAM
    return p


def offers():
    rows = [  # sku, wh, day, price, accepted, buyer, qty
        ("B", "W1", day(1), 100, True, "b1", 2), ("B", "W1", day(2), 110, True, "b2", 1),
        ("B", "W1", day(3), 120, True, "b1", 1), ("B", "W1", day(3), 80, False, "b3", 5),
        ("A", "W1", day(1), 200, True, "b1", 1), ("A", "W1", D0, 999, True, "b9", 1),   # same day: excluded
        ("A", "W1", day(40), 50, True, "b1", 1),                                          # outside window
    ]
    o = pd.DataFrame(rows, columns=["sku", "wh", "day", "price", "accepted", "buyer_id", "qty"])
    o["day"] = o.day.astype("datetime64[ns]")
    return o


def sales():
    s = pd.DataFrame([("B", "W1", day(2), 3.0, 105.0, "STOCKLIST"), ("B", "W1", D0, 9.0, 1.0, "STOCKLIST"),
                      ("B", "W1", day(10), 1.0, 99.0, "AUCTION")],
                     columns=["sku", "wh", "day", "units", "unit_price", "order_type"])
    s["day"] = s.day.astype("datetime64[ns]")
    return s


def floors():
    f = pd.DataFrame([("B", "W1", day(5), 90.0, 95.0), ("B", "W1", D0, 50.0, 60.0)],
                     columns=["sku", "wh", "day", "lo", "hi"])
    f["day"] = f.day.astype("datetime64[ns]")
    return f


TARGETS = pd.DataFrame({"sku": ["A", "B"], "wh": ["W1", "W1"], "day": [D0, D0]}).astype({"day": "datetime64[ns]"})


def test_lagged_join_window_is_strictly_before():
    ev = pd.DataFrame({"sku": ["A"] * 4, "day": [day(0), day(1), day(30), day(31)], "v": [1, 2, 3, 4]})
    j = lagged_join(ev, TARGETS, ["sku"], 30)
    assert sorted(j.v.tolist()) == [2, 3] and sorted(j.lag.tolist()) == [1, 30]


def test_reference_sku_then_family():
    r = reference_prices(offers(), products(), TARGETS).set_index("sku")
    assert r.loc["B", "ref_source"] == "sku" and r.loc["B", "ref_price"] == 110 and r.loc["B", "ref_n"] == 3
    # A has 1 accepted offer in window -> family: daily medians day1 median(100,200)=150, day2 110, day3 120
    assert r.loc["A", "ref_source"] == "family" and r.loc["A", "ref_price"] == 120 and r.loc["A", "ref_n"] == 4


def test_build_features_values():
    f = build_features(offers(), sales(), products(), floors(), TARGETS).set_index("sku")
    b = f.loc["B"]
    assert b.n_offers_30 == 4 and b.n_offers_7 == 4 and b.n_bidders_30 == 3 and b.units_req_30 == 9
    assert b.acc_30 == 3 and np.isclose(b.accept_share_30, 0.75)
    assert b.bid_top_30 == 120 and np.isclose(b.bid_top_ratio_30, 120 / 110)
    assert b.units_sold_30 == 4 and b.units_sold_7 == 3 and b.days_since_last_sale == 2
    assert b.sale_price_med_30 == 105                       # auction sale ignored for price
    assert b.last_lo == 90 and b.last_hi == 95 and b.days_since_last_floor == 5
    assert b.grade == "B" and b.family == FAM
    a = f.loc["A"]
    assert a.n_offers_30 == 1 and np.isnan(a.days_since_last_sale) and np.isnan(a.last_lo_ratio)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve_clm.features'`

- [ ] **Step 3: Write minimal implementation**

`reserve_clm/features.py`:
```python
"""Features for each target SKU x warehouse x day, using only days strictly before it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MIN_REF_ACCEPTS, REF_WINDOW_DAYS, TREND_WINDOW_DAYS

KEYS = ["sku", "wh"]
GKEYS = ["sku", "wh", "day"]


def lagged_join(events: pd.DataFrame, targets: pd.DataFrame, keys: list[str], window: int) -> pd.DataFrame:
    """Event rows from days d-window ... d-1 for every target (keys, d); `day` is the target day."""
    t = targets[keys + ["day"]].dropna(subset=keys).drop_duplicates()
    parts = []
    for lag in range(1, window + 1):
        e = events.assign(day=events["day"] + pd.Timedelta(days=lag), lag=lag)
        parts.append(t.merge(e, on=keys + ["day"], how="inner"))
    return pd.concat(parts, ignore_index=True)


def _window_ref(daily: pd.DataFrame, t: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    j = lagged_join(daily, t, keys, REF_WINDOW_DAYS)
    gk = keys + ["day"]
    g = j.groupby(gk)
    f = pd.DataFrame({"ref30": g.med.median(), "ref_n": g.cnt.sum()})
    f["ref7"] = j[j.lag <= TREND_WINDOW_DAYS].groupby(gk).med.median()
    return f.reset_index()


def reference_prices(offers: pd.DataFrame, products: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """30-day median of daily median accepted prices: SKU level, else product family, else none."""
    fam = products[KEYS + ["family"]]
    acc = offers[offers.accepted].merge(fam, on=KEYS, how="left")
    sku_daily = acc.groupby(["sku", "day"]).price.agg(med="median", cnt="size").reset_index()
    fam_daily = (acc.dropna(subset=["family"]).groupby(["family", "day"]).price
                 .agg(med="median", cnt="size").reset_index())
    tt = targets[GKEYS].drop_duplicates().merge(fam, on=KEYS, how="left")
    s = _window_ref(sku_daily, tt, ["sku"]).rename(columns={"ref30": "ref30_sku", "ref_n": "ref_n_sku", "ref7": "ref7_sku"})
    f = _window_ref(fam_daily, tt, ["family"]).rename(columns={"ref30": "ref30_fam", "ref_n": "ref_n_fam", "ref7": "ref7_fam"})
    out = tt.merge(s, on=["sku", "day"], how="left").merge(f, on=["family", "day"], how="left")
    use_sku = (out.ref_n_sku.fillna(0) >= MIN_REF_ACCEPTS).to_numpy()
    use_fam = ~use_sku & (out.ref_n_fam.fillna(0) >= MIN_REF_ACCEPTS).to_numpy()
    out["ref_source"] = np.where(use_sku, "sku", np.where(use_fam, "family", "none"))
    out["ref_price"] = np.where(use_sku, out.ref30_sku, np.where(use_fam, out.ref30_fam, np.nan))
    out["ref_n"] = np.where(use_sku, out.ref_n_sku, np.where(use_fam, out.ref_n_fam, 0.0))
    ref7 = np.where(use_sku, out.ref7_sku, np.where(use_fam, out.ref7_fam, np.nan))
    out["ref_trend"] = ref7 / out["ref_price"]
    return out[GKEYS + ["family", "ref_source", "ref_price", "ref_n", "ref_trend"]]


def bid_features(offers: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    daily = offers.groupby(GKEYS).agg(n=("price", "size"), qty=("qty", "sum"), acc=("accepted", "sum"),
                                      med=("price", "median"), top=("price", "max")).reset_index()
    j = lagged_join(daily, t, KEYS, REF_WINDOW_DAYS)
    g = j.groupby(GKEYS)
    f = pd.DataFrame({"n_offers_30": g.n.sum(), "units_req_30": g.qty.sum(), "acc_30": g.acc.sum(),
                      "bid_med_30": g.med.median(), "bid_top_30": g.top.max()})
    f["n_offers_7"] = j[j.lag <= TREND_WINDOW_DAYS].groupby(GKEYS).n.sum()
    bidders = offers[GKEYS + ["buyer_id"]].drop_duplicates()
    f["n_bidders_30"] = lagged_join(bidders, t, KEYS, REF_WINDOW_DAYS).groupby(GKEYS).buyer_id.nunique()
    out = t[GKEYS].merge(f.reset_index(), on=GKEYS, how="left")
    counts = ["n_offers_30", "units_req_30", "acc_30", "n_offers_7", "n_bidders_30"]
    out[counts] = out[counts].fillna(0)
    return out


def _last_before(events: pd.DataFrame, t: pd.DataFrame, name: str) -> pd.DataFrame:
    """Most recent event strictly before each target day (same sku, wh); event day renamed to `name`."""
    r = events.rename(columns={"day": name}).dropna(subset=[name]).sort_values(name)
    left = t[GKEYS].sort_values("day")
    return pd.merge_asof(left, r, left_on="day", right_on=name, by=KEYS,
                         allow_exact_matches=False, direction="backward")


def sales_features(sales: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    daily = sales.groupby(GKEYS, as_index=False).units.sum()
    j = lagged_join(daily, t, KEYS, REF_WINDOW_DAYS)
    f = j.groupby(GKEYS).units.sum().rename("units_sold_30").to_frame()
    f["units_sold_7"] = j[j.lag <= TREND_WINDOW_DAYS].groupby(GKEYS).units.sum()
    stock = sales[sales.order_type == "STOCKLIST"].groupby(GKEYS, as_index=False).unit_price.median()
    f["sale_price_med_30"] = lagged_join(stock, t, KEYS, REF_WINDOW_DAYS).groupby(GKEYS).unit_price.median()
    out = t[GKEYS].merge(f.reset_index(), on=GKEYS, how="left")
    out[["units_sold_30", "units_sold_7"]] = out[["units_sold_30", "units_sold_7"]].fillna(0)
    last = _last_before(daily[GKEYS], t, "last_sale_day")
    out = out.merge(last, on=GKEYS, how="left")
    out["days_since_last_sale"] = (out.day - out.last_sale_day).dt.days
    return out.drop(columns=["last_sale_day"])


def last_floor_features(floors: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    ev = floors[GKEYS + ["lo", "hi"]].rename(columns={"lo": "last_lo", "hi": "last_hi"})
    m = _last_before(ev, t, "last_floor_day")
    m["days_since_last_floor"] = (m.day - m.last_floor_day).dt.days
    return m.drop(columns=["last_floor_day"])


def build_features(offers, sales, products, floors, targets) -> pd.DataFrame:
    t = targets[GKEYS].drop_duplicates().reset_index(drop=True)
    out = t.merge(reference_prices(offers, products, t), on=GKEYS, how="left")
    out = out.merge(bid_features(offers, t), on=GKEYS, how="left")
    out = out.merge(sales_features(sales, t), on=GKEYS, how="left")
    out = out.merge(last_floor_features(floors, t), on=GKEYS, how="left")
    out = out.merge(products.drop(columns=["family"]), on=KEYS, how="left")
    for src, dst in [("bid_med_30", "bid_med_ratio_30"), ("bid_top_30", "bid_top_ratio_30"),
                     ("sale_price_med_30", "sale_price_ratio_30"), ("last_lo", "last_lo_ratio"),
                     ("last_hi", "last_hi_ratio")]:
        out[dst] = out[src] / out["ref_price"]
    out["accept_share_30"] = out.acc_30 / out.n_offers_30.where(out.n_offers_30 > 0)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_features.py -v`
Expected: 3 passed

- [ ] **Step 5: Checkpoint** — Run: `python -m pytest -q` → all pass.

---

### Task 5: State text and band descriptions

**Files:**
- Create: `reserve_clm/text.py`
- Test: `tests/test_text.py`

**Interfaces:**
- Consumes: a feature row (any object with the `build_features` column names as attributes, e.g. a `DataFrame.itertuples()` row, plus `wh`, `day`); `bands.BAND_KEYS`, `bands.EDGES`, `bands.N_BANDS`
- Produces: `QUESTION: str`; `state_context(row) -> str`; `state_text(row) -> str` (context + blank line + QUESTION); `band_descriptions() -> dict[str, str]` keyed by `BAND_KEYS`.

- [ ] **Step 1: Write the failing test**

`tests/test_text.py`:
```python
from types import SimpleNamespace

import numpy as np
import pandas as pd

from reserve_clm.bands import BAND_KEYS
from reserve_clm.text import QUESTION, band_descriptions, state_text


def row(**kw):
    base = dict(manufacturer="Apple", model="iPhone 13", capacity="128GB", grade="B", color="Blue",
                carrier="OTH", lock_status="UNLOCKED", wh="W23", ref_price=412.4, ref_source="sku", ref_n=17.0,
                ref_trend=0.95, n_offers_7=11.0, n_offers_30=40.0, n_bidders_30=5.0, units_req_30=60.0,
                bid_med_ratio_30=0.94, bid_top_ratio_30=0.99, accept_share_30=0.55, units_sold_7=23.0,
                units_sold_30=80.0, days_since_last_sale=1.0, sale_price_ratio_30=0.97, last_lo_ratio=0.95,
                last_hi_ratio=0.97, days_since_last_floor=3.0, day=pd.Timestamp("2026-03-03"))
    base.update(kw)
    return SimpleNamespace(**base)


def test_state_text_layout_and_content():
    t = state_text(row())
    assert t.endswith("\n\n" + QUESTION)
    assert "Apple iPhone 13, 128GB, grade B" in t and "$412" in t and "falling (-5.0%)" in t
    assert "11 (high demand)" in t and "0.94x market" in t and "55%" in t
    assert "Previous reserve between 0.95x and 0.97x market, 3 days ago." in t and "Tuesday" in t
    assert "nan" not in t.lower()


def test_state_text_missing_values():
    t = state_text(row(ref_source="family", ref_trend=np.nan, bid_med_ratio_30=np.nan, accept_share_30=np.nan,
                       days_since_last_sale=np.nan, last_lo_ratio=np.nan, n_offers_7=0.0))
    assert "similar products" in t and "No previous reserve observed" in t and "No sale recorded" in t
    assert "0 (none demand)" in t and "nan" not in t.lower()


def test_band_descriptions():
    d = band_descriptions()
    assert list(d) == BAND_KEYS and len(set(d.values())) == 14
    assert "below 0.80x" in d["b00"] and "1.10x" in d["b13"]
    assert "between 0.950x and 0.975x" in d["b07"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_text.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve_clm.text'`

- [ ] **Step 3: Write minimal implementation**

`reserve_clm/text.py`:
```python
"""What CLM reads: the state text (one SKU x warehouse x day) and the 14 band candidates."""
from __future__ import annotations

import math

import pandas as pd

from .bands import BAND_KEYS, EDGES, N_BANDS

QUESTION = ("Which reserve (minimum) price band should be set today for this SKU at this warehouse? "
            "Bands are multiples of the SKU's recent market price.")


def _int(x) -> int:
    return 0 if pd.isna(x) else int(x)


def _ratio(x) -> str:
    return "unknown" if pd.isna(x) else f"{x:.2f}x market"


def _pct(x) -> str:
    return "unknown" if pd.isna(x) else f"{x:.0%}"


def _trend(x) -> str:
    if pd.isna(x):
        return "unknown"
    pct = (x - 1) * 100
    word = "rising" if pct > 2 else "falling" if pct < -2 else "flat"
    return f"{word} ({pct:+.1f}%)"


def _demand(n: int) -> str:
    return "none" if n == 0 else "low" if n < 3 else "moderate" if n < 10 else "high"


def _velocity(u: int) -> str:
    return "no sales" if u == 0 else "slow" if u < 5 else "steady" if u < 30 else "fast"


def state_context(r) -> str:
    source = "this SKU" if r.ref_source == "sku" else "similar products (same model, capacity and grade)"
    last_sale = ("No sale recorded before today" if pd.isna(r.days_since_last_sale)
                 else f"Last sale {_int(r.days_since_last_sale)} days ago")
    if pd.isna(r.last_lo_ratio) or pd.isna(r.last_hi_ratio):
        last_floor = "No previous reserve observed for this SKU at this warehouse."
    else:
        last_floor = (f"Previous reserve between {r.last_lo_ratio:.2f}x and {r.last_hi_ratio:.2f}x market, "
                      f"{_int(r.days_since_last_floor)} days ago.")
    n7 = _int(r.n_offers_7)
    sold30 = _int(r.units_sold_30)
    return "\n".join([
        f"Product: {r.manufacturer} {r.model}, {r.capacity}, grade {r.grade}, colour {r.color}, "
        f"carrier {r.carrier}, {r.lock_status}. Warehouse {r.wh}.",
        f"Market price: ${r.ref_price:.0f} from {source}, last 30 days, {_int(r.ref_n)} accepted offers. "
        f"Trend last 7 days vs 30 days: {_trend(r.ref_trend)}.",
        f"Bids last 7 days: {n7} ({_demand(n7)} demand). Bids last 30 days: {_int(r.n_offers_30)} "
        f"from {_int(r.n_bidders_30)} buyers for {_int(r.units_req_30)} units.",
        f"Median bid {_ratio(r.bid_med_ratio_30)}, top bid {_ratio(r.bid_top_ratio_30)}. "
        f"Share of recent bids accepted: {_pct(r.accept_share_30)}.",
        f"Units sold last 7 days: {_int(r.units_sold_7)}, last 30 days: {sold30} ({_velocity(sold30)}). "
        f"{last_sale}. Median sale price {_ratio(r.sale_price_ratio_30)}.",
        last_floor,
        f"Day: {pd.Timestamp(r.day):%A}.",
    ])


def state_text(r) -> str:
    """Context first, question last: the layout CLM's heads were trained on (clm.schema.state_text)."""
    return f"{state_context(r).strip()}\n\n{QUESTION}"


_TIERS = [  # (upper bound on band midpoint, name, when to use)
    (0.875, "Heavy discount reserve", "Use when stock is slow or ageing, bids are well below market and prices are falling."),
    (0.95, "Moderate discount reserve", "Use when demand is soft, bids sit below market and sell-through is slow."),
    (1.00, "Slight discount reserve", "Use when demand is steady, several buyers bid close to market and prices are flat."),
    (1.05, "At-market reserve", "Use when demand is healthy, bids reach market price and stock sells steadily."),
    (math.inf, "Premium reserve", "Use when stock is scarce, competing bids exceed market and prices are rising."),
]


def band_descriptions() -> dict[str, str]:
    out = {}
    for i, key in enumerate(BAND_KEYS):
        lo, hi = EDGES[i], EDGES[i + 1]
        if i == 0:
            out[key] = ("Deep clearance reserve: below 0.80x the market price. Use for stale stock with very "
                        "weak, low bids and falling prices.")
        elif i == N_BANDS - 1:
            out[key] = ("Strong premium reserve: at or above 1.10x the market price. Use for scarce stock with "
                        "strong competing bids well above market and rising prices.")
        else:
            name, hint = next((n, h) for bound, n, h in _TIERS if (lo + hi) / 2 < bound)
            out[key] = f"{name}: between {lo:.3f}x and {hi:.3f}x the market price. {hint}"
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_text.py -v`
Expected: 3 passed

- [ ] **Step 5: Checkpoint** — Run: `python -m pytest -q` → all pass.

---

### Task 6: Baselines and metric primitives

**Files:**
- Create: `reserve_clm/baselines.py`, `reserve_clm/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `bands.N_BANDS`
- Produces:
  - `baselines.carry_forward_ratio(df, fallback: float) -> np.ndarray`; `baselines.fixed_ratio(n: int, ratio: float) -> np.ndarray`; `baselines.majority_band(true_band: np.ndarray) -> int`
  - `evaluate.decode_bands(probs, how="median"|"argmax") -> np.ndarray[int]`
  - `evaluate.interval_metrics(price, lo, hi, mid) -> dict` keys `n, interval_hit_rate, too_high_rate, too_low_rate, mape_vs_mid, median_ape_vs_mid, bias_vs_mid`
  - `evaluate.band_metrics(pred_band, true_band) -> dict` keys `band_exact, band_within_1, band_mae`
  - `evaluate.replay(offers, groups[sku, wh, day, reserve]) -> dict` keys `n_groups, n_offers, agreement, lost_sales_rate, underpricing_rate, pred_accept_rate, actual_accept_rate, accepted_value_pred, accepted_value_actual`
  - `evaluate.calibration_table(conf, hit, n_bins=5) -> DataFrame[bin, n, mean_confidence, interval_hit_rate]`
  - `evaluate.band_ece(probs, true_band, n_bins=10) -> float`

- [ ] **Step 1: Write the failing test**

`tests/test_evaluate.py`:
```python
import numpy as np
import pandas as pd

from reserve_clm.baselines import carry_forward_ratio, fixed_ratio, majority_band
from reserve_clm.evaluate import (band_ece, band_metrics, calibration_table, decode_bands,
                                  interval_metrics, replay)


def test_decode_median_vs_argmax():
    p = np.zeros((1, 14)); p[0, [2, 9, 10]] = [0.4, 0.3, 0.3]
    assert decode_bands(p, "argmax")[0] == 2 and decode_bands(p)[0] == 9


def test_interval_metrics():
    m = interval_metrics([100, 120, 80, 95], [90, 90, 90, 95], [110, 110, 110, 95], [100, 100, 100, 95])
    assert m["n"] == 4 and m["interval_hit_rate"] == 0.5
    assert m["too_high_rate"] == 0.25 and m["too_low_rate"] == 0.25
    assert np.isclose(m["mape_vs_mid"], 0.1) and np.isclose(m["bias_vs_mid"], 0.0)
    assert np.isnan(interval_metrics([], [], [], [])["interval_hit_rate"])


def test_band_metrics():
    m = band_metrics(np.array([3, 4, 9]), np.array([3, 5, 5]))
    assert np.isclose(m["band_exact"], 1 / 3) and np.isclose(m["band_within_1"], 2 / 3) and np.isclose(m["band_mae"], 5 / 3)


def test_replay_counts():
    d = pd.Timestamp("2026-03-02")
    offers = pd.DataFrame({"sku": ["A"] * 4, "wh": ["W"] * 4, "day": [d] * 4, "price": [90.0, 100, 110, 120],
                           "qty": [1.0, 1, 2, 1], "accepted": [False, True, False, True]})
    groups = pd.DataFrame({"sku": ["A"], "wh": ["W"], "day": [d], "reserve": [105.0]})
    r = replay(offers, groups)
    assert r["n_offers"] == 4 and r["agreement"] == 0.5
    assert r["lost_sales_rate"] == 0.25 and r["underpricing_rate"] == 0.25
    assert r["accepted_value_pred"] == 340 and r["accepted_value_actual"] == 220


def test_calibration_and_ece():
    t = calibration_table(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]), n_bins=2)
    assert t.n.tolist() == [2, 2] and t.interval_hit_rate.tolist() == [0.0, 1.0]
    p = np.zeros((2, 14)); p[:, 0] = 1.0
    assert band_ece(p, np.array([0, 0])) == 0.0 and band_ece(p, np.array([1, 1])) == 1.0


def test_baselines():
    df = pd.DataFrame({"last_lo_ratio": [0.9, np.nan], "last_hi_ratio": [1.0, np.nan]})
    np.testing.assert_allclose(carry_forward_ratio(df, 0.97), [0.95, 0.97])
    assert fixed_ratio(2, 0.9).tolist() == [0.9, 0.9]
    assert majority_band(np.array([3, 3, 5])) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve_clm.baselines'`

- [ ] **Step 3: Write minimal implementation**

`reserve_clm/baselines.py`:
```python
"""Rule baselines the CLM has to beat (no training)."""
import numpy as np
import pandas as pd

from .bands import N_BANDS


def carry_forward_ratio(df: pd.DataFrame, fallback: float) -> np.ndarray:
    """B0: midpoint of Assurant's previous floor at this SKU x warehouse, else the fixed ratio."""
    r = (df["last_lo_ratio"] + df["last_hi_ratio"]) / 2
    return r.fillna(fallback).to_numpy(dtype=float)


def fixed_ratio(n: int, ratio: float) -> np.ndarray:
    """B1/B2: one ratio for every row."""
    return np.full(n, float(ratio))


def majority_band(true_band: np.ndarray) -> int:
    return int(np.bincount(np.asarray(true_band, dtype=int), minlength=N_BANDS).argmax())
```

`reserve_clm/evaluate.py`:
```python
"""Metric primitives: interval hit, price error, band accuracy, offer replay, calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd

NAN = float("nan")


def decode_bands(probs: np.ndarray, how: str = "median") -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if how == "argmax":
        return probs.argmax(1)
    return (probs.cumsum(1) >= 0.5).argmax(1)


def interval_metrics(price, lo, hi, mid) -> dict:
    price, lo, hi, mid = (np.asarray(x, dtype=float) for x in (price, lo, hi, mid))
    keys = ["interval_hit_rate", "too_high_rate", "too_low_rate", "mape_vs_mid", "median_ape_vs_mid", "bias_vs_mid"]
    if len(price) == 0:
        return {"n": 0, **{k: NAN for k in keys}}
    rel = (price - mid) / mid
    return {"n": int(len(price)),
            "interval_hit_rate": float(((price >= lo) & (price <= hi)).mean()),
            "too_high_rate": float((price > hi).mean()), "too_low_rate": float((price < lo).mean()),
            "mape_vs_mid": float(np.abs(rel).mean()), "median_ape_vs_mid": float(np.median(np.abs(rel))),
            "bias_vs_mid": float(rel.mean())}


def band_metrics(pred_band, true_band) -> dict:
    p, t = np.asarray(pred_band, dtype=int), np.asarray(true_band, dtype=int)
    if len(p) == 0:
        return {"band_exact": NAN, "band_within_1": NAN, "band_mae": NAN}
    d = np.abs(p - t)
    return {"band_exact": float((d == 0).mean()), "band_within_1": float((d <= 1).mean()), "band_mae": float(d.mean())}


def replay(offers: pd.DataFrame, groups: pd.DataFrame) -> dict:
    """Accept every offer whose bid >= the group's reserve; compare with Assurant's OFFER_STATUS."""
    m = offers.merge(groups[["sku", "wh", "day", "reserve"]], on=["sku", "wh", "day"], how="inner")
    out = {"n_groups": int(len(groups)), "n_offers": int(len(m))}
    if m.empty:
        return out
    pred = (m.price >= m.reserve).to_numpy()
    act = m.accepted.to_numpy(dtype=bool)
    value = (m.price * m.qty).to_numpy()
    out.update(agreement=float((pred == act).mean()), lost_sales_rate=float((~pred & act).mean()),
               underpricing_rate=float((pred & ~act).mean()), pred_accept_rate=float(pred.mean()),
               actual_accept_rate=float(act.mean()), accepted_value_pred=float(value[pred].sum()),
               accepted_value_actual=float(value[act].sum()))
    return out


def calibration_table(conf, hit, n_bins: int = 5) -> pd.DataFrame:
    conf, hit = np.asarray(conf, dtype=float), np.asarray(hit, dtype=float)
    if len(conf) == 0:
        return pd.DataFrame(columns=["bin", "n", "mean_confidence", "interval_hit_rate"])
    d = pd.DataFrame({"conf": conf, "hit": hit, "bin": pd.qcut(conf, n_bins, duplicates="drop")})
    g = d.groupby("bin", observed=True)
    t = pd.DataFrame({"n": g.size(), "mean_confidence": g.conf.mean(), "interval_hit_rate": g.hit.mean()})
    t = t.reset_index()
    t["bin"] = t["bin"].astype(str)
    return t


def band_ece(probs, true_band, n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float)
    if len(probs) == 0:
        return NAN
    conf = probs.max(1)
    correct = probs.argmax(1) == np.asarray(true_band, dtype=int)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: 6 passed

- [ ] **Step 5: Checkpoint** — Run: `python -m pytest -q` → all pass.

---

### Task 7: CLM head training

**Files:**
- Create: `reserve_clm/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `clm.heads.make_head(width, depth, proj, activation, layernorm, residual, hidden)` from the pinned CLM package.
- Produces:
  - `TrainConfig(epochs=60, batch=256, lr=5e-4, weight_decay=0.0, patience=10, seed=1234, width=1536, depth=3, proj=512)`
  - `HeadModel(sh, ah, logit_scale)` with `.logits(X, C) -> Tensor[n, k]`
  - `build_model(hidden, cfg, init_ckpt=None) -> (HeadModel, head_cfg: dict, warm: bool)`
  - `soft_infonce(logits, targets) -> Tensor`
  - `TrainResult(model, head_cfg, warm_start, best_epoch, best_score, history)`
  - `train_heads(X_tr, T_tr, C, X_val, val_score, cfg, init_ckpt=None, device=None, log=print) -> TrainResult` — `val_score(probs: np.ndarray) -> float`, higher is better
  - `predict_proba(model, X, C, batch=4096) -> np.ndarray[n, k]`
  - `save_checkpoint(result, path) -> None` (CLM-compatible checkpoint dict)

- [ ] **Step 1: Write the failing test**

`tests/test_train.py`:
```python
import numpy as np
import torch

from reserve_clm.train import TrainConfig, build_model, predict_proba, save_checkpoint, soft_infonce, train_heads


def _data(n, rng, centers):
    y = rng.integers(0, len(centers), n)
    x = centers[y] + 0.3 * rng.standard_normal((n, centers.shape[1]))
    return x.astype(np.float32), y


def test_soft_infonce_prefers_correct_logits():
    t = torch.eye(3)
    good = soft_infonce(10 * torch.eye(3), t)
    bad = soft_infonce(-10 * torch.eye(3), t)
    assert good < bad


def test_train_heads_learns_separable_bands():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((14, 32)).astype(np.float32)
    x_tr, y_tr = _data(1400, rng, centers)
    x_va, y_va = _data(280, rng, centers)
    t_tr = np.eye(14, dtype=np.float32)[y_tr]
    cfg = TrainConfig(epochs=40, batch=128, lr=3e-3, patience=40, width=64, depth=3, proj=32)
    res = train_heads(x_tr, t_tr, centers, x_va, lambda p: float((p.argmax(1) == y_va).mean()), cfg, log=lambda *_: None)
    p = predict_proba(res.model, x_va, centers)
    assert p.shape == (280, 14) and np.allclose(p.sum(1), 1, atol=1e-5)
    assert (p.argmax(1) == y_va).mean() > 0.9 and res.best_score > 0.9 and not res.warm_start


def test_checkpoint_round_trip_warm_starts(tmp_path):
    rng = np.random.default_rng(1)
    centers = rng.standard_normal((14, 16)).astype(np.float32)
    x, y = _data(200, rng, centers)
    cfg = TrainConfig(epochs=2, batch=64, width=32, depth=3, proj=8)
    res = train_heads(x, np.eye(14, dtype=np.float32)[y], centers, x, lambda p: 0.0, cfg, log=lambda *_: None)
    path = tmp_path / "head.pt"
    save_checkpoint(res, str(path))
    model, head_cfg, warm = build_model(16, TrainConfig(width=999), init_ckpt=str(path))
    assert warm and head_cfg["width"] == 32 and head_cfg["proj"] == 8
    np.testing.assert_allclose(predict_proba(model, x, centers), predict_proba(res.model, x, centers), atol=1e-5)
    _, _, warm2 = build_model(24, cfg, init_ckpt=str(path))    # width mismatch -> fresh heads
    assert not warm2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve_clm.train'`

- [ ] **Step 3: Write minimal implementation**

`reserve_clm/train.py`:
```python
"""Fine-tune CLM's projection heads on cached encoder embeddings.

Same objective as CLM's `train/finetune.py --task choice --loss infonce --targets soft`:
bidirectional InfoNCE between states and the band texts, against a soft target
distribution. The 8B encoder never changes; only the two small heads train.
"""
from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainConfig:
    epochs: int = 60
    batch: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.0
    patience: int = 10
    seed: int = 1234
    width: int = 1536      # used only when there is no compatible warm-start checkpoint
    depth: int = 3
    proj: int = 512


class HeadModel(nn.Module):
    def __init__(self, sh: nn.Module, ah: nn.Module, logit_scale: nn.Parameter):
        super().__init__()
        self.sh, self.ah, self.logit_scale = sh, ah, logit_scale

    def logits(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        zq = F.normalize(self.sh(x), dim=-1)
        zc = F.normalize(self.ah(c), dim=-1)
        return self.logit_scale.exp().clamp(max=100.0) * zq @ zc.t()


def build_model(hidden: int, cfg: TrainConfig, init_ckpt: str | None = None):
    from clm.heads import make_head
    head_cfg = dict(width=cfg.width, depth=cfg.depth, proj=cfg.proj, activation="gelu",
                    layernorm=True, residual=False, hidden=hidden)
    ck = torch.load(init_ckpt, map_location="cpu", weights_only=False) if init_ckpt else None
    warm = False
    if ck is not None and ck["state_head"]["inp.weight"].shape[1] == hidden:
        c0 = ck.get("cfg", {})
        head_cfg.update(width=c0["width"], depth=c0["depth"],
                        proj=c0.get("projection_dim", ck.get("projection_dim", cfg.proj)),
                        activation=c0.get("activation", "gelu"), residual=c0.get("residual", False),
                        layernorm=any(k.startswith("norms.") for k in ck["state_head"]))
        warm = True
    sh, ah = make_head(**head_cfg), make_head(**head_cfg)
    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
    if warm:
        sh.load_state_dict(ck["state_head"])
        ah.load_state_dict(ck["action_head"])
        with torch.no_grad():
            logit_scale.copy_(torch.as_tensor(ck["logit_scale"]).float())
    return HeadModel(sh, ah, logit_scale), head_cfg, warm


def soft_infonce(lg: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """States -> bands against the soft targets, bands -> states against the states that hold them."""
    fwd = -(t * F.log_softmax(lg, 1)).sum(1).mean()
    w = t.t()
    keep = w.sum(1) > 0
    w = w[keep] / w[keep].sum(1, keepdim=True)
    bwd = -(w * F.log_softmax(lg.t()[keep], 1)).sum(1).mean()
    return (fwd + bwd) / 2


@torch.no_grad()
def predict_proba(model: HeadModel, x, c, batch: int = 4096) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    ct = torch.as_tensor(np.asarray(c), dtype=torch.float32, device=device)
    out = [F.softmax(model.logits(torch.as_tensor(np.asarray(x[i:i + batch]), dtype=torch.float32,
                                                  device=device), ct), 1).cpu().numpy()
           for i in range(0, len(x), batch)]
    return np.concatenate(out) if out else np.zeros((0, len(c)), dtype=np.float32)


@dataclass
class TrainResult:
    model: HeadModel
    head_cfg: dict
    warm_start: bool
    best_epoch: int
    best_score: float
    history: list = field(default_factory=list)


def train_heads(x_tr, t_tr, c, x_val, val_score, cfg: TrainConfig, init_ckpt: str | None = None,
                device: str | None = None, log=print) -> TrainResult:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    model, head_cfg, warm = build_model(int(np.asarray(x_tr).shape[1]), cfg, init_ckpt)
    model.to(device)
    x = torch.as_tensor(np.asarray(x_tr), dtype=torch.float32, device=device)
    t = torch.as_tensor(np.asarray(t_tr), dtype=torch.float32, device=device)
    ct = torch.as_tensor(np.asarray(c), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = math.ceil(len(x) / cfg.batch) * cfg.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(1, steps),
                                                pct_start=0.1, anneal_strategy="cos")
    gen = torch.Generator().manual_seed(cfg.seed)
    best = val_score(predict_proba(model, x_val, c))
    best_state, best_epoch, stale = copy.deepcopy(model.state_dict()), 0, 0
    history = [{"epoch": 0, "train_loss": None, "val_score": best}]
    log(f"[train] seed {cfg.seed} warm_start={warm} epoch 0 val {best:.4f}")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(x), generator=gen).split(cfg.batch):
            idx = idx.to(device)
            loss = soft_infonce(model.logits(x[idx], ct), t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        score = val_score(predict_proba(model, x_val, c))
        history.append({"epoch": epoch, "train_loss": total / len(x), "val_score": score})
        log(f"[train] seed {cfg.seed} epoch {epoch} loss {total / len(x):.4f} val {score:.4f}")
        if score > best + 1e-6:
            best, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg.patience:
                log(f"[train] early stop at epoch {epoch} (best {best_epoch})")
                break
    model.load_state_dict(best_state)
    return TrainResult(model, head_cfg, warm, best_epoch, float(best), history)


def save_checkpoint(res: TrainResult, path: str) -> None:
    """Same layout as CLM checkpoints, so clm.heads.HeadPair / clm.Engine can load it."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    c = res.head_cfg
    torch.save({"state_head": res.model.sh.state_dict(), "action_head": res.model.ah.state_dict(),
                "logit_scale": res.model.logit_scale.detach().cpu(),
                "cfg": {"width": c["width"], "depth": c["depth"], "projection_dim": c["proj"],
                        "activation": c["activation"], "layernorm": c["layernorm"], "residual": c["residual"],
                        "hidden_size": c["hidden"], "task": "reserve_band"},
                "epoch": res.best_epoch, "metrics": {"val_interval_hit_rate": res.best_score}}, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train.py -v`
Expected: 3 passed

- [ ] **Step 5: Checkpoint** — Run: `python -m pytest -q` → all pass.

---

### Task 8: Qwen embedder with cache

**Files:**
- Create: `reserve_clm/embed.py`
- Test: `tests/test_embed.py`

**Interfaces:**
- Produces: `QwenEmbedder(model_name, max_len=2048, batch_size=16, cache_path=None, log=print)` with `.hidden: int`, `.embed(texts: list[str]) -> np.ndarray[float32, n×hidden]` (L2-normalised last-token states, cached by sha1 of the text), `.close()` (frees the model).

- [ ] **Step 1: Write the failing test**

`tests/test_embed.py`:
```python
import numpy as np
import pytest

from reserve_clm.embed import QwenEmbedder

pytestmark = pytest.mark.slow
MODEL = "Qwen/Qwen3-0.6B"


def test_embed_shapes_norms_and_cache(tmp_path):
    cache = tmp_path / "c.npz"
    e = QwenEmbedder(MODEL, max_len=64, batch_size=2, cache_path=str(cache), log=lambda *_: None)
    texts = ["short text", "a somewhat longer text about phones and prices", "short text"]
    v = e.embed(texts)
    assert v.shape == (3, e.hidden) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-2)
    np.testing.assert_array_equal(v[0], v[2])
    single = e.embed([texts[1]])                      # padding in a batch must not change the vector
    np.testing.assert_allclose(single[0], v[1], atol=2e-3)
    e.close()
    e2 = QwenEmbedder.__new__(QwenEmbedder)           # cache reload without touching the model
    e2._load_cache(str(cache))
    assert len(e2.cache) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_embed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve_clm.embed'`

- [ ] **Step 3: Write minimal implementation**

`reserve_clm/embed.py`:
```python
"""Qwen3 last-token embeddings (the reader CLM's heads sit on), cached on disk by text hash.

Recipe matches CLM's train/embed_utils.py: tokens without special tokens, keep the last
max_len - 1, last-token hidden state, L2-normalised. Batches are length-sorted and right
padded; causal attention means real tokens never see the padding.
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np


def _key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


class QwenEmbedder:
    def __init__(self, model_name: str, max_len: int = 2048, batch_size: int = 16,
                 cache_path: str | None = None, log=print):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.log = torch, log
        self.max_len, self.batch_size, self.cache_path = max_len, batch_size, cache_path
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, device_map="auto").eval()
        else:
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).eval()
        self.hidden = int(self.model.config.hidden_size)
        self.device = self.model.get_input_embeddings().weight.device
        pad = self.tok.pad_token_id
        self.pad_id = pad if pad is not None else (self.tok.eos_token_id or 0)
        self._load_cache(cache_path)

    def _load_cache(self, path: str | None) -> None:
        self.cache = {}
        if path and os.path.exists(path):
            z = np.load(path)
            self.cache = dict(zip(z["keys"].tolist(), z["vecs"]))

    def _save_cache(self) -> None:
        if not self.cache_path or not self.cache:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        keys = list(self.cache)
        np.savez(self.cache_path, keys=np.array(keys), vecs=np.stack([self.cache[k] for k in keys]))

    def _ids(self, text: str) -> list[int]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = self.tok(" ", add_special_tokens=False)["input_ids"]
        return ids[-(self.max_len - 1):]

    def embed(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        keys = [_key(t) for t in texts]
        todo = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in self.cache))
        if todo:
            ids = [self._ids(t) for t in todo]
            order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
            started, done = time.time(), 0
            with torch.no_grad():
                for b, s in enumerate(range(0, len(order), self.batch_size)):
                    chunk = order[s:s + self.batch_size]
                    width = max(len(ids[i]) for i in chunk)
                    x = torch.full((len(chunk), width), self.pad_id, dtype=torch.long)
                    m = torch.zeros((len(chunk), width), dtype=torch.long)
                    for r, i in enumerate(chunk):
                        x[r, :len(ids[i])] = torch.tensor(ids[i])
                        m[r, :len(ids[i])] = 1
                    h = self.model(input_ids=x.to(self.device), attention_mask=m.to(self.device)).last_hidden_state
                    last = torch.tensor([len(ids[i]) - 1 for i in chunk], device=h.device)
                    v = h[torch.arange(len(chunk), device=h.device), last].float()
                    v = torch.nn.functional.normalize(v, dim=-1).cpu().numpy().astype(np.float16)
                    for i, vec in zip(chunk, v):
                        self.cache[_key(todo[i])] = vec
                    done += len(chunk)
                    if b % 50 == 0 or done == len(todo):
                        rate = done / max(1e-9, time.time() - started)
                        self.log(f"[embed] {done}/{len(todo)} new texts ({rate:.1f}/s)")
            self._save_cache()
        if not keys:
            return np.zeros((0, self.hidden), dtype=np.float32)
        return np.stack([self.cache[k] for k in keys]).astype(np.float32)

    def close(self) -> None:
        """Release the 8B model (GPU memory) once all texts are embedded."""
        self.model = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_embed.py -v`
Expected: 1 passed (first run downloads Qwen3-0.6B, ~1.2 GB)

- [ ] **Step 5: Checkpoint** — Run: `python -m pytest -q` → all pass.

---

### Task 9: Report tables and the pipeline (with real-data integration test)

**Files:**
- Create: `reserve_clm/report.py`, `reserve_clm/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `report.method_ratios(df, train_df, runs, split_key) -> (dict[str, np.ndarray], np.ndarray)` (ratios per method, ensemble probs)
  - `report.score_table(df, ratios) -> DataFrame`
  - `report.replay_table(replay_df, offers, train_df, runs) -> DataFrame`
  - `report.breakdowns(df, hits: dict[str, np.ndarray]) -> DataFrame`
  - `report.prediction_examples(df, probs, n=8, seed=0) -> DataFrame`
  - `pipeline.RunConfig(data_roots, work_dir="work", embed_model="Qwen/Qwen3-8B", max_len=2048, embed_batch=16, seeds=(1234, 1, 2), train=TrainConfig(), replay_max_groups=20000, init_ckpt=None, smoke_rows=None)`
  - `pipeline.Prepared(labels, replay, offers, counts)`
  - `pipeline.prepare(cfg, log=print) -> Prepared`
  - `pipeline.views(prep) -> dict[str, DataFrame]` keys `train, val, test, test_fallback`
  - `pipeline.embed_all(prep, cfg, embedder=None, log=print) -> (views, emb: dict[str, np.ndarray])` keys `train, val, test, test_fallback, replay, bands`
  - `pipeline.run_training(V, E, cfg, log=print) -> list[dict(seed, result, probs)]`
  - `pipeline.evaluate_all(prep, V, runs) -> dict[str, DataFrame]` keys `training, headline_test, headline_fallback, replay, calibration, band_ece, breakdowns, examples`
  - `pipeline.save_results(results, work_dir) -> None`

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
import numpy as np
import pandas as pd
import pytest

from conftest import DATA_DIR
from reserve_clm.bands import N_BANDS
from reserve_clm.pipeline import RunConfig, evaluate_all, prepare, run_training, views
from reserve_clm.train import TrainConfig

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not DATA_DIR.exists(), reason="real data folder not present")]


@pytest.fixture(scope="module")
def prep():
    return prepare(RunConfig(data_roots=[str(DATA_DIR)], smoke_rows=150, replay_max_groups=150), log=lambda *_: None)


def test_prepare_shapes_and_texts(prep):
    L = prep.labels
    assert set(L.split) == {"train", "val", "test"} and (L.lo <= L.hi).all()
    priced = L[L.ref_source != "none"]
    assert len(priced) > 100 and priced.text.str.endswith("market price.").all()
    assert not priced.ref_price.isna().any() and set(priced.true_band) <= set(range(N_BANDS))
    assert len(prep.replay) > 0 and (prep.replay.ref_source != "none").all()
    assert {"sku", "family"} <= set(prep.counts.columns)


def test_features_use_only_past_days(prep):
    o = prep.offers
    for r in prep.labels.sample(20, random_state=0).itertuples():
        w = o[(o.sku == r.sku) & (o.wh == r.wh) & (o.day < r.day) & (o.day >= r.day - pd.Timedelta(days=30))]
        assert r.n_offers_30 == len(w)


def test_training_and_evaluation_with_fake_embeddings(prep):
    V = views(prep)
    rng = np.random.default_rng(0)
    E = {k: rng.standard_normal((len(v), 16)).astype(np.float32) for k, v in V.items()}
    E["replay"] = rng.standard_normal((len(prep.replay), 16)).astype(np.float32)
    E["bands"] = rng.standard_normal((N_BANDS, 16)).astype(np.float32)
    cfg = RunConfig(data_roots=[], work_dir="work/test_pipeline", seeds=(1, 2),
                    train=TrainConfig(epochs=2, batch=64, width=32, depth=3, proj=8))
    runs = run_training(V, E, cfg, log=lambda *_: None)
    res = evaluate_all(prep, V, runs)
    h = res["headline_test"]
    assert {"B0 carry-forward", "B1 fixed ratio", "B2 majority band", "CLM seeds (mean)",
            "CLM ensemble (median band)", "CLM ensemble (expected ratio)"} <= set(h.method)
    assert h.interval_hit_rate.between(0, 1).all()
    assert {"reference", "method", "agreement"} <= set(res["replay"].columns)
    assert len(res["training"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve_clm.pipeline'`

- [ ] **Step 3: Write `report.py`**

`reserve_clm/report.py`:
```python
"""Comparison tables: every method scored the same way on the same rows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bands import BAND_LABELS, BAND_MIDS, band_of
from .baselines import carry_forward_ratio, fixed_ratio, majority_band
from .evaluate import band_metrics, decode_bands, interval_metrics, replay

KEYS3 = ["sku", "wh", "day"]


def method_ratios(df: pd.DataFrame, train_df: pd.DataFrame, runs: list[dict], split_key: str):
    """Reserve ratio (x reference price) per method for the rows of df."""
    fallback = float(train_df.mid_ratio.median())
    maj = majority_band(train_df.true_band.to_numpy())
    ratios = {"B0 carry-forward": carry_forward_ratio(df, fallback),
              "B1 fixed ratio": fixed_ratio(len(df), fallback),
              "B2 majority band": fixed_ratio(len(df), BAND_MIDS[maj])}
    for r in runs:
        ratios[f"CLM seed {r['seed']}"] = BAND_MIDS[decode_bands(r["probs"][split_key])]
    ens = np.mean([r["probs"][split_key] for r in runs], axis=0)
    ratios["CLM ensemble (median band)"] = BAND_MIDS[decode_bands(ens)]
    ratios["CLM ensemble (expected ratio)"] = ens @ BAND_MIDS
    return ratios, ens


def score_table(df: pd.DataFrame, ratios: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, r in ratios.items():
        m = interval_metrics(df.ref_price.to_numpy() * r, df.lo, df.hi, df.mid)
        m.update(band_metrics(band_of(r), df.true_band.to_numpy()))
        rows.append({"method": name, **m})
    t = pd.DataFrame(rows)
    seeds = t[t.method.str.startswith("CLM seed")]
    if len(seeds) > 1:
        mean = seeds.drop(columns="method").mean().to_dict()
        mean.update(method="CLM seeds (mean)", hit_rate_std=float(seeds.interval_hit_rate.std()))
        t = pd.concat([t, pd.DataFrame([mean])], ignore_index=True)
    return t


def replay_table(rep: pd.DataFrame, offers: pd.DataFrame, train_df: pd.DataFrame, runs: list[dict]) -> pd.DataFrame:
    ratios, _ = method_ratios(rep, train_df, runs, "replay")
    rows = []
    for src in ("sku", "family"):
        mask = (rep.ref_source == src).to_numpy()
        if not mask.any():
            continue
        for name, r in ratios.items():
            groups = rep.loc[mask, KEYS3].assign(reserve=rep.ref_price.to_numpy()[mask] * r[mask])
            rows.append({"reference": src, "method": name, **replay(offers, groups)})
    return pd.DataFrame(rows)


def breakdowns(df: pd.DataFrame, hits: dict[str, np.ndarray]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    top = d.manufacturer.value_counts().index[:6]
    d["brand"] = np.where(d.manufacturer.isin(top), d.manufacturer, "other")
    d["price_tier"] = pd.cut(d.ref_price, [0, 150, 350, np.inf], labels=["<$150", "$150-350", ">=$350"])
    d["bid_history"] = pd.cut(d.n_offers_30, [-1, 9, 29, np.inf], labels=["<10 bids/30d", "10-29", "30+"])
    for name, h in hits.items():
        d[name] = h
    out = []
    for dim in ("brand", "price_tier", "bid_history"):
        g = d.groupby(dim, observed=True)
        t = g[list(hits)].mean()
        t.insert(0, "n", g.size())
        t = t.reset_index().rename(columns={dim: "value"})
        t.insert(0, "dimension", dim)
        t["value"] = t["value"].astype(str)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def prediction_examples(df: pd.DataFrame, probs: np.ndarray, n: int = 8, seed: int = 0) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    idx = np.random.default_rng(seed).choice(len(df), size=min(n, len(df)), replace=False)
    rows = []
    for i in idx:
        r, p = df.iloc[i], probs[i]
        b = decode_bands(p[None])[0]
        top = np.argsort(p)[::-1][:3]
        rows.append({"sku": r.sku, "wh": r.wh, "day": r.day.date(), "market_price": round(r.ref_price, 2),
                     "assurant_floor_interval": f"{r.lo:.0f}-{r.hi:.0f}",
                     "clm_reserve": round(r.ref_price * BAND_MIDS[b], 2),
                     "top_bands": ", ".join(f"{BAND_LABELS[k]} ({p[k]:.0%})" for k in top)})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Write `pipeline.py`**

`reserve_clm/pipeline.py`:
```python
"""End-to-end steps the notebook calls: prepare -> embed -> train -> evaluate."""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .bands import BAND_KEYS, BAND_MIDS, band_of, soft_targets
from .config import VAL_END
from .data import find_data_files, load_offers, load_products, load_sales
from .evaluate import band_ece, calibration_table, decode_bands, interval_metrics
from .features import build_features
from .labels import assign_split, floor_intervals
from .report import breakdowns, method_ratios, prediction_examples, replay_table, score_table
from .text import band_descriptions, state_text
from .train import TrainConfig, predict_proba, save_checkpoint, train_heads

KEYS3 = ["sku", "wh", "day"]


@dataclass
class RunConfig:
    data_roots: list
    work_dir: str = "work"
    embed_model: str = "Qwen/Qwen3-8B"
    max_len: int = 2048
    embed_batch: int = 16
    seeds: tuple = (1234, 1, 2)
    train: TrainConfig = field(default_factory=TrainConfig)
    replay_max_groups: int = 20000
    init_ckpt: str | None = None
    smoke_rows: int | None = None        # subsample labels per split (quick runs only)


@dataclass
class Prepared:
    labels: pd.DataFrame
    replay: pd.DataFrame
    offers: pd.DataFrame
    counts: pd.DataFrame


def prepare(cfg: RunConfig, log=print) -> Prepared:
    files = find_data_files(cfg.data_roots)
    log(f"[data] {files}")
    offers = load_offers(files["offers"])
    sales = load_sales(files["sales"])
    products = load_products(files["products"])
    floors = floor_intervals(offers)
    labels = floors.assign(split=assign_split(floors.day))
    test_groups = offers.loc[offers.day >= VAL_END, KEYS3].drop_duplicates()
    replay = test_groups.sample(n=min(cfg.replay_max_groups, len(test_groups)), random_state=0)
    if cfg.smoke_rows:
        labels = pd.concat([g.sample(min(len(g), cfg.smoke_rows), random_state=0)
                            for _, g in labels.groupby("split")])
        replay = replay.head(cfg.smoke_rows)
    log(f"[data] offers {len(offers):,}  sales lines {len(sales):,}  floor days {len(floors):,}  "
        f"labels used {len(labels):,}  replay groups {len(replay):,}")
    targets = pd.concat([labels[KEYS3], replay[KEYS3]]).drop_duplicates()
    feats = build_features(offers, sales, products, floors, targets)
    labels = labels.merge(feats, on=KEYS3, how="left")
    for c in ("lo", "hi", "mid"):
        labels[f"{c}_ratio"] = labels[c] / labels.ref_price
    labels["true_band"] = band_of(labels.mid_ratio.fillna(1.0))
    priced = (labels.ref_source != "none").to_numpy()
    labels["text"] = ""
    labels.loc[priced, "text"] = [state_text(r) for r in labels[priced].itertuples(index=False)]
    replay = replay.merge(feats, on=KEYS3, how="left")
    replay = replay[replay.ref_source != "none"].reset_index(drop=True)
    replay["text"] = [state_text(r) for r in replay.itertuples(index=False)]
    counts = labels.pivot_table(index="split", columns="ref_source", values="lo", aggfunc="size", fill_value=0)
    for c in ("sku", "family", "none"):
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[["sku", "family", "none"]].reindex(["train", "val", "test"]).fillna(0).astype(int)
    return Prepared(labels.reset_index(drop=True), replay, offers, counts)


def views(prep: Prepared) -> dict[str, pd.DataFrame]:
    L = prep.labels
    main, fam = L.ref_source == "sku", L.ref_source == "family"
    out = {"train": L[main & (L.split == "train")], "val": L[main & (L.split == "val")],
           "test": L[main & (L.split == "test")], "test_fallback": L[fam & (L.split == "test")]}
    return {k: v.reset_index(drop=True) for k, v in out.items()}


def embed_all(prep: Prepared, cfg: RunConfig, embedder=None, log=print):
    from .embed import QwenEmbedder
    V = views(prep)
    band_texts = [band_descriptions()[k] for k in BAND_KEYS]
    texts = list(dict.fromkeys(band_texts + [t for v in V.values() for t in v.text] + list(prep.replay.text)))
    if embedder is None:
        slug = cfg.embed_model.replace("/", "_")
        embedder = QwenEmbedder(cfg.embed_model, cfg.max_len, cfg.embed_batch, log=log,
                                cache_path=os.path.join(cfg.work_dir, f"embed_cache_{slug}_{cfg.max_len}.npz"))
    log(f"[embed] {len(texts):,} unique texts")
    vecs = embedder.embed(texts)
    index = {t: i for i, t in enumerate(texts)}

    def get(ts):
        return vecs[[index[t] for t in ts]] if len(ts) else np.zeros((0, vecs.shape[1]), dtype=np.float32)

    emb = {k: get(list(v.text)) for k, v in V.items()}
    emb["replay"] = get(list(prep.replay.text))
    emb["bands"] = get(band_texts)
    return V, emb


def run_training(V: dict, E: dict, cfg: RunConfig, log=print) -> list[dict]:
    tr, val = V["train"], V["val"]
    t_tr = soft_targets(tr.lo_ratio, tr.hi_ratio)

    def val_score(p):
        price = val.ref_price.to_numpy() * BAND_MIDS[decode_bands(p)]
        return interval_metrics(price, val.lo, val.hi, val.mid)["interval_hit_rate"]

    runs = []
    for seed in cfg.seeds:
        res = train_heads(E["train"], t_tr, E["bands"], E["val"], val_score,
                          dataclasses.replace(cfg.train, seed=seed), init_ckpt=cfg.init_ckpt, log=log)
        save_checkpoint(res, os.path.join(cfg.work_dir, f"head_seed{seed}.pt"))
        probs = {k: predict_proba(res.model, E[k], E["bands"]) for k in ("val", "test", "test_fallback", "replay")}
        runs.append({"seed": seed, "result": res, "probs": probs})
    return runs


def evaluate_all(prep: Prepared, V: dict, runs: list[dict]) -> dict[str, pd.DataFrame]:
    train_df = V["train"]
    training = pd.DataFrame([{"seed": r["seed"], "warm_start": r["result"].warm_start,
                              "best_epoch": r["result"].best_epoch,
                              "val_interval_hit_rate": r["result"].best_score} for r in runs])
    test, fb = V["test"], V["test_fallback"]
    ratios, ens = method_ratios(test, train_df, runs, "test")
    fb_ratios, _ = method_ratios(fb, train_df, runs, "test_fallback")
    offers_test = prep.offers[prep.offers.day >= VAL_END]

    def hit(df, r):
        p = df.ref_price.to_numpy() * r
        return ((p >= df.lo.to_numpy()) & (p <= df.hi.to_numpy())).astype(float)

    ens_ratio = ratios["CLM ensemble (median band)"]
    return {
        "training": training,
        "headline_test": score_table(test, ratios),
        "headline_fallback": score_table(fb, fb_ratios),
        "replay": replay_table(prep.replay, offers_test, train_df, runs),
        "calibration": calibration_table(ens.max(1) if len(ens) else [], hit(test, ens_ratio)),
        "band_ece": pd.DataFrame([{"split": "test", "band_ece": band_ece(ens, test.true_band.to_numpy())}]),
        "breakdowns": breakdowns(test, {"CLM ensemble hit": hit(test, ens_ratio),
                                        "B0 carry-forward hit": hit(test, ratios["B0 carry-forward"])}),
        "examples": prediction_examples(test, ens),
    }


def save_results(results: dict[str, pd.DataFrame], work_dir: str) -> None:
    out = os.path.join(work_dir, "results")
    os.makedirs(out, exist_ok=True)
    for name, t in results.items():
        t.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    head = results["headline_test"].set_index("method").interval_hit_rate.to_dict()
    json.dump({"test_interval_hit_rate": head}, open(os.path.join(out, "summary.json"), "w"), indent=1)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: 3 passed (loads ~200 MB of CSV; allow a few minutes)

- [ ] **Step 6: Full-scale data-prep dry run (no GPU)**

Run:
```bash
python -c "import time; from reserve_clm.pipeline import RunConfig, prepare, views; t=time.time(); p=prepare(RunConfig(data_roots=['../../Data'])); print(p.counts); print({k: len(v) for k, v in views(p).items()}, 'replay', len(p.replay)); print('text example:'); print(views(p)['test'].text.iloc[0]); print(f'{time.time()-t:.0f}s')"
```
Expected: counts table with non-zero `sku` and `family` columns for every split, train/val/test/test_fallback sizes in the thousands, replay ≈ 20,000 or fewer, a readable state text, runtime under ~10 minutes. Record these numbers for the final report.

- [ ] **Step 7: Checkpoint** — Run: `python -m pytest -q` → all pass.

---

### Task 10: Notebook generation and local smoke execution

**Files:**
- Create: `tools/build_notebook.py`, `tools/run_smoke.py`
- Generate: `notebooks/reserve_price_clm.ipynb`
- Test: `tests/test_notebook.py`

**Interfaces:**
- Consumes: all `reserve_clm/*.py` sources; `pipeline` API from Task 9.
- Produces: `notebooks/reserve_price_clm.ipynb`; `tools/run_smoke.py` writes `work/smoke/executed.ipynb`.

- [ ] **Step 1: Write the failing test**

`tests/test_notebook.py`:
```python
import ast
import subprocess
import sys

import nbformat

from conftest import ROOT

MODULES = ["__init__", "config", "bands", "data", "labels", "features", "text", "baselines",
           "evaluate", "report", "embed", "train", "pipeline"]


def test_build_notebook_inlines_current_modules():
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_notebook.py")], check=True)
    nb = nbformat.read(ROOT / "notebooks" / "reserve_price_clm.ipynb", as_version=4)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    for m in MODULES:
        cell = next(s for s in code if s.startswith(f"%%writefile reserve_clm/{m}.py\n"))
        assert cell.split("\n", 1)[1] == (ROOT / "reserve_clm" / f"{m}.py").read_text()
    for s in code:                                     # every non-magic cell must parse
        if not s.startswith("%%"):
            ast.parse(s)
    assert any("evaluate_all(" in s for s in code)
    assert not any("zero-shot" in s.lower() or "zero_shot" in s.lower() for s in code)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_notebook.py -v`
Expected: FAIL (`tools/build_notebook.py` missing → `CalledProcessError`)

- [ ] **Step 3: Write `tools/build_notebook.py`**

```python
"""Generate notebooks/reserve_price_clm.ipynb from the tested reserve_clm modules."""
import pathlib

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULES = ["__init__", "config", "bands", "data", "labels", "features", "text", "baselines",
           "evaluate", "report", "embed", "train", "pipeline"]

INTRO = """# Reserve price prediction with CLM-8B fine-tuning (POC)

Predicts a **daily reserve (minimum) price band per SKU × warehouse** by fine-tuning
[CLM-8B](https://github.com/Contrastive-LM/CLM) (frozen Qwen3-8B reader + small trained heads),
and scores it against Assurant's own historical floors.

**What the label is.** On a day where the same SKU × warehouse had both an accepted and a
non-accepted (countered / rejected) offer, Assurant's floor lies between the highest bid it turned
down and the lowest bid it accepted. The model learns to place today's reserve in that interval,
so a perfect score means *matches Assurant's current floor*, not *optimal price*.

**Inputs (all from days before the target day):** product attributes, a 30-day market reference
price, bids (counts, buyers, units, median/top bid vs market, accepted share), sales velocity,
Assurant's previous floor, weekday. Inventory and Buyer Profile files are not used (no date
overlap / label leakage).

**Candidates:** 14 reserve bands (multiples of the market price) described in English.
**Training:** CLM's soft-target InfoNCE on the heads, warm-started from the released CLM head,
3 seeds, early stopping on February validation. **Test:** March–April 2026.

**Runtime:** a GPU with ≥ 24 GB (L4 / A10G / A100) or 2 × T4 (Kaggle). Qwen3-8B needs ~16 GB.
Embedding ~35K texts takes roughly 10–25 minutes depending on the GPU; training the heads takes minutes.

**Data:** put `Offers.csv`, `Product Information.csv` and `Sales Order&Line items.csv` anywhere
under `DATA_DIR` (set below) — e.g. upload the `Data Files` folder as a Kaggle dataset or to Google Drive.
"""

SETUP = '''#@title 1. Environment setup
import os, subprocess, sys
CLM_COMMIT = "d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6"
if os.environ.get("RESERVE_SKIP_INSTALL") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.51", "accelerate",
                    "pandas", "pyarrow", "huggingface_hub"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    f"contrastive-lm @ git+https://github.com/Contrastive-LM/CLM@{CLM_COMMIT}"], check=True)
import torch
print("torch", torch.__version__, "| GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or "none")
'''

DATA = '''#@title 2. Where is the data?  (edit DATA_DIR if auto-detection fails)
DATA_DIR = os.environ.get("RESERVE_DATA_DIR", "")
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB and not DATA_DIR:
    from google.colab import drive
    drive.mount("/content/drive")
DATA_ROOTS = [DATA_DIR, "/kaggle/input", "/content/data", "/content/drive/MyDrive", "."]
WORK = os.environ.get("RESERVE_WORK_DIR") or ("/kaggle/working/reserve_clm_run" if os.path.isdir("/kaggle/working")
                                              else "reserve_clm_run")
os.makedirs("reserve_clm", exist_ok=True)
os.makedirs(WORK, exist_ok=True)
print("searching", [r for r in DATA_ROOTS if r], "| outputs ->", WORK)
'''

CONFIG = '''#@title 4. Run configuration
import pandas as pd
from IPython.display import display
from reserve_clm.pipeline import RunConfig, prepare, views, embed_all, run_training, evaluate_all, save_results
from reserve_clm.train import TrainConfig
from reserve_clm.bands import BAND_KEYS, BAND_LABELS
from reserve_clm.text import band_descriptions
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.width", 200)

SMOKE = os.environ.get("RESERVE_SMOKE") == "1"   # tiny CPU run used to test the notebook itself
if SMOKE:
    cfg = RunConfig(data_roots=DATA_ROOTS, work_dir=WORK, embed_model="Qwen/Qwen3-0.6B", max_len=512,
                    embed_batch=8, seeds=(1234,), train=TrainConfig(epochs=3, patience=3),
                    replay_max_groups=40, smoke_rows=40, init_ckpt=None)
else:
    from clm.heads import download
    cfg = RunConfig(data_roots=DATA_ROOTS, work_dir=WORK, init_ckpt=download())   # released CLM head
print(cfg)
'''

PREP = '''#@title 5. Load data, build labels and features
prep = prepare(cfg)
print("Labelled SKU x warehouse x days by split and reference source")
print("(sku = main set, family = fallback set tested separately, none = cannot be priced)")
display(prep.counts)
V = views(prep)
print({k: len(v) for k, v in V.items()}, "| replay groups:", len(prep.replay))
'''

SHOW = '''#@title 6. What CLM reads: one state and the 14 band candidates
print(V["test"].text.iloc[0])
display(pd.DataFrame({"band": BAND_LABELS, "description": [band_descriptions()[k] for k in BAND_KEYS]}))
'''

EMBED = '''#@title 7. Embed every text once with Qwen3-8B (cached to WORK)
V, E = embed_all(prep, cfg)
import gc; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
print({k: v.shape for k, v in E.items()})
'''

TRAIN = '''#@title 8. Fine-tune the CLM heads (one run per seed)
runs = run_training(V, E, cfg)
'''

EVAL = '''#@title 9. Evaluate on March-April 2026 and save results
results = evaluate_all(prep, V, runs)
save_results(results, WORK)
print("Training"); display(results["training"])
cols = ["method", "n", "interval_hit_rate", "too_high_rate", "too_low_rate", "mape_vs_mid", "bias_vs_mid",
        "band_exact", "band_within_1", "band_mae"]
print("MAIN TEST SET (SKU-level market price)")
display(results["headline_test"].reindex(columns=cols + ["hit_rate_std"]).round(4))
print("FALLBACK TEST SET (family-level market price)")
display(results["headline_fallback"].reindex(columns=cols).round(4))
'''

EVAL2 = '''#@title 10. Offer-level replay, calibration, breakdowns, examples
print("Replay: accept if bid >= reserve, compared with Assurant's actual OFFER_STATUS")
display(results["replay"].round(4))
print("Calibration of the CLM ensemble (test, main set)")
display(results["calibration"].round(4)); display(results["band_ece"].round(4))
print("Interval hit rate by segment (test, main set)")
display(results["breakdowns"].round(4))
print("Example predictions")
display(results["examples"])
print("All tables saved to", os.path.join(WORK, "results"))
'''

READ = """## How to read the results

- **interval_hit_rate** (main metric): share of test days where the predicted reserve lies inside
  Assurant's floor interval [highest bid turned down, lowest bid accepted].
- **too_high / too_low**: reserve above the lowest accepted bid (would have lost that sale) /
  below the highest bid Assurant turned down (would have accepted a bid Assurant did not).
- **mape / bias vs mid**: price error against the interval midpoint; positive bias = reserve too high.
- **band_exact / within_1 / mae**: band accuracy against the band containing the interval midpoint.
- **Replay**: every test-period offer on sampled SKU × warehouse × days, accepted if bid ≥ reserve.
  *lost_sales_rate* = we reject, Assurant accepted; *underpricing_rate* = we accept, Assurant did not.
- **B0 carry-forward** (previous floor) is the strongest rule baseline; CLM must beat it to be useful.
- Fallback rows use a family-level market price and are reported separately; the model never trains on them.
"""


def build() -> pathlib.Path:
    cells = [new_markdown_cell(INTRO), new_code_cell(SETUP), new_code_cell(DATA),
             new_markdown_cell("## 3. Project code\nThe tested modules, written to `reserve_clm/` (collapsed).")]
    for m in MODULES:
        src = (ROOT / "reserve_clm" / f"{m}.py").read_text()
        cell = new_code_cell(f"%%writefile reserve_clm/{m}.py\n{src}")
        cell.metadata["jupyter"] = {"source_hidden": True}
        cell.metadata["cellView"] = "form"
        cells.append(cell)
    cells += [new_code_cell(CONFIG), new_code_cell(PREP), new_code_cell(SHOW), new_code_cell(EMBED),
              new_code_cell(TRAIN), new_code_cell(EVAL), new_code_cell(EVAL2), new_markdown_cell(READ)]
    nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                            "language": "python"},
                                             "accelerator": "GPU"})
    out = ROOT / "notebooks" / "reserve_price_clm.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    print(build())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_notebook.py -v`
Expected: 1 passed

- [ ] **Step 5: Write `tools/run_smoke.py`**

```python
"""Execute the generated notebook end to end in smoke mode (CPU, Qwen3-0.6B, subsampled data)."""
import asyncio
import os
import pathlib
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parents[1]

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

work = ROOT / "work" / "smoke"
work.mkdir(parents=True, exist_ok=True)
os.environ.update(RESERVE_SMOKE="1", RESERVE_SKIP_INSTALL="1", RESERVE_DATA_DIR=str(ROOT.parents[1] / "Data"),
                  RESERVE_WORK_DIR=str(work / "run"))
nb = nbformat.read(ROOT / "notebooks" / "reserve_price_clm.ipynb", as_version=4)
started = time.time()
try:
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(work)}}).execute()
finally:
    nbformat.write(nb, work / "executed.ipynb")
print(f"smoke run OK in {time.time() - started:.0f}s -> {work / 'executed.ipynb'}")
```

- [ ] **Step 6: Execute the notebook locally in smoke mode**

Run: `python tools/run_smoke.py`
Expected: `smoke run OK in …s`. Then inspect: `work/smoke/run/results/headline_test.csv` exists and has the rows `B0 carry-forward`, `B1 fixed ratio`, `B2 majority band`, `CLM seed 1234`, `CLM ensemble (median band)`, `CLM ensemble (expected ratio)`. If a cell fails, open `work/smoke/executed.ipynb`, fix the module (not the notebook), rerun `python tools/build_notebook.py` and repeat.

- [ ] **Step 7: Checkpoint** — Run: `python -m pytest -q` → all pass.

---

### Task 11: Final verification and hand-off

**Files:**
- Modify: none (verification only), unless a check fails.

- [ ] **Step 1: Full test suite** — Run: `python -m pytest -v` → every test passes (report the count).
- [ ] **Step 2: Rebuild and re-verify the deliverable** — Run: `python tools/build_notebook.py && python -m pytest tests/test_notebook.py -q` → passes; the notebook inlines the final module versions.
- [ ] **Step 3: Static checks on the notebook** — Confirm: no zero-shot cells; `RESERVE_SMOKE` defaults off; default `RunConfig` uses `Qwen/Qwen3-8B`, seeds `(1234, 1, 2)`, `replay_max_groups=20000`, `init_ckpt=download()`.
- [ ] **Step 4: Report to the user** — Deliver `notebooks/reserve_price_clm.ipynb` with: how to run on Kaggle (2×T4, add data as dataset) and Colab (L4/A100, Drive), expected runtime, the full-scale data counts from Task 9 Step 6, the smoke-run evidence, and the known limitations (label = Assurant's current floor; no inventory data; reference needs ≥3 accepted offers in 30 days).
