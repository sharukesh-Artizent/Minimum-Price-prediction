# Reserve (Daily Minimum) Price POC — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline POC that predicts a daily reserve price per SKU × Warehouse and proves, by replaying historical bids, whether it beats Assurant's current implicit floor.

**Architecture:** A pandas pipeline turns three local CSVs into one point-in-time feature row per SKU × Warehouse × day. On top of that sit:
- **Phase 1 baselines:** B0, Assurant's floor inferred with an interval-censored XGBoost AFT model, and B1, a tuned discount rule.
- **Phase 2:** predict-then-optimize. Market value (M1), bid distribution (M3) and counter take-up (M4) feed a grid search that maximizes expected value per unit, within a trust region around B0. A sibling isotonic projection follows.

A replay harness scores every policy on held-out offers. A report checks the spec's success criteria.

**Tech Stack:** Python 3.13, uv, pandas, numpy, pyarrow, scikit-learn (IsotonicRegression), xgboost (AFT), lightgbm, matplotlib, pytest.

**Spec:** `poc/Pareto/docs/spec.md`

## Global Constraints

- All paths below are relative to `poc/Pareto/`. Run every command from `C:\Artizent\Offer Clearance\poc\Pareto` in Git Bash.
- Data is read **only** from `../../Data/Data Files/`. Nothing is written there.
- **`ORDER_STATUS` is never loaded or used.**
- `OFFER_STATUS` and `COUNTER_PRICE` are used as Assurant's decisions.
- Sales are filtered to `ORDER_TYPE == "STOCKLIST"` and `STATUS != "CANCELED"`, with `units = SALES_QTY − QUANTITY_CANCELLED > 0`.
- The decision unit is `KEYS = ["sku", "wh", "day"]`, and `day` is a normalized `datetime64`.
- **Point-in-time rule:** every feature for day `d` uses only events with `day < d`.
- **No k-fold.** All validation is time-ordered.
- Default dates:
  - `DATA_START = 2025-10-10`, with `WARMUP_DAYS = 14`
  - `INNER_VALID_DAYS = 28`
  - `TEST_START = 2026-03-01`
  - `TEST_END = 2026-04-09`
- Default economics, from the spec:
  - `holding_cost_pct_per_day = 0.001`
  - `salvage_floor_pct = 0.50`
  - grid `g ∈ [0.70, 1.15]`, step `0.01`
  - `trust_region = 0.15`
- Success thresholds: F1 ≥ 0.80, F2 ±5%, F3 ±10%, S1 CI lower bound > 0, S2 ≥ −2 points, S3 LightGBM MAPE ≤ naive, S4 sign unchanged.
- Every commit ends with the trailer line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Data facts the code relies on (verified 2026-10-08)

**Offers.csv** (531,783 rows)
- `OFFER_CREATE_DATETIME` uses the format `%m/%d/%Y %H:%M`, and month and day may be single-digit.
- `COUNTER_PRICE` holds the literal string `NULL` when there is no counter.
- `OFFER_STATUS` values: ACCEPTED 284,754; COUNTERED 205,444; NOT_ACCEPTED 40,706; PENDING_ACCEPTANCE 879.

**Sales Order&Line items.csv** (645,100 lines)
- `DATE` uses the same format.
- `ORDER_TYPE` values: STOCKLIST, LPN AUCTION, AUCTION, PO.
- `PRODUCT_ID` joins to Product `SKU` for 99.9% of rows.

**Product Information.csv**
- Has up to 17 rows per SKU (one per warehouse); attributes are per SKU.
- `GRADE` values include B+, C, B, A+, R, REC, N, L, C+ and blank.
- `CAPACITY` values include 16GB … 1TB and `*`.
- `LAUNCH_DATE` is about 97% blank, so it is not used.

**Offer to sale link** (buyer + SKU + ≤ 3 days, nearest later sale, price within 1%)
- Match rates: ACCEPTED ≈ 49%, COUNTERED ≈ 50%, NOT_ACCEPTED ≈ 1%.

**Inventory.csv** is not used. It covers only 2025-07-01 → 2025-08-27.

## File Structure

```
poc/Pareto/
  pyproject.toml                 # uv project, deps, pytest config
  .gitignore
  docs/spec.md                   # design (exists)
  docs/plans/2026-10-08-reserve-price-poc.md   # this plan
  src/reserve/
    __init__.py
    config.py        # paths, dates, Params dataclass, hierarchy, orders, thresholds
    loaders.py       # CSV -> clean frames (offers, sales, product) + parquet cache
    labels.py        # floor_bounds, match_offers_to_sales, match_recall
    rolling.py       # trailing_sum (point-in-time windowed sums)
    shrink.py        # level_trailing_mean, shrink, shrink_frame (hierarchical EB)
    frame.py         # categorical encoding helpers shared by GBM models
    features.py      # attach_product, build_features
    market.py        # realized_targets, mape, MarketModel (M1)
    economics.py     # estimate_delta, estimate_rho, continuation_value
    counter_curve.py # CounterCurve, fit_counter_curve (M4)
    bids.py          # BidDist (M3)
    floor_model.py   # FloorModel (B0, XGBoost AFT)
    optimizer.py     # price_grid, expected_value, optimize (Phase 2)
    siblings.py      # enforce_monotone
    replay.py        # replay_policy, replay_actual, summarize, hindsight_reserve
    metrics.py       # bootstrap_uplift, pareto_segment, segment_table, deviation_bucket
    rule_b1.py       # tau_edges_from, bands, tune_rule, apply_rule (B1)
    pipeline.py      # Data, Fitted, prepare*, fit_models, enrich, score_groups, run_window, run_backtest, daily_reserve
    report.py        # explain, md_table, build_report (markdown + charts)
  scripts/run_poc.py # CLI: backtest | daily
  tests/
    factories.py     # tiny frame builders + synthetic_world
    test_*.py        # one per module
  data/interim/      # parquet cache (gitignored)
  output/            # reports, csv, charts (gitignored)
```

---

### Task 1: Project scaffold, config, test factories

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/reserve/__init__.py`, `src/reserve/config.py`, `tests/factories.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `reserve.config`: `ROOT`, `RAW`, `INTERIM`, `OUTPUT`, `OFFERS_CSV`, `SALES_CSV`, `PRODUCT_CSV`, `KEYS`, `DATA_START`, `WARMUP_DAYS`, `INNER_VALID_DAYS`, `TEST_START`, `TEST_END`, `LEVELS`, `CAPACITY_ORDER`, `GRADE_ORDER`, `DECISION_STATUSES`, `H_GRID`, `SENSITIVITY`, `Params`, plus the thresholds `F1_MIN`, `F2_TOL`, `F3_TOL`, `S2_MAX_DROP`
  - `tests/factories.py`: `offers_df(rows)`, `sales_df(rows)`, `product_df(rows)`

- [ ] **Step 1: Init git and uv project files**

```bash
cd "/c/Artizent/Offer Clearance/poc/Pareto"
git init
mkdir -p src/reserve tests scripts data/interim output
```

Create `pyproject.toml`:

```toml
[project]
name = "reserve-poc"
version = "0.1.0"
description = "Daily reserve (minimum) price POC"
requires-python = ">=3.11,<3.14"
dependencies = [
  "pandas>=2.2",
  "numpy>=1.26",
  "pyarrow>=15",
  "scikit-learn>=1.4",
  "xgboost>=2.0",
  "lightgbm>=4.3",
  "matplotlib>=3.8",
]

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/reserve"]

[tool.pytest.ini_options]
pythonpath = ["src", "tests"]
testpaths = ["tests"]
```

Create `.gitignore`:

```
.venv/
__pycache__/
*.pyc
data/
output/
*.parquet
.pytest_cache/
```

Create `src/reserve/__init__.py` as an empty file.

- [ ] **Step 2: Install deps**

Run: `uv sync`
Expected: `.venv` is created, and xgboost, lightgbm, pandas and scikit-learn install without errors.

- [ ] **Step 3: Write the failing config test**

Create `tests/test_config.py`:

```python
from reserve import config


def test_raw_files_exist():
    assert config.OFFERS_CSV.exists(), config.OFFERS_CSV
    assert config.SALES_CSV.exists(), config.SALES_CSV
    assert config.PRODUCT_CSV.exists(), config.PRODUCT_CSV


def test_levels_coarse_to_fine():
    assert config.LEVELS[0] == ["category"]
    assert config.LEVELS[-1] == ["sku", "wh"]


def test_params_grid_defaults():
    p = config.Params()
    assert (p.grid_lo, p.grid_hi, p.grid_step) == (0.70, 1.15, 0.01)
    assert p.trust_region == 0.15
    assert p.holding_cost_pct_per_day == 0.001
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'config'`

- [ ] **Step 5: Write config**

Create `src/reserve/config.py`:

```python
"""Paths, dates and tunable constants for the reserve-price POC."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # poc/Pareto
RAW = ROOT.parents[1] / "Data" / "Data Files"       # C:/Artizent/Offer Clearance/Data/Data Files
INTERIM = ROOT / "data" / "interim"
OUTPUT = ROOT / "output"

OFFERS_CSV = RAW / "Offers" / "Offers.csv"
SALES_CSV = RAW / "Sales" / "Historical Sales" / "Sales Order&Line items.csv"
PRODUCT_CSV = RAW / "Product" / "Product Information.csv"

KEYS = ["sku", "wh", "day"]
DECISION_STATUSES = ["ACCEPTED", "COUNTERED", "NOT_ACCEPTED"]

DATA_START = "2025-10-10"
WARMUP_DAYS = 14
INNER_VALID_DAYS = 28
TEST_START = "2026-03-01"
TEST_END = "2026-04-09"

# Product hierarchy, coarse -> fine. Each entry lists the key columns of that level.
LEVELS: list[list[str]] = [
    ["category"],
    ["model"],
    ["model", "grade"],
    ["model", "capacity", "grade"],
    ["sku"],
    ["sku", "wh"],
]
CAPACITY_ORDER = ["16GB", "32GB", "64GB", "128GB", "256GB", "512GB", "1TB"]
# Worst -> best. Grades outside this list (N, REC, R, L, UNK) are not constrained. ASSUMPTION.
GRADE_ORDER = ["D", "C", "C+", "B", "B+", "A", "A+"]

H_GRID = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]
# (holding-cost multiplier, depreciation multiplier)
SENSITIVITY = [(0.5, 1.0), (1.5, 1.0), (1.0, 0.5), (1.0, 1.5)]

F1_MIN = 0.80
F2_TOL = 0.05
F3_TOL = 0.10
S2_MAX_DROP = 0.02


@dataclass(frozen=True)
class Params:
    shrink_k: float = 5.0
    price_window_days: int = 90
    velocity_window_days: int = 28
    bid_window_days: int = 90
    min_group_obs: int = 30
    n_bid_quantiles: int = 20
    match_window_days: int = 3
    match_price_tol: float = 0.01
    holding_cost_pct_per_day: float = 0.001   # ASSUMPTION until Assurant provides h
    salvage_floor_pct: float = 0.50
    grid_lo: float = 0.70
    grid_hi: float = 1.15
    grid_step: float = 0.01
    trust_region: float = 0.15
    max_tau_days: float = 90.0
```

- [ ] **Step 6: Write test factories**

Create `tests/factories.py`:

```python
"""Tiny frame builders matching the schemas produced by reserve.loaders."""
import numpy as np
import pandas as pd

OFFER_COLS = ["offer_id", "buyer_id", "sku", "wh", "qty", "offer_price",
              "counter_price", "status", "ts", "day"]
OFFER_DEFAULTS = {"buyer_id": "b1", "wh": "W1", "qty": 1.0,
                  "counter_price": np.nan, "status": "ACCEPTED"}
SALE_DEFAULTS = {"buyer_id": "b1", "wh": "W1", "units": 1.0}
PRODUCT_DEFAULTS = {"category": "Phones", "manufacturer": "Apple", "model": "M1",
                    "capacity": "128GB", "grade": "A"}


def offers_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{**OFFER_DEFAULTS, **r} for r in rows])
    df["day"] = pd.to_datetime(df["day"]).dt.normalize()
    if "ts" in df:
        df["ts"] = pd.to_datetime(df["ts"])
    else:
        df["ts"] = df["day"] + pd.Timedelta(hours=10)
    df["offer_id"] = [str(i) for i in range(len(df))]
    for c in ["qty", "offer_price", "counter_price"]:
        df[c] = df[c].astype(float)
    return df[OFFER_COLS]


def sales_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{**SALE_DEFAULTS, **r} for r in rows])
    df["day"] = pd.to_datetime(df["day"]).dt.normalize()
    if "ts" in df:
        df["ts"] = pd.to_datetime(df["ts"])
    else:
        df["ts"] = df["day"] + pd.Timedelta(hours=12)
    df["unit_price"] = df["unit_price"].astype(float)
    df["units"] = df["units"].astype(float)
    return df[["buyer_id", "sku", "wh", "unit_price", "units", "ts", "day"]]


def product_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{**PRODUCT_DEFAULTS, **r} for r in rows])
    return df[["sku", "category", "manufacturer", "model", "capacity", "grade"]]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src/reserve/__init__.py src/reserve/config.py tests/factories.py tests/test_config.py docs/
git commit -m "chore: scaffold reserve-price POC with config and test factories" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Loaders and parquet cache

**Files:**
- Create: `src/reserve/loaders.py`, `tests/test_loaders.py`

**Interfaces:**
- Consumes: `config.OFFERS_CSV`, `config.SALES_CSV`, `config.PRODUCT_CSV`, `config.INTERIM`
- Produces:
  - `load_offers(path) -> DataFrame[offer_id, buyer_id, sku, wh, qty, offer_price, counter_price, status, ts, day]`
  - `load_sales(path) -> DataFrame[buyer_id, sku, wh, unit_price, units, ts, day]`
  - `load_product(path) -> DataFrame[sku, category, manufacturer, model, capacity, grade]`
  - `load_all(use_cache=True) -> tuple[offers, sales, product]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_loaders.py`:

```python
import csv

import numpy as np
import pandas as pd

from reserve.loaders import load_offers, load_product, load_sales

OFFER_HDR = ["OFFER_ID", "BUYER_ID", "SKU", "WAREHOUSE", "OFFER_QUANTITY", "OFFER_PRICE",
             "COUNTER_PRICE", "OFFER_STATUS", "ORDER_STATUS", "OFFER_CREATE_DATETIME",
             "TIMESTAMP", "SUBMITTED_SHOPPING_CART_ID"]
SALES_HDR = ["SALE_ID", "BUYER_ID", "WAREHOUSE_CODE", "CURRENCY_CODE", "STATUS", "ORDER_TYPE",
             "DATE", "SALE_COMPLETION_DATE", "SHOPPING_CART_ID", "PAYMENT_METHOD",
             "SHIPPING_CARRIER", "CPR_ORDER", "PX_CONNECT_ORDER", "PXN_BUYER_ID",
             "HYLA_ORDER_NUMBER", "HYLA_ORDER_STATUS", "LINE_ITEM_ID", "PRODUCT_ID", "LPN",
             "UNIT_PRICE", "SALES_QTY", "QUANTITY_SHIPPED", "QUANTITY_BACKORDERED",
             "QUANTITY_CANCELLED", "TOTAL_PRICE"]
PRODUCT_HDR = ["SKU", "WAREHOUSE", "ITEM_DESCRIPTION", "CATEGORY", "PRODUCT_FAMILY",
               "MANUFACTURER", "MODEL", "MODEL_NUMBER", "CAPACITY", "COLOR", "CARRIER",
               "LOCK_STATUS", "LAUNCH_DATE", "PROTOCOL", "CHANNEL", "INACTIVE", "DEACTIVATED",
               "UPDATE_DATETIME", "GRADE_STANDARD", "GRADE"]


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in header})


def test_load_offers_parses_and_drops_order_status(tmp_path):
    p = tmp_path / "o.csv"
    write_csv(p, OFFER_HDR, [
        dict(OFFER_ID="1", BUYER_ID="70", SKU="111", WAREHOUSE="W23", OFFER_QUANTITY="6",
             OFFER_PRICE="239", COUNTER_PRICE="NULL", OFFER_STATUS="ACCEPTED",
             ORDER_STATUS="REDEEMED", OFFER_CREATE_DATETIME="10/20/2025 13:08"),
        dict(OFFER_ID="2", BUYER_ID="70", SKU="111", WAREHOUSE="W23", OFFER_QUANTITY="2",
             OFFER_PRICE="200", COUNTER_PRICE="215", OFFER_STATUS="COUNTERED",
             ORDER_STATUS="NULL", OFFER_CREATE_DATETIME="1/5/2026 9:07"),
        dict(OFFER_ID="3", BUYER_ID="71", SKU="112", WAREHOUSE="TX1", OFFER_QUANTITY="1",
             OFFER_PRICE="0", COUNTER_PRICE="NULL", OFFER_STATUS="NOT_ACCEPTED",
             OFFER_CREATE_DATETIME="1/5/2026 9:07"),
        dict(OFFER_ID="4", BUYER_ID="71", SKU="112", WAREHOUSE="TX1", OFFER_QUANTITY="1",
             OFFER_PRICE="150", COUNTER_PRICE="NULL", OFFER_STATUS="NOT_ACCEPTED",
             OFFER_CREATE_DATETIME="garbage"),
    ])
    df = load_offers(p)
    assert list(df.offer_id) == ["1", "2"]
    assert not any("order" in c.lower() for c in df.columns)
    assert np.isnan(df.counter_price.iloc[0])
    assert df.counter_price.iloc[1] == 215.0
    assert df.day.iloc[1] == pd.Timestamp("2026-01-05")
    assert df.ts.iloc[1].hour == 9
    assert df.wh.iloc[0] == "W23"


def test_load_sales_keeps_only_stocklist_non_cancelled(tmp_path):
    p = tmp_path / "s.csv"
    base = dict(BUYER_ID="418", WAREHOUSE_CODE="TX1", DATE="10/19/2025 11:26",
                PRODUCT_ID="GLCP1", UNIT_PRICE="433")
    write_csv(p, SALES_HDR, [
        {**base, "STATUS": "INVOICED", "ORDER_TYPE": "STOCKLIST", "SALES_QTY": "6", "QUANTITY_CANCELLED": "0"},
        {**base, "STATUS": "INVOICED", "ORDER_TYPE": "AUCTION", "SALES_QTY": "6", "QUANTITY_CANCELLED": "0"},
        {**base, "STATUS": "CANCELED", "ORDER_TYPE": "STOCKLIST", "SALES_QTY": "6", "QUANTITY_CANCELLED": "0"},
        {**base, "STATUS": "INVOICED", "ORDER_TYPE": "STOCKLIST", "SALES_QTY": "2", "QUANTITY_CANCELLED": "2"},
    ])
    df = load_sales(p)
    assert len(df) == 1
    assert df.units.iloc[0] == 6.0
    assert df.sku.iloc[0] == "GLCP1"
    assert df.unit_price.iloc[0] == 433.0


def test_load_product_dedupes_and_fills(tmp_path):
    p = tmp_path / "p.csv"
    write_csv(p, PRODUCT_HDR, [
        dict(SKU="111", WAREHOUSE="W1", CATEGORY="Phones", MANUFACTURER="Apple", MODEL="iPhone 12", CAPACITY="128GB", GRADE=""),
        dict(SKU="111", WAREHOUSE="W2", CATEGORY="Phones", MANUFACTURER="Apple", MODEL="iPhone 12", CAPACITY="128GB", GRADE="A"),
        dict(SKU="112", WAREHOUSE="W1", CATEGORY="Phones", MANUFACTURER="Samsung", MODEL="S21", CAPACITY="256GB", GRADE=""),
    ])
    df = load_product(p).set_index("sku")
    assert len(df) == 2
    assert df.loc["111", "grade"] == "A"
    assert df.loc["112", "grade"] == "UNK"
    assert list(df.columns) == ["category", "manufacturer", "model", "capacity", "grade"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_loaders.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve.loaders'`

- [ ] **Step 3: Implement loaders**

Create `src/reserve/loaders.py`:

```python
"""Read the raw CSVs into clean, typed frames. ORDER_STATUS is never read."""
import pandas as pd

from reserve import config

_NA = ["NULL", ""]
_DT_FMT = "%m/%d/%Y %H:%M"


def _read(path, cols: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=cols, dtype=str, na_values=_NA,
                       keep_default_na=False, encoding="utf-8", encoding_errors="ignore")


def load_offers(path=config.OFFERS_CSV) -> pd.DataFrame:
    raw = _read(path, ["OFFER_ID", "BUYER_ID", "SKU", "WAREHOUSE", "OFFER_QUANTITY",
                       "OFFER_PRICE", "COUNTER_PRICE", "OFFER_STATUS", "OFFER_CREATE_DATETIME"])
    df = pd.DataFrame({
        "offer_id": raw["OFFER_ID"],
        "buyer_id": raw["BUYER_ID"],
        "sku": raw["SKU"],
        "wh": raw["WAREHOUSE"],
        "qty": pd.to_numeric(raw["OFFER_QUANTITY"], errors="coerce"),
        "offer_price": pd.to_numeric(raw["OFFER_PRICE"], errors="coerce"),
        "counter_price": pd.to_numeric(raw["COUNTER_PRICE"], errors="coerce"),
        "status": raw["OFFER_STATUS"],
        "ts": pd.to_datetime(raw["OFFER_CREATE_DATETIME"], format=_DT_FMT, errors="coerce"),
    })
    df = df.dropna(subset=["ts", "offer_price", "qty", "sku", "wh", "buyer_id", "status"])
    df = df[(df.offer_price > 0) & (df.qty > 0)].copy()
    df["day"] = df.ts.dt.normalize()
    return df.reset_index(drop=True)


def load_sales(path=config.SALES_CSV) -> pd.DataFrame:
    raw = _read(path, ["BUYER_ID", "WAREHOUSE_CODE", "STATUS", "ORDER_TYPE", "DATE",
                       "PRODUCT_ID", "UNIT_PRICE", "SALES_QTY", "QUANTITY_CANCELLED"])
    raw = raw[(raw.ORDER_TYPE == "STOCKLIST") & (raw.STATUS != "CANCELED")]
    units = (pd.to_numeric(raw.SALES_QTY, errors="coerce").fillna(0)
             - pd.to_numeric(raw.QUANTITY_CANCELLED, errors="coerce").fillna(0))
    df = pd.DataFrame({
        "buyer_id": raw.BUYER_ID,
        "sku": raw.PRODUCT_ID,
        "wh": raw.WAREHOUSE_CODE,
        "unit_price": pd.to_numeric(raw.UNIT_PRICE, errors="coerce"),
        "units": units,
        "ts": pd.to_datetime(raw.DATE, format=_DT_FMT, errors="coerce"),
    })
    df = df.dropna(subset=["ts", "unit_price", "sku", "wh"])
    df = df[(df.unit_price > 0) & (df.units > 0)].copy()
    df["day"] = df.ts.dt.normalize()
    return df.reset_index(drop=True)


def load_product(path=config.PRODUCT_CSV) -> pd.DataFrame:
    raw = _read(path, ["SKU", "CATEGORY", "MANUFACTURER", "MODEL", "CAPACITY", "GRADE"])
    raw = raw.rename(columns={"SKU": "sku", "CATEGORY": "category", "MANUFACTURER": "manufacturer",
                              "MODEL": "model", "CAPACITY": "capacity", "GRADE": "grade"})
    cols = ["sku", "category", "manufacturer", "model", "capacity", "grade"]
    df = raw[cols].dropna(subset=["sku"]).groupby("sku", as_index=False).first()
    return df[cols].fillna("UNK")


def load_all(use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, fn in [("offers", load_offers), ("sales", load_sales), ("product", load_product)]:
        f = config.INTERIM / f"{name}.parquet"
        if use_cache and f.exists():
            frames.append(pd.read_parquet(f))
        else:
            df = fn()
            df.to_parquet(f, index=False)
            frames.append(df)
    return tuple(frames)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_loaders.py -v`
Expected: 3 passed

- [ ] **Step 5: Smoke-run on real data**

Run: `uv run python -c "from reserve.loaders import load_all; o,s,p=load_all(use_cache=False); print(len(o), len(s), len(p)); print(o.status.value_counts())"`

Expected:
- offers ≈ 531,7xx rows
- sales: a few hundred thousand STOCKLIST rows
- product ≈ 174K SKUs
- the status counts match the data facts above

- [ ] **Step 6: Commit**

```bash
git add src/reserve/loaders.py tests/test_loaders.py
git commit -m "feat: add CSV loaders with STOCKLIST filter and parquet cache" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Labels: floor bounds, offer-to-sale match, match recall

**Files:**
- Create: `src/reserve/labels.py`, `tests/test_labels.py`

**Interfaces:**
- Consumes: the offers and sales frame schemas from Task 2
- Produces:
  - `floor_bounds(offers) -> DataFrame[sku, wh, day, lower, upper, n_offers, consistent]`. The floor lies in `(lower, upper]`, and NaN means unbounded.
  - `match_offers_to_sales(offers, sales, window_days, price_tol) -> offers + [ref_price, sale_match, counter_taken]`, with row order not guaranteed
  - `match_recall(matched, cutoff) -> pd.Series`, indexed by warehouse plus `"__all__"`, values in [0.05, 1]

- [ ] **Step 1: Write the failing tests**

Create `tests/test_labels.py`:

```python
import numpy as np
import pandas as pd

from factories import offers_df, sales_df
from reserve.labels import floor_bounds, match_offers_to_sales, match_recall


def test_floor_bounds_interval_and_consistency():
    o = offers_df([
        dict(sku="S", day="2026-01-01", offer_price=100, status="ACCEPTED"),
        dict(sku="S", day="2026-01-01", offer_price=95, status="ACCEPTED"),
        dict(sku="S", day="2026-01-01", offer_price=80, status="NOT_ACCEPTED"),
        dict(sku="S", day="2026-01-01", offer_price=85, status="COUNTERED", counter_price=92),
        dict(sku="S", day="2026-01-01", offer_price=10, status="PENDING_ACCEPTANCE"),
        dict(sku="T", day="2026-01-01", offer_price=90, status="ACCEPTED"),
        dict(sku="T", day="2026-01-01", offer_price=95, status="NOT_ACCEPTED"),
    ])
    b = floor_bounds(o).set_index("sku")
    assert b.loc["S", "lower"] == 85
    assert b.loc["S", "upper"] == 92
    assert bool(b.loc["S", "consistent"])
    assert b.loc["S", "n_offers"] == 4
    assert not bool(b.loc["T", "consistent"])


def test_match_offers_to_sales():
    o = offers_df([
        dict(buyer_id="b1", sku="S", day="2026-01-01", offer_price=100, status="ACCEPTED"),
        dict(buyer_id="b1", sku="S2", day="2026-01-01", offer_price=100, counter_price=104, status="COUNTERED"),
        dict(buyer_id="b2", sku="S", day="2026-01-01", offer_price=100, status="ACCEPTED"),
        dict(buyer_id="b3", sku="S", day="2026-01-01", offer_price=100, status="ACCEPTED"),
        dict(buyer_id="b4", sku="S", day="2026-01-01", offer_price=100, status="NOT_ACCEPTED"),
    ])
    s = sales_df([
        dict(buyer_id="b1", sku="S", day="2026-01-02", unit_price=100),
        dict(buyer_id="b1", sku="S2", day="2026-01-01", unit_price=104),
        dict(buyer_id="b2", sku="S", day="2026-01-06", unit_price=100),
        dict(buyer_id="b3", sku="S", day="2026-01-02", unit_price=90),
    ])
    m = match_offers_to_sales(o, s, window_days=3, price_tol=0.01).set_index("buyer_id")
    m = m.reset_index().set_index(["buyer_id", "sku"])
    assert bool(m.loc[("b1", "S"), "sale_match"])
    assert bool(m.loc[("b1", "S2"), "sale_match"])
    assert bool(m.loc[("b1", "S2"), "counter_taken"])
    assert m.loc[("b1", "S2"), "ref_price"] == 104
    assert not bool(m.loc[("b2", "S"), "sale_match"])      # outside 3-day window
    assert not bool(m.loc[("b3", "S"), "sale_match"])      # price off by 10%
    assert not bool(m.loc[("b4", "S"), "sale_match"])      # no sale
    assert not bool(m.loc[("b1", "S"), "counter_taken"])   # accepted, not countered


def test_match_recall_per_wh_with_global_fallback():
    rows = ([dict(wh="W1", status="ACCEPTED", sale_match=i < 300) for i in range(600)]
            + [dict(wh="W2", status="ACCEPTED", sale_match=True) for _ in range(10)]
            + [dict(wh="W1", status="COUNTERED", sale_match=False) for _ in range(50)])
    m = pd.DataFrame(rows)
    m["day"] = pd.Timestamp("2026-01-01")
    r = match_recall(m, cutoff=pd.Timestamp("2026-02-01"))
    assert r["W1"] == 0.5
    assert np.isclose(r["W2"], 310 / 610)
    assert np.isclose(r["__all__"], 310 / 610)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve.labels'`

- [ ] **Step 3: Implement labels**

Create `src/reserve/labels.py`:

```python
"""Labels derived from Assurant's decisions (OFFER_STATUS) and realized STOCKLIST sales."""
import numpy as np
import pandas as pd

from reserve.config import DECISION_STATUSES

MIN_WH_OBS = 500


def floor_bounds(offers: pd.DataFrame) -> pd.DataFrame:
    """Per SKU x WH x day, the interval (lower, upper] that contains Assurant's implicit floor."""
    o = offers[offers.status.isin(DECISION_STATUSES)]
    lower = o.offer_price.where(o.status.isin(["COUNTERED", "NOT_ACCEPTED"]))
    upper = o.offer_price.where(o.status.eq("ACCEPTED"))
    upper = upper.fillna(o.counter_price.where(o.status.eq("COUNTERED")))
    t = pd.DataFrame({"sku": o.sku, "wh": o.wh, "day": o.day, "lower": lower, "upper": upper})
    g = t.groupby(["sku", "wh", "day"], as_index=False).agg(
        lower=("lower", "max"), upper=("upper", "min"), n_offers=("lower", "size"))
    g["consistent"] = g.lower.isna() | g.upper.isna() | (g.lower < g.upper)
    return g


def match_offers_to_sales(offers: pd.DataFrame, sales: pd.DataFrame,
                          window_days: int, price_tol: float) -> pd.DataFrame:
    """Link each offer to the buyer's next sale of the same SKU within the window."""
    o = offers.sort_values("ts")
    s = (sales[["buyer_id", "sku", "ts", "unit_price"]]
         .rename(columns={"ts": "sale_ts", "unit_price": "sale_price"})
         .sort_values("sale_ts"))
    m = pd.merge_asof(o, s, left_on="ts", right_on="sale_ts", by=["buyer_id", "sku"],
                      direction="forward", tolerance=pd.Timedelta(days=window_days))
    is_counter = m.status.eq("COUNTERED") & m.counter_price.notna()
    m["ref_price"] = m.counter_price.where(is_counter, m.offer_price)
    m["sale_match"] = m.sale_ts.notna() & ((m.sale_price - m.ref_price).abs() / m.ref_price <= price_tol)
    m["counter_taken"] = is_counter & m.sale_match
    return m.drop(columns=["sale_ts", "sale_price"]).reset_index(drop=True)


def match_recall(matched: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Share of ACCEPTED offers before cutoff that link to a sale; per warehouse with global fallback."""
    acc = matched[(matched.status == "ACCEPTED") & (matched.day < cutoff)]
    glob = float(acc.sale_match.mean())
    by_wh = acc.groupby("wh").sale_match.agg(["mean", "size"])
    rate = pd.Series(np.where(by_wh["size"] >= MIN_WH_OBS, by_wh["mean"], glob), index=by_wh.index)
    rate["__all__"] = glob
    return rate.clip(lower=0.05)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_labels.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/reserve/labels.py tests/test_labels.py
git commit -m "feat: add floor bounds, offer-to-sale matching and match recall" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Point-in-time trailing sums and hierarchical shrinkage

**Files:**
- Create: `src/reserve/rolling.py`, `src/reserve/shrink.py`, `tests/test_rolling_shrink.py`

**Interfaces:**
- Produces:
  - `trailing_sum(events, keys, value_cols, queries, window_days) -> DataFrame[keys + day + value_cols]`. Each value is the sum over `[day − window, day)`, with 0 when there are no events.
  - `level_trailing_mean(events, value_col, weight_col, queries, levels, window_days, prefix) -> queries + {prefix}_mean_L{i}, {prefix}_n_L{i}`. The mean is weighted by `weight_col`, and n counts event rows.
  - `shrink(values, counts, k, prior) -> np.ndarray`, where the lists are ordered coarse to fine
  - `shrink_frame(df, prefix, n_levels, k, prior) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rolling_shrink.py`:

```python
import numpy as np
import pandas as pd

from reserve.rolling import trailing_sum
from reserve.shrink import level_trailing_mean, shrink

D0 = pd.Timestamp("2026-01-01")


def d(i):
    return D0 + pd.Timedelta(days=i)


def test_trailing_sum_excludes_query_day_and_window_edges():
    ev = pd.DataFrame({"sku": ["A"] * 3, "day": [d(1), d(2), d(5)], "v": [1.0, 2.0, 4.0]})
    q = pd.DataFrame({"sku": ["A"] * 4 + ["B"], "day": [d(2), d(3), d(6), d(9), d(3)]})
    out = trailing_sum(ev, ["sku"], ["v"], q, window_days=3).set_index(["sku", "day"]).v
    assert out[("A", d(2))] == 1.0     # [d-1, d2) -> day1 only, query day excluded
    assert out[("A", d(3))] == 3.0     # [d0, d3) -> days 1, 2
    assert out[("A", d(6))] == 4.0     # [d3, d6) -> day 5
    assert out[("A", d(9))] == 0.0     # [d6, d9) -> nothing
    assert out[("B", d(3))] == 0.0     # unseen key


def test_shrink_moves_toward_parent_by_count():
    v = [np.array([10.0, 10.0]), np.array([np.nan, 20.0])]
    n = [np.array([100.0, 100.0]), np.array([0.0, 5.0])]
    est = shrink(v, n, k=5.0, prior=0.0)
    root = (100 * 10 + 5 * 0) / 105
    assert np.isclose(est[0], root)
    assert np.isclose(est[1], (5 * 20 + 5 * root) / 10)


def test_level_trailing_mean_weighted_and_counted():
    ev = pd.DataFrame({"model": ["M", "M", "M"], "sku": ["a", "a", "b"],
                       "day": [d(1), d(1), d(1)], "logp": [1.0, 3.0, 5.0], "units": [1.0, 3.0, 1.0]})
    q = pd.DataFrame({"model": ["M"], "sku": ["a"], "day": [d(2)]})
    out = level_trailing_mean(ev, "logp", "units", q, [["model"], ["sku"]], 30, "px").iloc[0]
    assert np.isclose(out.px_mean_L0, (1 + 9 + 5) / 5)
    assert out.px_n_L0 == 3
    assert np.isclose(out.px_mean_L1, (1 + 9) / 4)
    assert out.px_n_L1 == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rolling_shrink.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve.rolling'`

- [ ] **Step 3: Implement rolling**

Create `src/reserve/rolling.py`:

```python
"""Point-in-time windowed sums: the value for day d uses events in [d - window, d)."""
import pandas as pd


def trailing_sum(events: pd.DataFrame, keys: list[str], value_cols: list[str],
                 queries: pd.DataFrame, window_days: int) -> pd.DataFrame:
    ev = events.groupby(keys + ["day"], as_index=False)[value_cols].sum()
    q = queries[keys + ["day"]].drop_duplicates()
    zeros = q.assign(**{c: 0.0 for c in value_cols})
    allr = (pd.concat([ev, zeros], ignore_index=True)
            .groupby(keys + ["day"], as_index=False)[value_cols].sum()
            .sort_values(keys + ["day"]))
    rolled = (allr.set_index("day").groupby(keys)[value_cols]
              .rolling(f"{window_days}D", closed="left").sum()
              .reset_index())
    out = q.merge(rolled, on=keys + ["day"], how="left")
    out[value_cols] = out[value_cols].fillna(0.0)
    return out
```

- [ ] **Step 4: Implement shrink**

Create `src/reserve/shrink.py`:

```python
"""Hierarchical empirical-Bayes shrinkage of trailing means (coarse -> fine)."""
import numpy as np
import pandas as pd

from reserve.rolling import trailing_sum


def level_trailing_mean(events: pd.DataFrame, value_col: str, weight_col: str,
                        queries: pd.DataFrame, levels: list[list[str]], window_days: int,
                        prefix: str) -> pd.DataFrame:
    out = queries.copy()
    ev = events.assign(_s=events[value_col] * events[weight_col], _w=events[weight_col], _c=1.0)
    for i, keys in enumerate(levels):
        t = trailing_sum(ev, keys, ["_s", "_w", "_c"], out, window_days)
        t[f"{prefix}_mean_L{i}"] = t["_s"] / t["_w"].where(t["_w"] > 0)
        t = t.rename(columns={"_c": f"{prefix}_n_L{i}"}).drop(columns=["_s", "_w"])
        out = out.merge(t, on=keys + ["day"], how="left")
    return out


def shrink(values: list[np.ndarray], counts: list[np.ndarray], k: float, prior: float) -> np.ndarray:
    est = np.full(len(values[0]), float(prior))
    for v, n in zip(values, counts):
        v = np.asarray(v, float)
        n = np.where(np.isnan(v), 0.0, np.asarray(n, float))
        est = (n * np.nan_to_num(v) + k * est) / (n + k)
    return est


def shrink_frame(df: pd.DataFrame, prefix: str, n_levels: int, k: float, prior: float) -> np.ndarray:
    vals = [df[f"{prefix}_mean_L{i}"].to_numpy(float) for i in range(n_levels)]
    cnts = [df[f"{prefix}_n_L{i}"].to_numpy(float) for i in range(n_levels)]
    return shrink(vals, cnts, k, prior)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_rolling_shrink.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/reserve/rolling.py src/reserve/shrink.py tests/test_rolling_shrink.py
git commit -m "feat: add point-in-time trailing sums and hierarchical shrinkage" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Feature builder

**Files:**
- Create: `src/reserve/features.py`, `tests/test_features.py`

**Interfaces:**
- Consumes:
  - `trailing_sum` (Task 4)
  - `level_trailing_mean` and `shrink_frame` (Task 4)
  - `config.LEVELS`, `config.DATA_START`, `config.WARMUP_DAYS`, `config.Params`
- Produces:
  - `attach_product(df, product) -> df + [category, manufacturer, model, capacity, grade]`, with missing values set to "UNK"
  - `build_features(queries, offers, sales, product, p) -> DataFrame` keyed by `KEYS`, with columns:
    - product attributes
    - `px_mean_L0..5`, `px_n_L0..5`
    - `log_m_naive`, `units_sold_w`, `tau`, `nb`, `ub`, `age_days`, `left_censored`, `dow`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_features.py`:

```python
import numpy as np
import pandas as pd

from factories import offers_df, product_df, sales_df
from reserve.config import Params
from reserve.features import build_features


def _data():
    product = product_df([dict(sku="S1"), dict(sku="S2", capacity="256GB")])
    sales = sales_df([dict(sku="S1", day=f"2025-10-{d:02d}", unit_price=100 + d, units=1)
                      for d in range(10, 31)])
    offers = offers_df([dict(sku="S1", day=f"2025-10-{d:02d}", offer_price=95) for d in range(10, 31)])
    return offers, sales, product


def test_point_in_time_no_future_leakage():
    # Query day is after the 14-day warm-up, so the shrinkage prior is identical in both runs.
    offers, sales, product = _data()
    q = pd.DataFrame({"sku": ["S1"], "wh": ["W1"], "day": [pd.Timestamp("2025-10-28")]})
    full = build_features(q, offers, sales, product, Params()).iloc[0]
    cut = pd.Timestamp("2025-10-28")
    part = build_features(q, offers[offers.day < cut], sales[sales.day < cut], product, Params()).iloc[0]
    for c in ["log_m_naive", "tau", "nb", "ub", "units_sold_w", "px_mean_L5"]:
        assert np.isclose(full[c], part[c]), c


def test_tau_age_and_counts():
    offers, sales, product = _data()
    q = pd.DataFrame({"sku": ["S1"], "wh": ["W1"], "day": [pd.Timestamp("2025-10-20")]})
    r = build_features(q, offers, sales, product, Params()).iloc[0]
    assert r.units_sold_w == 10                      # Oct 10..19
    assert np.isclose(r.tau, 28 / 10.5)
    assert r.nb == 10
    assert r.age_days == 10
    assert bool(r.left_censored)
    assert r.dow == pd.Timestamp("2025-10-20").dayofweek


def test_unseen_sku_falls_back_to_parent_level():
    offers, sales, product = _data()
    q = pd.DataFrame({"sku": ["S2"], "wh": ["W1"], "day": [pd.Timestamp("2025-10-25")]})
    r = build_features(q, offers, sales, product, Params()).iloc[0]
    assert np.isnan(r.px_mean_L5)
    assert np.log(100) < r.log_m_naive < np.log(130)  # borrowed from model M1 / category
    assert r.tau == Params().velocity_window_days / 0.5 or r.tau == Params().max_tau_days
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve.features'`

- [ ] **Step 3: Implement features**

Create `src/reserve/features.py`:

```python
"""One point-in-time feature row per SKU x WH x day."""
import numpy as np
import pandas as pd

from reserve.config import DATA_START, LEVELS, WARMUP_DAYS, Params
from reserve.rolling import trailing_sum
from reserve.shrink import level_trailing_mean, shrink_frame

PRODUCT_COLS = ["category", "manufacturer", "model", "capacity", "grade"]


def attach_product(df: pd.DataFrame, product: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(product, on="sku", how="left")
    out[PRODUCT_COLS] = out[PRODUCT_COLS].fillna("UNK")
    return out


def _prior(s: pd.DataFrame) -> float:
    early = s[s.day < pd.Timestamp(DATA_START) + pd.Timedelta(days=WARMUP_DAYS)]
    base = early if len(early) else s
    return float(np.average(base.logp, weights=base.units))


def build_features(queries: pd.DataFrame, offers: pd.DataFrame, sales: pd.DataFrame,
                   product: pd.DataFrame, p: Params) -> pd.DataFrame:
    keys = ["sku", "wh", "day"]
    q = attach_product(queries[keys].drop_duplicates().reset_index(drop=True), product)
    s = attach_product(sales, product).assign(logp=lambda d: np.log(d.unit_price))

    q = level_trailing_mean(s, "logp", "units", q, LEVELS, p.price_window_days, "px")
    q["log_m_naive"] = shrink_frame(q, "px", len(LEVELS), p.shrink_k, _prior(s))

    vel = trailing_sum(s.assign(units_sold_w=s.units), ["sku", "wh"], ["units_sold_w"],
                       q, p.velocity_window_days)
    q = q.merge(vel, on=keys, how="left")
    q["tau"] = np.clip(p.velocity_window_days / (q.units_sold_w + 0.5), 1.0, p.max_tau_days)

    ob = trailing_sum(offers.assign(nb=1.0, ub=offers.qty), ["sku", "wh"], ["nb", "ub"],
                      q, p.velocity_window_days)
    q = q.merge(ob, on=keys, how="left")

    first = (pd.concat([offers[keys], sales[keys]])
             .groupby(["sku", "wh"], as_index=False).day.min()
             .rename(columns={"day": "first_day"}))
    q = q.merge(first, on=["sku", "wh"], how="left")
    q["age_days"] = (q.day - q.first_day).dt.days.clip(lower=0).fillna(0).astype(float)
    q["left_censored"] = q.first_day <= pd.Timestamp(DATA_START)
    q["dow"] = q.day.dt.dayofweek
    return q.drop(columns="first_day")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features.py -v`
Expected: 3 passed. If `test_unseen_sku_falls_back_to_parent_level` fails only on the `tau` line, check that `units_sold_w` is 0 for S2. It should be, so `tau = 28 / 0.5 = 56`.

- [ ] **Step 5: Commit**

```bash
git add src/reserve/features.py tests/test_features.py
git commit -m "feat: build point-in-time SKU x WH x day features" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: M1 market value model

**Files:**
- Create: `src/reserve/frame.py`, `src/reserve/market.py`, `tests/test_market.py`

**Interfaces:**
- Consumes: the feature columns from Task 5
- Produces:
  - `frame.CAT_COLS`, `frame.fit_categories(df) -> dict`, `frame.apply_categories(df, cats) -> DataFrame`
  - `market.FEATURES_NUM`
  - `market.realized_targets(sales) -> DataFrame[sku, wh, day, y]`, where `y` is the log of the units-weighted mean price
  - `market.mape(y_log, yhat_log) -> float`
  - `market.MarketModel` with:
    - `.fit(train, valid) -> self`, where both frames have the features plus `y`
    - `.predict(df) -> np.ndarray` of log m̂
    - `.use_lgb: bool`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_market.py`:

```python
import numpy as np
import pandas as pd

from factories import sales_df
from reserve.frame import CAT_COLS
from reserve.market import FEATURES_NUM, MarketModel, mape, realized_targets


def test_realized_targets_weighted_mean():
    s = sales_df([dict(sku="S", day="2026-01-01", unit_price=100, units=1),
                  dict(sku="S", day="2026-01-01", unit_price=200, units=3)])
    t = realized_targets(s)
    assert np.isclose(t.y.iloc[0], np.log(175))


def _synthetic(n, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FEATURES_NUM})
    df["log_m_naive"] = rng.uniform(4, 6, n)
    for c in CAT_COLS:
        df[c] = "X"
    df["grade"] = rng.choice(["A", "B"], n)
    df["y"] = df.log_m_naive + np.where(df.grade == "A", 0.2, -0.1) + rng.normal(0, 0.01, n)
    return df


def test_market_model_learns_residual_and_beats_naive():
    tr, va = _synthetic(3000, 0), _synthetic(1000, 1)
    m = MarketModel().fit(tr, va)
    assert m.use_lgb
    assert mape(va.y, m.predict(va)) < 0.03
    assert mape(va.y, va.log_m_naive) > 0.09
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_market.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve.frame'`

- [ ] **Step 3: Implement the frame helpers**

Create `src/reserve/frame.py`:

```python
"""Stable categorical encoding shared by the GBM models."""
import pandas as pd

CAT_COLS = ["wh", "category", "manufacturer", "model", "capacity", "grade"]


def fit_categories(df: pd.DataFrame, cols: list[str] = CAT_COLS) -> dict[str, list[str]]:
    return {c: sorted(df[c].astype(str).unique().tolist()) for c in cols}


def apply_categories(df: pd.DataFrame, cats: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    for c, levels in cats.items():
        out[c] = pd.Categorical(out[c].astype(str), categories=levels)
    return out
```

- [ ] **Step 4: Implement the market model**

Create `src/reserve/market.py`:

```python
"""M1: market value m-hat = shrunk trailing log price + optional LightGBM residual."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from reserve.frame import CAT_COLS, apply_categories, fit_categories

FEATURES_NUM = ["log_m_naive", "px_mean_L1", "px_mean_L3", "px_mean_L5", "px_n_L3", "px_n_L5",
                "units_sold_w", "nb", "ub", "tau", "age_days", "dow"]
LGB_PARAMS = dict(objective="l1", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)


def realized_targets(sales: pd.DataFrame) -> pd.DataFrame:
    g = (sales.assign(pu=sales.unit_price * sales.units)
         .groupby(["sku", "wh", "day"], as_index=False)
         .agg(pu=("pu", "sum"), units=("units", "sum")))
    g["y"] = np.log(g.pu / g.units)
    return g[["sku", "wh", "day", "y"]]


def mape(y_log, yhat_log) -> float:
    y, yh = np.exp(np.asarray(y_log, float)), np.exp(np.asarray(yhat_log, float))
    return float(np.mean(np.abs(yh - y) / y))


class MarketModel:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.booster = None
        self.cats = None
        self.use_lgb = False

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        return apply_categories(df[FEATURES_NUM + CAT_COLS], self.cats)

    def _predict_lgb(self, df: pd.DataFrame) -> np.ndarray:
        resid = self.booster.predict(self._X(df), num_iteration=self.booster.best_iteration)
        return df.log_m_naive.to_numpy(float) + resid

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> "MarketModel":
        self.cats = fit_categories(train)
        dtr = lgb.Dataset(self._X(train), train.y - train.log_m_naive)
        dva = lgb.Dataset(self._X(valid), valid.y - valid.log_m_naive, reference=dtr)
        self.booster = lgb.train({**LGB_PARAMS, "seed": self.seed}, dtr, num_boost_round=2000,
                                 valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        self.use_lgb = mape(valid.y, self._predict_lgb(valid)) < mape(valid.y, valid.log_m_naive)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._predict_lgb(df) if self.use_lgb else df.log_m_naive.to_numpy(float)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_market.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/reserve/frame.py src/reserve/market.py tests/test_market.py
git commit -m "feat: add M1 market value model with naive fallback" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Economics: depreciation, realization ratio, continuation value

**Files:**
- Create: `src/reserve/economics.py`, `tests/test_economics.py`

**Interfaces:**
- Produces:
  - `estimate_delta(sales_p, cutoff, min_weeks=8, cap=0.01) -> float`. `sales_p` is sales with product columns attached. The result is a daily depreciation rate.
  - `estimate_rho(rows) -> float`, where `rows` has `y` and `log_m`, and the result is the median of `exp(y − log_m)`
  - `continuation_value(m, tau, delta, rho, h_pct, salvage_pct) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_economics.py`:

```python
import numpy as np
import pandas as pd

from reserve.economics import continuation_value, estimate_delta, estimate_rho


def test_continuation_value_formula_and_salvage_floor():
    w = continuation_value(np.array([100.0, 100.0]), np.array([10.0, 90.0]),
                           delta=0.01, rho=1.0, h_pct=np.array([0.001, 0.01]), salvage_pct=0.5)
    assert np.isclose(w[0], 100 * 0.99 ** 10 - 0.001 * 100 * 10)
    assert w[1] == 50.0


def test_estimate_delta_recovers_slope():
    days = pd.date_range("2025-10-10", periods=84, freq="D")
    s = pd.DataFrame({"model": "M", "capacity": "128GB", "grade": "A", "day": days,
                      "unit_price": 100 * np.exp(-0.002 * np.arange(84)), "units": 1.0})
    delta = estimate_delta(s, cutoff=pd.Timestamp("2026-06-01"))
    assert abs(delta - 0.002) < 2e-4


def test_estimate_delta_zero_when_no_long_series():
    s = pd.DataFrame({"model": "M", "capacity": "1", "grade": "A",
                      "day": pd.date_range("2025-10-10", periods=10), "unit_price": 100.0, "units": 1.0})
    assert estimate_delta(s, cutoff=pd.Timestamp("2026-01-01")) == 0.0


def test_estimate_rho_median_ratio():
    rows = pd.DataFrame({"y": np.log([102.0, 102.0, 102.0]), "log_m": np.log([100.0, 100.0, 100.0])})
    assert np.isclose(estimate_rho(rows), 1.02)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_economics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve.economics'`

- [ ] **Step 3: Implement economics**

Create `src/reserve/economics.py`:

```python
"""Value of waiting: depreciation, realization ratio, continuation value W."""
import numpy as np
import pandas as pd

GROUP = ["model", "capacity", "grade"]


def estimate_delta(sales_p: pd.DataFrame, cutoff: pd.Timestamp,
                   min_weeks: int = 8, cap: float = 0.01) -> float:
    s = sales_p[sales_p.day < cutoff]
    if s.empty:
        return 0.0
    s = s.assign(week=(s.day - s.day.min()).dt.days // 7, logp=np.log(s.unit_price))
    wk = s.groupby(GROUP + ["week"], as_index=False).logp.mean()
    slopes = [np.polyfit(g.week, g.logp, 1)[0] / 7.0
              for _, g in wk.groupby(GROUP) if len(g) >= min_weeks]
    if not slopes:
        return 0.0
    return float(np.clip(-np.median(slopes), 0.0, cap))


def estimate_rho(rows: pd.DataFrame) -> float:
    return float(np.median(np.exp(rows.y - rows.log_m)))


def continuation_value(m, tau, delta: float, rho: float, h_pct, salvage_pct: float) -> np.ndarray:
    m = np.asarray(m, float)
    tau = np.asarray(tau, float)
    w = m * rho * (1.0 - delta) ** tau - np.asarray(h_pct, float) * m * tau
    return np.maximum(w, salvage_pct * m)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_economics.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/reserve/economics.py tests/test_economics.py
git commit -m "feat: add depreciation, realization ratio and continuation value" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: M4 counter take-up curve

**Files:**
- Create: `src/reserve/counter_curve.py`, `tests/test_counter_curve.py`

**Interfaces:**
- Consumes: the matched offers frame (Task 3) with `status, counter_price, offer_price, counter_taken, wh, day`, and the recall Series (Task 3)
- Produces:
  - `CounterCurve(grid: np.ndarray, curves: dict[str, np.ndarray], max_gap: float)`
  - `CounterCurve.pi(gap, wh) -> np.ndarray`. `gap` may be N-dimensional with first axis length `len(wh)`. The result is 0 where `gap < 0` or `gap > max_gap`.
  - `fit_counter_curve(matched, cutoff, recall, min_obs=500, n_grid=61) -> CounterCurve`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_counter_curve.py`:

```python
import numpy as np
import pandas as pd

from reserve.counter_curve import CounterCurve, fit_counter_curve


def _matched(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    gap = rng.uniform(0.005, 0.10, n)
    true_p = 0.8 - 5 * gap
    taken = rng.random(n) < 0.5 * true_p            # 50% match recall
    b = np.full(n, 100.0)
    return pd.DataFrame({"status": "COUNTERED", "offer_price": b, "counter_price": b * (1 + gap),
                         "counter_taken": taken, "wh": "W1", "day": pd.Timestamp("2026-01-01")})


def test_fit_counter_curve_recall_corrected_and_monotone():
    recall = pd.Series({"W1": 0.5, "__all__": 0.5})
    c = fit_counter_curve(_matched(), pd.Timestamp("2026-02-01"), recall)
    assert np.all(np.diff(c.curves["W1"]) <= 1e-12)
    p = c.pi(np.array([0.01, 0.08]), np.array(["W1", "W1"]))
    assert abs(p[0] - 0.75) < 0.1
    assert abs(p[1] - 0.40) < 0.1


def test_pi_zero_outside_range_and_global_fallback():
    c = CounterCurve(grid=np.array([0.0, 0.1]), curves={"__all__": np.array([0.6, 0.2])}, max_gap=0.1)
    p = c.pi(np.array([[0.05, 0.2, -0.01]]), np.array(["UNSEEN"]))
    assert np.allclose(p, [[0.4, 0.0, 0.0]])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_counter_curve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve.counter_curve'`

- [ ] **Step 3: Implement the counter curve**

Create `src/reserve/counter_curve.py`:

```python
"""M4: P(buyer takes a counter) as a decreasing function of the counter gap (c - b) / b."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass
class CounterCurve:
    grid: np.ndarray
    curves: dict
    max_gap: float

    def pi(self, gap, wh) -> np.ndarray:
        gap = np.asarray(gap, float)
        wh = np.asarray(wh).astype(str)
        out = np.zeros_like(gap)
        for w in np.unique(wh):
            rows = wh == w
            curve = self.curves.get(w, self.curves["__all__"])
            out[rows] = np.interp(gap[rows], self.grid, curve)
        out[(gap < 0) | (gap > self.max_gap)] = 0.0
        return out


def fit_counter_curve(matched: pd.DataFrame, cutoff: pd.Timestamp, recall: pd.Series,
                      min_obs: int = 500, n_grid: int = 61) -> CounterCurve:
    c = matched[(matched.status == "COUNTERED") & (matched.day < cutoff) & matched.counter_price.notna()]
    gap = (c.counter_price - c.offer_price) / c.offer_price
    c, gap = c[gap > 0], gap[gap > 0]
    max_gap = float(np.quantile(gap, 0.99))
    keep = gap <= max_gap
    c, gap = c[keep], gap[keep]
    grid = np.linspace(0.0, max_gap, n_grid)

    def fit(g, y):
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip", y_min=0.0, y_max=1.0)
        return iso.fit(g, y).predict(grid)

    curves = {"__all__": np.clip(fit(gap, c.counter_taken.astype(float)) / recall["__all__"], 0, 1)}
    for w, idx in c.groupby("wh").groups.items():
        if len(idx) >= min_obs:
            r = recall.get(w, recall["__all__"])
            curves[str(w)] = np.clip(fit(gap.loc[idx], c.loc[idx, "counter_taken"].astype(float)) / r, 0, 1)
    return CounterCurve(grid, curves, max_gap)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_counter_curve.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/reserve/counter_curve.py tests/test_counter_curve.py
git commit -m "feat: add recall-corrected isotonic counter take-up curve" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: M3 bid distribution

**Files:**
- Create: `src/reserve/bids.py`, `tests/test_bids.py`

**Interfaces:**
- Consumes: offers merged with features and `m` (columns `offer_price`, `m`, `day`, and every key column in `LEVELS`), plus `Params`
- Produces:
  - `BidDist.fit(offers_f, cutoff, p, levels=LEVELS) -> BidDist`
  - `BidDist.lookup(df) -> np.ndarray[len(df), p.n_bid_quantiles]` of bid / m̂ ratios, taken from the finest level with at least `min_group_obs` bids, falling back to global

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bids.py`:

```python
import numpy as np
import pandas as pd

from reserve.bids import BidDist
from reserve.config import Params

LV = [["category"], ["model"]]


def _offers():
    rows = [dict(category="Phones", model="X", offer_price=90.0) for _ in range(40)]
    rows += [dict(category="Phones", model="Y", offer_price=50.0) for _ in range(5)]
    rows += [dict(category="Phones", model="Z", offer_price=70.0) for _ in range(40)]
    df = pd.DataFrame(rows)
    df["m"] = 100.0
    df["day"] = pd.Timestamp("2026-01-15")
    return df


def test_lookup_uses_finest_level_with_enough_obs():
    p = Params(n_bid_quantiles=4, min_group_obs=30)
    bd = BidDist.fit(_offers(), pd.Timestamp("2026-02-01"), p, levels=LV)
    q = bd.lookup(pd.DataFrame({"category": ["Phones", "Phones", "Tablet"], "model": ["X", "Y", "Q"]}))
    assert q.shape == (3, 4)
    assert np.allclose(q[0], 0.9)                         # model X has 40 obs
    assert 0.5 < q[1].mean() < 0.9                        # Y has 5 obs -> category level mix
    assert np.allclose(q[2], q[1])                        # unknown category -> global == Phones here


def test_fit_ignores_offers_outside_window():
    o = _offers()
    o.loc[o.model == "X", "day"] = pd.Timestamp("2025-01-01")    # older than 90 days
    bd = BidDist.fit(o, pd.Timestamp("2026-02-01"), Params(n_bid_quantiles=4), levels=LV)
    q = bd.lookup(pd.DataFrame({"category": ["Phones"], "model": ["X"]}))
    assert not np.allclose(q[0], 0.9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bids.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve.bids'`

- [ ] **Step 3: Implement the bid distribution**

Create `src/reserve/bids.py`:

```python
"""M3: empirical distribution of bid / m-hat at the finest hierarchy level with enough bids."""
import numpy as np
import pandas as pd

from reserve.config import LEVELS, Params


class BidDist:
    def __init__(self, levels, qlevels, tables, global_q, min_obs):
        self.levels, self.qlevels, self.tables = levels, qlevels, tables
        self.global_q, self.min_obs = global_q, min_obs
        self.qcols = [f"q{i}" for i in range(len(qlevels))]

    @classmethod
    def fit(cls, offers_f: pd.DataFrame, cutoff: pd.Timestamp, p: Params,
            levels: list[list[str]] = LEVELS) -> "BidDist":
        start = cutoff - pd.Timedelta(days=p.bid_window_days)
        o = offers_f[(offers_f.day < cutoff) & (offers_f.day >= start)]
        o = o.assign(x=(o.offer_price / o.m).clip(0.2, 3.0))
        ql = (np.arange(p.n_bid_quantiles) + 0.5) / p.n_bid_quantiles
        qcols = [f"q{i}" for i in range(len(ql))]
        tables = []
        for keys in levels:
            t = o.groupby(keys).x.quantile(ql).unstack()
            t.columns = qcols
            t["n"] = o.groupby(keys).size()
            tables.append(t.reset_index())
        return cls(levels, ql, tables, np.quantile(o.x, ql), p.min_group_obs)

    def lookup(self, df: pd.DataFrame) -> np.ndarray:
        out = np.tile(self.global_q, (len(df), 1))
        for keys, t in zip(self.levels, self.tables):
            t = t[t.n >= self.min_obs]
            j = df[keys].reset_index(drop=True).merge(t, on=keys, how="left")
            ok = j["n"].notna().to_numpy()
            out[ok] = j[self.qcols].to_numpy()[ok]
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bids.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/reserve/bids.py tests/test_bids.py
git commit -m "feat: add hierarchical empirical bid-ratio distribution" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: B0 floor model (XGBoost AFT, interval-censored)

**Files:**
- Create: `src/reserve/floor_model.py`, `tests/test_floor_model.py`

**Interfaces:**
- Consumes: `floor_bounds` output (Task 3) merged with features and `m`, with the columns `lower, upper, consistent, m` plus `FLOOR_NUM` and `CAT_COLS`
- Produces:
  - `FLOOR_NUM`
  - `FloorModel().fit(train, valid) -> self`
  - `FloorModel.predict_ratio(df) -> np.ndarray` of the floor divided by m̂

- [ ] **Step 1: Write the failing test**

Create `tests/test_floor_model.py`:

```python
import numpy as np
import pandas as pd

from reserve.floor_model import FLOOR_NUM, FloorModel
from reserve.frame import CAT_COLS


def _bounds(n, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FLOOR_NUM})
    for c in CAT_COLS:
        df[c] = "X"
    df["grade"] = rng.choice(["A", "B"], n)
    df["m"] = 100.0
    floor = np.where(df.grade == "A", 90.0, 80.0)
    b = rng.uniform(60, 110, n)
    acc = b >= floor
    df["lower"] = np.where(acc, np.nan, b)
    df["upper"] = np.where(acc, b, floor * 1.02)
    df["consistent"] = True
    return df


def test_floor_model_recovers_grade_floors():
    fm = FloorModel().fit(_bounds(4000, 0), _bounds(1000, 1))
    test = _bounds(200, 2)
    r = fm.predict_ratio(test)
    assert abs(np.median(r[test.grade == "A"]) - 0.90) < 0.05
    assert abs(np.median(r[test.grade == "B"]) - 0.80) < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_floor_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve.floor_model'`

- [ ] **Step 3: Implement the floor model**

Create `src/reserve/floor_model.py`:

```python
"""B0: Assurant's implicit floor (as a ratio to m-hat) learned from interval-censored decisions."""
import numpy as np
import pandas as pd
import xgboost as xgb

from reserve.frame import CAT_COLS, apply_categories, fit_categories

FLOOR_NUM = ["age_days", "tau", "nb", "ub", "dow", "px_n_L3", "px_n_L5", "units_sold_w"]
XGB_PARAMS = {"objective": "survival:aft", "eval_metric": "aft-nloglik",
              "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 0.2,
              "tree_method": "hist", "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 20}


class FloorModel:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.cats = None
        self.booster = None

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        return apply_categories(df[FLOOR_NUM + CAT_COLS], self.cats)

    def _dmatrix(self, df: pd.DataFrame) -> xgb.DMatrix:
        d = xgb.DMatrix(self._X(df), enable_categorical=True)
        d.set_float_info("label_lower_bound", (df.lower / df.m).fillna(0.0).to_numpy(float))
        d.set_float_info("label_upper_bound", (df.upper / df.m).fillna(np.inf).to_numpy(float))
        return d

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> "FloorModel":
        train, valid = train[train.consistent], valid[valid.consistent]
        self.cats = fit_categories(train)
        self.booster = xgb.train({**XGB_PARAMS, "seed": self.seed}, self._dmatrix(train),
                                 num_boost_round=2000, evals=[(self._dmatrix(valid), "valid")],
                                 early_stopping_rounds=100, verbose_eval=False)
        return self

    def predict_ratio(self, df: pd.DataFrame) -> np.ndarray:
        d = xgb.DMatrix(self._X(df), enable_categorical=True)
        return self.booster.predict(d, iteration_range=(0, self.booster.best_iteration + 1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_floor_model.py -v`
Expected: 1 passed. If the medians are biased by more than 0.05, try `aft_loss_distribution_scale` values of 0.1 and 0.5. Keep whichever passes and record the choice in a comment. The scale is the noise in log(floor ratio); real floors move by about 5–10%, so 0.1–0.2 is the plausible range.

- [ ] **Step 5: Commit**

```bash
git add src/reserve/floor_model.py tests/test_floor_model.py
git commit -m "feat: add B0 interval-censored AFT floor model" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Phase 2 optimizer

**Files:**
- Create: `src/reserve/optimizer.py`, `tests/test_optimizer.py`

**Interfaces:**
- Consumes: `CounterCurve` (Task 8), `Params`
- Produces:
  - `price_grid(p) -> np.ndarray` (0.70 … 1.15)
  - `expected_value(m, W, xq, wh, g, curve) -> (V, sold, rev)`, each of shape `[N, G]` and expressed per unit
  - `optimize(df, xq, curve, p, b0_ratio=None, chunk=5000) -> DataFrame[g_star, reserve, exp_value, p_sell, exp_rev]`, aligned to `df.index`. `df` needs `m`, `W` and `wh`. On a tie it picks the highest `g`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_optimizer.py`:

```python
import numpy as np
import pandas as pd

from reserve.config import Params
from reserve.counter_curve import CounterCurve
from reserve.optimizer import expected_value, optimize, price_grid

ZERO = CounterCurve(np.array([0.0, 0.2]), {"__all__": np.array([0.0, 0.0])}, 0.2)
HALF = CounterCurve(np.array([0.0, 0.2]), {"__all__": np.array([0.5, 0.5])}, 0.2)


def _df(W):
    return pd.DataFrame({"m": [100.0], "W": [W], "wh": ["W1"]})


def test_price_grid():
    g = price_grid(Params())
    assert g[0] == 0.70 and g[-1] == 1.15 and len(g) == 46


def test_expected_value_counter_branch():
    V, S, R = expected_value(np.array([100.0]), np.array([80.0]), np.array([[0.9]]),
                             np.array(["W1"]), np.array([1.0]), HALF)
    assert np.isclose(V[0, 0], 0.5 * 100 + 0.5 * 80)
    assert np.isclose(S[0, 0], 0.5)
    assert np.isclose(R[0, 0], 50.0)


def test_optimize_sells_when_bid_beats_waiting():
    out = optimize(_df(W=50.0), np.full((1, 4), 0.9), ZERO, Params())
    assert np.isclose(out.reserve.iloc[0], 90.0)          # highest price that still accepts
    assert np.isclose(out.exp_value.iloc[0], 90.0)
    assert np.isclose(out.p_sell.iloc[0], 1.0)


def test_optimize_holds_when_waiting_beats_bid():
    out = optimize(_df(W=95.0), np.full((1, 4), 0.9), ZERO, Params())
    assert out.reserve.iloc[0] > 90.0
    assert np.isclose(out.exp_value.iloc[0], 95.0)


def test_trust_region_limits_reserve():
    out = optimize(_df(W=95.0), np.full((1, 4), 0.9), ZERO, Params(), b0_ratio=np.array([0.80]))
    assert out.g_star.iloc[0] <= 0.80 * 1.15 + 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_optimizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve.optimizer'`

- [ ] **Step 3: Implement the optimizer**

Create `src/reserve/optimizer.py`:

```python
"""Phase 2: choose r = m * g maximizing expected per-unit value of today's bids vs waiting."""
import numpy as np
import pandas as pd

from reserve.config import Params
from reserve.counter_curve import CounterCurve


def price_grid(p: Params) -> np.ndarray:
    return np.round(np.arange(p.grid_lo, p.grid_hi + p.grid_step / 2, p.grid_step), 4)


def expected_value(m, W, xq, wh, g, curve: CounterCurve):
    r = m[:, None] * g[None, :]                      # N, G
    b = (xq * m[:, None])[:, None, :]                # N, 1, Q
    rr = r[:, :, None]                               # N, G, 1
    accept = b >= rr
    pi = curve.pi((rr - b) / b, wh)
    rev = np.where(accept, b, pi * rr)
    sold = np.where(accept, 1.0, pi)
    val = rev + (1.0 - sold) * W[:, None, None]
    return val.mean(axis=2), sold.mean(axis=2), rev.mean(axis=2)


def optimize(df: pd.DataFrame, xq: np.ndarray, curve: CounterCurve, p: Params,
             b0_ratio=None, chunk: int = 5000) -> pd.DataFrame:
    g = price_grid(p)
    G = len(g)
    m = df.m.to_numpy(float)
    W = df.W.to_numpy(float)
    wh = df.wh.astype(str).to_numpy()
    b0 = None if b0_ratio is None else np.asarray(b0_ratio, float)
    n = len(df)
    cols = {k: np.empty(n) for k in ["g_star", "exp_value", "p_sell", "exp_rev"]}
    for s in range(0, n, chunk):
        e = slice(s, min(s + chunk, n))
        V, S, R = expected_value(m[e], W[e], xq[e], wh[e], g, curve)
        feas = np.broadcast_to(g >= p.salvage_floor_pct, V.shape).copy()
        if b0 is not None:
            lo, hi = b0[e, None] * (1 - p.trust_region), b0[e, None] * (1 + p.trust_region)
            feas &= (g[None, :] >= lo) & (g[None, :] <= hi)
        Vm = np.where(feas, V, -np.inf)
        idx = G - 1 - Vm[:, ::-1].argmax(axis=1)     # tie -> highest price
        none = ~feas.any(axis=1)
        if none.any():
            target = b0[e][none] if b0 is not None else np.ones(none.sum())
            idx[none] = np.abs(g[None, :] - target[:, None]).argmin(axis=1)
        rows = np.arange(len(idx))
        cols["g_star"][e] = g[idx]
        cols["exp_value"][e] = V[rows, idx]
        cols["p_sell"][e] = S[rows, idx]
        cols["exp_rev"][e] = R[rows, idx]
    out = pd.DataFrame(cols, index=df.index)
    out["reserve"] = m * out.g_star.to_numpy()
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_optimizer.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/reserve/optimizer.py tests/test_optimizer.py
git commit -m "feat: add Phase 2 grid optimizer with trust region" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Sibling monotonicity

**Files:**
- Create: `src/reserve/siblings.py`, `tests/test_siblings.py`

**Interfaces:**
- Consumes: `config.CAPACITY_ORDER` and `config.GRADE_ORDER`
- Produces: `enforce_monotone(df, value_col="reserve", weight_col="nb") -> df + reserve_mono`. `df` needs `model, grade, capacity, wh, day`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_siblings.py`:

```python
import numpy as np
import pandas as pd

from reserve.siblings import enforce_monotone

DAY = pd.Timestamp("2026-03-01")


def _df(rows):
    df = pd.DataFrame(rows)
    df["wh"], df["day"], df["nb"] = "W1", DAY, 1.0
    return df


def test_capacity_violation_is_pooled():
    df = _df([dict(model="M", grade="A", capacity="128GB", reserve=200.0),
              dict(model="M", grade="A", capacity="256GB", reserve=180.0)])
    out = enforce_monotone(df)
    assert np.allclose(out.reserve_mono, [190.0, 190.0])


def test_grade_violation_is_pooled_and_unknown_grade_untouched():
    df = _df([dict(model="M", grade="B", capacity="128GB", reserve=210.0),
              dict(model="M", grade="A", capacity="128GB", reserve=200.0),
              dict(model="M", grade="REC", capacity="128GB", reserve=50.0)])
    out = enforce_monotone(df).set_index("grade").reserve_mono
    assert np.isclose(out["B"], 205.0) and np.isclose(out["A"], 205.0)
    assert out["REC"] == 50.0


def test_already_monotone_unchanged():
    df = _df([dict(model="M", grade="A", capacity="128GB", reserve=180.0),
              dict(model="M", grade="A", capacity="256GB", reserve=200.0)])
    assert np.allclose(enforce_monotone(df).reserve_mono, [180.0, 200.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_siblings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve.siblings'`

- [ ] **Step 3: Implement siblings**

Create `src/reserve/siblings.py`:

```python
"""Isotonic projection so reserves rise with capacity and along the grade ladder."""
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from reserve.config import CAPACITY_ORDER, GRADE_ORDER

PASSES = [
    (["model", "grade", "wh", "day"], "capacity", CAPACITY_ORDER),
    (["model", "capacity", "wh", "day"], "grade", GRADE_ORDER),
]


def enforce_monotone(df: pd.DataFrame, value_col: str = "reserve", weight_col: str = "nb") -> pd.DataFrame:
    out = df.copy()
    out["reserve_mono"] = out[value_col].astype(float)
    w = out[weight_col].fillna(0.0) + 1.0
    for group_keys, col, order in PASSES:
        rank_of = {v: i for i, v in enumerate(order)}
        sub = out[out[col].isin(order)]
        sub = sub[sub.groupby(group_keys)[col].transform("nunique") >= 2]
        for _, g in sub.groupby(group_keys):
            x = g[col].map(rank_of).to_numpy(float)
            iso = IsotonicRegression(increasing=True)
            out.loc[g.index, "reserve_mono"] = iso.fit(x, g.reserve_mono, sample_weight=w[g.index]).predict(x)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_siblings.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/reserve/siblings.py tests/test_siblings.py
git commit -m "feat: enforce capacity and grade monotonicity on reserves" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Replay harness and metrics

**Files:**
- Create: `src/reserve/replay.py`, `src/reserve/metrics.py`, `tests/test_replay_metrics.py`

**Interfaces:**
- Consumes: `CounterCurve` (Task 8), `price_grid` (Task 11)
- Produces:
  - `replay_policy(ev, reserve, curve) -> ev + [reserve, sold, revenue, value, decision]`. `ev` needs `sku, wh, day, offer_price, qty, m, W`.
  - `replay_actual(ev, curve) -> ev + [...]`, the same columns. It uses `status` and `counter_price`, and `reserve` is NaN.
  - `summarize(res) -> dict`, with keys `revenue, units_sold, units_bid, sell_through, realization, total_value, n_offers, accept_rate, counter_rate, reject_rate`
  - `hindsight_reserve(ev, curve, p) -> DataFrame[sku, wh, day, reserve_oracle]`
  - `metrics.bootstrap_uplift(res_a, res_b, n_boot=1000, seed=0) -> dict[uplift, uplift_pct, ci_lo, ci_hi]`
  - `metrics.pareto_segment(offers, share=0.8) -> DataFrame[sku, wh, segment]`
  - `metrics.segment_table(results: dict[str, DataFrame], col) -> DataFrame`
  - `metrics.deviation_bucket(reserve, b0) -> pd.Categorical`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_replay_metrics.py`:

```python
import numpy as np
import pandas as pd

from reserve.config import Params
from reserve.counter_curve import CounterCurve
from reserve.metrics import bootstrap_uplift, deviation_bucket, pareto_segment, segment_table
from reserve.replay import hindsight_reserve, replay_actual, replay_policy, summarize

HALF = CounterCurve(np.array([0.0, 0.2]), {"__all__": np.array([0.5, 0.5])}, 0.2)
ZERO = CounterCurve(np.array([0.0, 0.2]), {"__all__": np.array([0.0, 0.0])}, 0.2)
DAY = pd.Timestamp("2026-03-02")


def _ev():
    return pd.DataFrame({"sku": "S", "wh": "W1", "day": DAY, "offer_price": [100.0, 90.0, 50.0],
                         "qty": [1.0, 2.0, 1.0], "m": 100.0, "W": 80.0,
                         "status": ["ACCEPTED", "COUNTERED", "NOT_ACCEPTED"],
                         "counter_price": [np.nan, 99.0, np.nan]})


def test_replay_policy_accept_counter_reject():
    r = replay_policy(_ev(), np.full(3, 95.0), HALF)
    assert list(r.decision) == ["ACCEPTED", "COUNTERED", "NOT_ACCEPTED"]
    assert np.allclose(r.sold, [1.0, 1.0, 0.0])
    assert np.allclose(r.revenue, [100.0, 95.0, 0.0])
    assert np.allclose(r.value, [100.0, 95.0 + 80.0, 80.0])


def test_replay_actual_uses_real_decisions():
    r = replay_actual(_ev(), HALF)
    assert np.allclose(r.revenue, [100.0, 2 * 0.5 * 99.0, 0.0])
    assert np.allclose(r.sold, [1.0, 1.0, 0.0])


def test_summarize():
    s = summarize(replay_policy(_ev(), np.full(3, 95.0), HALF))
    assert np.isclose(s["revenue"], 195.0)
    assert np.isclose(s["sell_through"], 2.0 / 4.0)
    assert np.isclose(s["realization"], 195.0 / 200.0)
    assert np.isclose(s["total_value"], 355.0)
    assert np.isclose(s["accept_rate"], 1 / 3)


def test_hindsight_picks_value_maximizing_highest_price():
    ev = pd.DataFrame({"sku": "S", "wh": "W1", "day": DAY, "offer_price": [100.0, 80.0],
                       "qty": 1.0, "m": 100.0, "W": 0.0})
    h = hindsight_reserve(ev, ZERO, Params())
    assert np.isclose(h.reserve_oracle.iloc[0], 80.0)


def test_bootstrap_constant_uplift():
    days = pd.date_range("2026-03-01", periods=5)
    b = pd.DataFrame({"day": days, "value": 100.0})
    a = b.assign(value=110.0)
    u = bootstrap_uplift(a, b, n_boot=200)
    assert np.isclose(u["uplift"], 50.0) and np.isclose(u["ci_lo"], 50.0) and np.isclose(u["ci_hi"], 50.0)
    assert np.isclose(u["uplift_pct"], 0.1)


def test_pareto_segment_top_share():
    o = pd.DataFrame({"sku": ["A"] * 8 + ["B", "C"], "wh": "W1"})
    seg = pareto_segment(o).set_index("sku").segment
    assert seg["A"] == "A" and seg["B"] == "BC" and seg["C"] == "BC"


def test_segment_table_and_deviation_bucket():
    r = replay_policy(_ev(), np.full(3, 95.0), HALF).assign(segment="A")
    t = segment_table({"P2": r}, "segment")
    assert set(t.columns) >= {"segment", "policy", "revenue", "sell_through", "total_value"}
    b = deviation_bucket(np.array([100.0, 107.0, 112.0, 130.0]), np.array([100.0] * 4))
    assert list(b.astype(str)) == ["<=5%", "5-10%", "10-15%", ">15%"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_replay_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve.replay'`

- [ ] **Step 3: Implement replay**

Create `src/reserve/replay.py`:

```python
"""Replay historical bids under a reserve policy. Accept if bid >= r, else counter at r."""
import numpy as np
import pandas as pd

from reserve.config import Params
from reserve.counter_curve import CounterCurve
from reserve.optimizer import price_grid


def _finish(ev: pd.DataFrame, reserve, sold, rev, decision) -> pd.DataFrame:
    unsold = ev.qty.to_numpy(float) - sold
    return ev.assign(reserve=reserve, sold=sold, revenue=rev,
                     value=rev + unsold * ev.W.to_numpy(float), decision=decision)


def replay_policy(ev: pd.DataFrame, reserve, curve: CounterCurve) -> pd.DataFrame:
    b = ev.offer_price.to_numpy(float)
    q = ev.qty.to_numpy(float)
    r = np.asarray(reserve, float)
    accept = b >= r
    gap = (r - b) / b
    pi = curve.pi(gap, ev.wh.astype(str).to_numpy())
    sold = q * np.where(accept, 1.0, pi)
    rev = q * np.where(accept, b, pi * r)
    decision = np.where(accept, "ACCEPTED", np.where(gap <= curve.max_gap, "COUNTERED", "NOT_ACCEPTED"))
    return _finish(ev, r, sold, rev, decision)


def replay_actual(ev: pd.DataFrame, curve: CounterCurve) -> pd.DataFrame:
    b = ev.offer_price.to_numpy(float)
    q = ev.qty.to_numpy(float)
    c = ev.counter_price.to_numpy(float)
    st = ev.status.to_numpy()
    is_c = st == "COUNTERED"
    gap = np.where(is_c & ~np.isnan(c), (c - b) / b, -1.0)
    pi = curve.pi(gap, ev.wh.astype(str).to_numpy())
    sold = q * np.select([st == "ACCEPTED", is_c], [1.0, pi], 0.0)
    rev = q * np.select([st == "ACCEPTED", is_c], [b, pi * np.nan_to_num(c)], 0.0)
    return _finish(ev, np.nan, sold, rev, st)


def summarize(res: pd.DataFrame) -> dict:
    rev, sold, bid = res.revenue.sum(), res.sold.sum(), res.qty.sum()
    return {
        "revenue": float(rev),
        "units_sold": float(sold),
        "units_bid": float(bid),
        "sell_through": float(sold / bid),
        "realization": float(rev / (res.sold * res.m).sum()),
        "total_value": float(res.value.sum()),
        "n_offers": int(len(res)),
        "accept_rate": float((res.decision == "ACCEPTED").mean()),
        "counter_rate": float((res.decision == "COUNTERED").mean()),
        "reject_rate": float((res.decision == "NOT_ACCEPTED").mean()),
    }


def hindsight_reserve(ev: pd.DataFrame, curve: CounterCurve, p: Params) -> pd.DataFrame:
    keys = ["sku", "wh", "day"]
    g = price_grid(p)
    per_g = []
    for gi in g:
        v = replay_policy(ev, ev.m.to_numpy(float) * gi, curve)
        per_g.append(v.groupby(keys).value.sum().rename(gi))
    tab = pd.concat(per_g, axis=1)
    best = tab.columns.to_numpy()[len(g) - 1 - tab.to_numpy()[:, ::-1].argmax(axis=1)]
    out = tab.index.to_frame(index=False).assign(g_oracle=best)
    m = ev.groupby(keys).m.first().reset_index()
    out = out.merge(m, on=keys)
    out["reserve_oracle"] = out.m * out.g_oracle
    return out[keys + ["reserve_oracle"]]
```

- [ ] **Step 4: Implement metrics**

Create `src/reserve/metrics.py`:

```python
"""Uplift confidence intervals, Pareto segments and breakdown tables."""
import numpy as np
import pandas as pd

from reserve.replay import summarize

DEV_BINS = [-1e-9, 0.05, 0.10, 0.15, np.inf]
DEV_LABELS = ["<=5%", "5-10%", "10-15%", ">15%"]


def bootstrap_uplift(res_a: pd.DataFrame, res_b: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    da = res_a.groupby("day").value.sum()
    db = res_b.groupby("day").value.sum().reindex(da.index, fill_value=0.0)
    diff = (da - db).to_numpy()
    rng = np.random.default_rng(seed)
    sums = np.array([diff[rng.integers(0, len(diff), len(diff))].sum() for _ in range(n_boot)])
    return {"uplift": float(diff.sum()), "uplift_pct": float(diff.sum() / db.sum()),
            "ci_lo": float(np.quantile(sums, 0.025)), "ci_hi": float(np.quantile(sums, 0.975))}


def pareto_segment(offers: pd.DataFrame, share: float = 0.8) -> pd.DataFrame:
    c = offers.groupby(["sku", "wh"]).size().sort_values(ascending=False)
    cum = c.cumsum() / c.sum()
    in_a = cum.shift(fill_value=0.0) < share
    return pd.DataFrame({"segment": np.where(in_a, "A", "BC")}, index=c.index).reset_index()


def segment_table(results: dict, col: str) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        for seg, part in res.groupby(col, observed=True):
            rows.append({col: seg, "policy": name, **summarize(part)})
    return pd.DataFrame(rows)


def deviation_bucket(reserve, b0) -> pd.Categorical:
    dev = np.abs(np.asarray(reserve, float) / np.asarray(b0, float) - 1.0)
    return pd.cut(dev, DEV_BINS, labels=DEV_LABELS)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_replay_metrics.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add src/reserve/replay.py src/reserve/metrics.py tests/test_replay_metrics.py
git commit -m "feat: add replay harness, hindsight oracle and uplift metrics" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: B1 discount rule

**Files:**
- Create: `src/reserve/rule_b1.py`, `tests/test_rule_b1.py`

**Interfaces:**
- Consumes: `replay_policy` (Task 13)
- Produces:
  - `tau_edges_from(df) -> np.ndarray` (2 tercile edges)
  - `bands(df, tau_edges) -> (age_band: np.ndarray, tau_band: np.ndarray)`
  - `tune_rule(ev, curve, tau_edges, d_grid=None) -> dict[(int, int), float]`
  - `apply_rule(df, table, tau_edges, default=0.05) -> np.ndarray` of reserves

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rule_b1.py`:

```python
import numpy as np
import pandas as pd

from reserve.counter_curve import CounterCurve
from reserve.rule_b1 import apply_rule, bands, tau_edges_from, tune_rule

ZERO = CounterCurve(np.array([0.0, 0.2]), {"__all__": np.array([0.0, 0.0])}, 0.2)


def test_bands():
    df = pd.DataFrame({"age_days": [0, 45, 75, 200], "tau": [1.0, 5.0, 20.0, 60.0]})
    a, t = bands(df, np.array([4.0, 30.0]))
    assert list(a) == [0, 1, 2, 3]
    assert list(t) == [0, 1, 1, 2]


def test_tau_edges_terciles():
    e = tau_edges_from(pd.DataFrame({"tau": np.arange(1.0, 10.0)}))
    assert len(e) == 2 and e[0] < e[1]


def test_tune_rule_picks_smallest_discount_that_clears():
    ev = pd.DataFrame({"sku": "S", "wh": "W1", "day": pd.Timestamp("2026-02-10"),
                       "offer_price": 90.0, "qty": 1.0, "m": 100.0, "W": 0.0,
                       "age_days": 10.0, "tau": 2.0})
    edges = np.array([5.0, 10.0])
    table = tune_rule(ev, ZERO, edges)
    assert table[(0, 0)] == 0.10
    r = apply_rule(pd.DataFrame({"m": [100.0, 100.0], "age_days": [10.0, 100.0], "tau": [2.0, 50.0]}),
                   table, edges, default=0.05)
    assert np.allclose(r, [90.0, 95.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rule_b1.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve.rule_b1'`

- [ ] **Step 3: Implement the rule**

Create `src/reserve/rule_b1.py`:

```python
"""B1: transparent rule r = m * (1 - d[age band, velocity band]), d tuned by replay."""
import numpy as np
import pandas as pd

from reserve.counter_curve import CounterCurve
from reserve.replay import replay_policy

AGE_EDGES = [-np.inf, 30, 60, 90, np.inf]
D_GRID = np.round(np.arange(0.0, 0.26, 0.01), 2)


def tau_edges_from(df: pd.DataFrame) -> np.ndarray:
    return np.quantile(df.tau, [1 / 3, 2 / 3])


def bands(df: pd.DataFrame, tau_edges) -> tuple[np.ndarray, np.ndarray]:
    age = pd.cut(df.age_days, AGE_EDGES, labels=False, right=False).to_numpy().astype(int)
    tau = np.digitize(df.tau.to_numpy(float), tau_edges)
    return age, tau


def tune_rule(ev: pd.DataFrame, curve: CounterCurve, tau_edges, d_grid=None) -> dict:
    d_grid = D_GRID if d_grid is None else d_grid
    a, t = bands(ev, tau_edges)
    table = {}
    for (ai, ti), cell in ev.assign(_a=a, _t=t).groupby(["_a", "_t"]):
        best_d, best_v = None, -np.inf
        for d in d_grid:                     # ascending: ties keep the smaller discount
            v = replay_policy(cell, cell.m.to_numpy(float) * (1 - d), curve).value.sum()
            if v > best_v:
                best_d, best_v = float(d), v
        table[(int(ai), int(ti))] = best_d
    return table


def apply_rule(df: pd.DataFrame, table: dict, tau_edges, default: float = 0.05) -> np.ndarray:
    a, t = bands(df, tau_edges)
    d = np.array([table.get((int(x), int(y)), default) for x, y in zip(a, t)])
    return df.m.to_numpy(float) * (1 - d)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rule_b1.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/reserve/rule_b1.py tests/test_rule_b1.py
git commit -m "feat: add B1 age/velocity discount rule tuned by replay" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 15: Pipeline, CLI and end-to-end synthetic test

**Files:**
- Create: `src/reserve/pipeline.py`, `scripts/run_poc.py`, `tests/test_pipeline.py`
- Modify: `tests/factories.py` (append `synthetic_world`)

**Interfaces:**
- Consumes: everything from Tasks 2–14
- Produces:
  - `Data` (dataclass: `offers, sales, product, matched, feats`)
  - `prepare_from_frames(offers, sales, product, p) -> Data`
  - `prepare(p, use_cache=True) -> Data`
  - `Fitted` (dataclass: `market, delta, rho, recall, curve, bids, floor, rule, tau_edges, segments, market_inner_mape`)
  - `enrich(feats, F, p, h_pct=None, delta_mult=1.0) -> feats + [log_m, m, W]`
  - `fit_models(D, cutoff, p) -> Fitted`
  - `score_groups(F, rows, p, h_pct=None, delta_mult=1.0) -> rows + [log_m, m, W, b0_ratio, reserve_b0, reserve_b1, p2_g_star, p2_reserve, p2_exp_value, p2_p_sell, p2_exp_rev, reserve_p2]`
  - `WindowResult` (dataclass: `results, ev, groups, fitted, market_test_mape, frontier, sensitivity`)
  - `run_window(D, cutoff, eval_end, p, extras=True) -> WindowResult`
  - `run_backtest(D, p, walk_forward=False) -> WindowResult`
  - `daily_reserve(D, date, p) -> DataFrame`

- [ ] **Step 1: Add the synthetic world factory**

Append to `tests/factories.py`:

```python
def synthetic_world(seed: int = 0):
    """Known ground truth: floor = 0.9 * true value; counters at the floor taken 60% of the time."""
    rng = np.random.default_rng(seed)
    skus = [(f"S{i}", f"M{i % 3}", ["128GB", "256GB"][i % 2], ["A", "B"][(i // 2) % 2], 150 + 40 * i)
            for i in range(10)]
    product = product_df([dict(sku=s, model=mo, capacity=c, grade=g) for s, mo, c, g, _ in skus])
    days = pd.date_range("2025-10-10", "2026-04-09", freq="D")
    offers, sales = [], []
    for s, _, _, _, v in skus:
        for wh in ["W1", "W2"]:
            for d in days:
                true_v = v * (1 - 0.0005 * (d - days[0]).days)
                for _ in range(rng.poisson(3)):
                    buyer = f"b{rng.integers(5)}"
                    b = round(true_v * rng.uniform(0.75, 1.05), 2)
                    floor = 0.9 * true_v
                    qty = float(rng.integers(1, 4))
                    ts = d + pd.Timedelta(hours=int(rng.integers(8, 18)))
                    if b >= floor:
                        offers.append(dict(buyer_id=buyer, sku=s, wh=wh, day=d, ts=ts, qty=qty,
                                           offer_price=b, status="ACCEPTED"))
                        sales.append(dict(buyer_id=buyer, sku=s, wh=wh, day=d,
                                          ts=ts + pd.Timedelta(hours=1), unit_price=b, units=qty))
                    elif b >= 0.8 * true_v:
                        c = round(floor, 2)
                        offers.append(dict(buyer_id=buyer, sku=s, wh=wh, day=d, ts=ts, qty=qty,
                                           offer_price=b, counter_price=c, status="COUNTERED"))
                        if rng.random() < 0.6:
                            sales.append(dict(buyer_id=buyer, sku=s, wh=wh, day=d,
                                              ts=ts + pd.Timedelta(hours=2), unit_price=c, units=qty))
                    else:
                        offers.append(dict(buyer_id=buyer, sku=s, wh=wh, day=d, ts=ts, qty=qty,
                                           offer_price=b, status="NOT_ACCEPTED"))
    return offers_df(offers), sales_df(sales), product
```

- [ ] **Step 2: Write the failing end-to-end tests**

Create `tests/test_pipeline.py`:

```python
import numpy as np
import pandas as pd
import pytest

from factories import synthetic_world
from reserve.config import Params
from reserve.pipeline import daily_reserve, prepare_from_frames, run_window


@pytest.fixture(scope="module")
def world():
    o, s, pr = synthetic_world(0)
    return prepare_from_frames(o, s, pr, Params())


def test_run_window_produces_all_policies(world):
    w = run_window(world, "2026-03-01", "2026-04-09", Params(), extras=False)
    assert set(w.results) == {"ACTUAL", "B0", "B1", "P2", "ORACLE"}
    for name in ["B0", "B1", "P2"]:
        assert np.isfinite(w.groups[f"reserve_{name.lower()}"]).all(), name
    agree = (w.results["B0"].decision == w.ev.status).mean()
    assert agree > 0.80                                   # B0 recovers the known 0.9 floor policy
    oracle = w.results["ORACLE"].value.sum()
    # Oracle is the per-group best on the price grid; P2 can sit slightly off-grid after
    # the isotonic projection, so allow 1% slack.
    assert oracle >= 0.99 * w.results["P2"].value.sum()


def test_daily_reserve_one_row_per_active_combo(world):
    out = daily_reserve(world, "2026-04-09", Params())
    assert len(out) == 20                                  # 10 SKUs x 2 WH
    assert out.reserve_p2.gt(0).all()
    assert {"sku", "wh", "reserve_p2", "reserve_b0", "m", "W", "p2_p_sell", "reason"} <= set(out.columns)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reserve.pipeline'`

- [ ] **Step 4: Implement the pipeline**

Create `src/reserve/pipeline.py`:

```python
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
from reserve.floor_model import FloorModel
from reserve.labels import floor_bounds, match_offers_to_sales, match_recall
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
```

- [ ] **Step 5: Add a minimal `explain` stub so `daily_reserve` imports**

Create `src/reserve/report.py`. Task 16 replaces it with the full module and keeps this function unchanged.

```python
"""Report generation (completed in Task 16)."""


def explain(row) -> str:
    return (f"Market ${row.m:,.0f}; ~{row.tau:.0f}d to sell, holding value ${row.W:,.0f}; "
            f"reserve ${row.reserve_p2:,.0f} ({row.reserve_p2 / row.m:.0%} of market, "
            f"{row.reserve_p2 / row.reserve_b0 - 1:+.0%} vs current floor); P(sell) ~{row.p2_p_sell:.0%}")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 2 passed. If the agreement assert fails, print `w.groups.b0_ratio.describe()` and compare it with the expected ≈ 0.9 / (m̂ / true value). The usual cause is too few valid rows for early stopping. Check that the `INNER_VALID_DAYS` window has offers.

- [ ] **Step 7: Write the CLI**

Create `scripts/run_poc.py`:

```python
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
```

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest -v`
Expected: all tests pass

- [ ] **Step 9: Commit**

```bash
git add src/reserve/pipeline.py src/reserve/report.py scripts/run_poc.py tests/factories.py tests/test_pipeline.py
git commit -m "feat: add pipeline, backtest windows, daily reserve and CLI" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 16: Report, real-data run, and success-criteria check

**Files:**
- Modify: `src/reserve/report.py` (full implementation; `explain` unchanged)
- Create: `tests/test_report.py`, `docs/results-2026-10-08.md` (generated, then committed)

**Interfaces:**
- Consumes: `WindowResult` (Task 15), the metrics functions (Task 13), and the config thresholds
- Produces:
  - `md_table(df, floatfmt="{:,.3f}") -> str`
  - `criteria(w) -> DataFrame[id, check, value, threshold, pass]`
  - `build_report(w, out_dir, tag) -> Path`, which writes `report_<tag>.md`, `frontier_<tag>.png` and `counter_curve_<tag>.png`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_report.py`:

```python
import pandas as pd
import pytest

from factories import synthetic_world
from reserve.config import Params
from reserve.pipeline import prepare_from_frames, run_window
from reserve.report import build_report, criteria, md_table


def test_md_table():
    s = md_table(pd.DataFrame({"a": [1.5], "b": ["x"]}))
    assert s.splitlines()[0] == "| a | b |"
    assert "1.500" in s


@pytest.fixture(scope="module")
def window():
    o, s, pr = synthetic_world(1)
    D = prepare_from_frames(o, s, pr, Params())
    return run_window(D, "2026-03-01", "2026-04-09", Params(), extras=True)


def test_criteria_rows(window):
    c = criteria(window)
    assert list(c.id) == ["F1", "F2", "F3", "S1", "S2", "S3", "S4"]
    assert c["pass"].dtype == bool


def test_build_report_writes_files(window, tmp_path):
    path = build_report(window, tmp_path, "t")
    text = path.read_text(encoding="utf-8")
    for section in ["## Success criteria", "## Policy comparison", "## Pareto segments",
                    "## Deviation from current floor", "## Frontier", "## Sensitivity",
                    "## Sample reserves"]:
        assert section in text, section
    assert (tmp_path / "frontier_t.png").exists()
    assert (tmp_path / "counter_curve_t.png").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_report'`

- [ ] **Step 3: Implement the report**

Replace `src/reserve/report.py` with:

```python
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
        ax.plot(c.grid, v, lw=2 if k == "__all__" else 0.8, label=k if k == "__all__" else None)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -v`
Expected: all tests pass

- [ ] **Step 5: Commit the report module**

```bash
git add src/reserve/report.py tests/test_report.py
git commit -m "feat: add backtest report with success criteria and charts" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Run the real-data backtest**

Run: `uv run python scripts/run_poc.py backtest`

Expected:
- It prints `report: .../output/report_single.md`.
- Runtime is roughly 5–20 minutes. Feature building dominates on the first run; later runs use the cache.

If it runs out of memory in `optimize`, lower `chunk` to 2000 in `score_groups` by passing `chunk=2000` to `optimize`.

- [ ] **Step 7: Run the walk-forward backtest**

Run: `uv run python scripts/run_poc.py backtest --walk-forward`
Expected: it prints `report: .../output/report_walkforward.md`

- [ ] **Step 8: Produce a daily reserve file**

Run: `uv run python scripts/run_poc.py daily --date 2026-04-09`
Expected: it prints `N reserves -> .../output/reserve_2026-04-09.csv`, with N roughly equal to the number of SKU × Warehouse combinations that had offers in the prior 90 days.

- [ ] **Step 9: Review the results against the spec and record them**

Read `output/report_single.md` and `output/report_walkforward.md`. For each criterion F1–S4, note pass or fail. Copy the single-window report to the docs folder:

```bash
cp output/report_single.md docs/results-2026-10-08.md
```

Append a `## Findings` section to `docs/results-2026-10-08.md` covering:
1. Which criteria failed and the most likely cause:
   - F1 low → AFT features too weak; inspect `b0_ratio` by grade
   - F3 off → recall correction or the counter curve
   - S1 fails → check whether the uplift sits in the `>15%` deviation bucket, which is untrustworthy
2. Walk-forward vs single-window uplift. Are they consistent?
3. The assumptions still pending from Assurant: h, salvage, grade ladder, and the conversion of accepted offers.

- [ ] **Step 10: Commit the results**

```bash
git add docs/results-2026-10-08.md
git commit -m "docs: record reserve-price POC backtest results" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Self-review notes (done while writing)

**Spec coverage**

| Spec item | Task |
|---|---|
| B0 AFT | Task 10 |
| B1 rule | Task 14 |
| M1 market value | Task 6 |
| τ (velocity) | Task 5 |
| δ, ρ, W | Task 7 |
| M3 bid distribution | Task 9 |
| M4 counter curve | Task 8 |
| Optimizer and trust region | Task 11 |
| Sibling monotonicity | Task 12 |
| Replay, ORACLE, bootstrap, segments, deviation buckets | Task 13 |
| Frontier and sensitivity | Task 15 (`run_window`) |
| Report and criteria | Task 16 |
| Daily CSV with reasons | Tasks 15 and 16 |
| Walk-forward | Task 15 (`run_backtest`) |
| ORDER_STATUS excluded | Task 2 (asserted in test) |

**Placeholders:** none. Every code step has full code.

**Name consistency:**
- `reserve_b0` / `reserve_b1` / `reserve_p2`, `p2_p_sell`, `p2_exp_value`, `b0_ratio`, `m`, `W`, `tau` and `nb` are used identically across Tasks 11, 13, 15 and 16.
- `CounterCurve.pi(gap, wh)` has the same signature in Tasks 8, 11 and 13.
- `Params` field names match `config.py`.

**Known limitations (also stated in the spec):**
- No stock levels are modelled.
- Bids are assumed not to react to a different reserve; this is mitigated by the trust region and the deviation buckets.
- Merging to the nearest later sale (`merge_asof`) can miss a matching sale when the same buyer has several sales of that SKU within 3 days.
- The two monotonicity passes (capacity, then grade) are applied in sequence, not jointly.
