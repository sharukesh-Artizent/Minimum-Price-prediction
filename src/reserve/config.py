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
