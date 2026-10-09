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
    rho: float = 2.0                 # ADMM penalty
    k_pareto: float = 10.0           # node weight w_i = n_i / (n_i + k)
    temperature: float = 0.03        # logistic softness in log-price units (~3%)
    prior_weight: float = 0.01       # pull toward family median when data and edges are silent
    recency_tau_days: float = 14.0   # offer weight = exp(-age_days / tau)
    max_iter: int = 600
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
