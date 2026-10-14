"""Rule baselines the CLM has to beat (no training)."""
from __future__ import annotations

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
