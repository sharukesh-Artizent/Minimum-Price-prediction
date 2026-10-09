"""Rule baselines the CLM has to beat (no training)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bands import BAND_MIDS, N_BANDS


def carry_forward_ratio(df: pd.DataFrame, fallback: float) -> np.ndarray:
    """B0: midpoint of Assurant's previous floor at this SKU x warehouse, else the fixed ratio."""
    r = (df["last_lo_ratio"] + df["last_hi_ratio"]) / 2
    return r.fillna(fallback).to_numpy(dtype=float)


def fixed_ratio(n: int, ratio: float) -> np.ndarray:
    """B1/B2: one ratio for every row."""
    return np.full(n, float(ratio))


def majority_band(true_band: np.ndarray) -> int:
    return int(np.bincount(np.asarray(true_band, dtype=int), minlength=N_BANDS).argmax())


def fit_baselines(train_df: pd.DataFrame) -> dict:
    """Fit baseline parameters from training data.

    Returns dict with fallback median ratio and majority band.
    """
    fallback = float(train_df["mid_ratio"].median()) if "mid_ratio" in train_df else 1.0
    maj = majority_band(train_df["true_band"].to_numpy()) if "true_band" in train_df else 0
    return {"fallback_ratio": fallback, "majority_band": maj}


def predict_baseline(df: pd.DataFrame, baseline: str, fitted: dict) -> np.ndarray:
    """Predict ratios for baseline B0, B1, or B2."""
    fallback = float(fitted.get("fallback_ratio", 1.0))
    if baseline in ("B0", "carry_forward"):
        return carry_forward_ratio(df, fallback)
    elif baseline in ("B1", "fixed_ratio"):
        return fixed_ratio(len(df), fallback)
    elif baseline in ("B2", "majority_band"):
        maj = int(fitted.get("majority_band", 0))
        return fixed_ratio(len(df), BAND_MIDS[maj])
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
