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
