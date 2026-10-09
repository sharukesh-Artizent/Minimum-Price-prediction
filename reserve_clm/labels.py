"""Assurant's implicit daily floor, read from OFFER_STATUS.

On a SKU x warehouse x day with both outcomes, the floor sits between the highest bid
Assurant did not accept (countered or rejected) and the lowest bid it accepted.
"""
import numpy as np
import pandas as pd

from .config import TRAIN_END, VAL_END

GKEYS = ["sku", "wh", "day"]


def floor_intervals(offers: pd.DataFrame) -> pd.DataFrame:
    if offers.empty:
        return pd.DataFrame(columns=["sku", "wh", "day", "lo", "hi", "n_offers_day", "mid"])
    lo = offers[~offers.accepted].groupby(GKEYS).price.max().rename("lo")
    hi = offers[offers.accepted].groupby(GKEYS).price.min().rename("hi")
    n = offers.groupby(GKEYS).size().rename("n_offers_day")
    df = pd.concat([lo, hi], axis=1, join="inner").join(n).reset_index()
    if df.empty:
        return pd.DataFrame(columns=["sku", "wh", "day", "lo", "hi", "n_offers_day", "mid"])
    df = df[df.lo <= df.hi].copy()
    df["mid"] = (df.lo + df.hi) / 2
    return df.reset_index(drop=True)


def assign_split(day: pd.Series) -> np.ndarray:
    day = pd.to_datetime(day)
    return np.where(day < TRAIN_END, "train", np.where(day < VAL_END, "val", "test"))
