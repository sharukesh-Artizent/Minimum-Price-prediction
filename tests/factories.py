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
