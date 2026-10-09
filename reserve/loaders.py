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
        "counter_price": pd.to_numeric(df["COUNTER_PRICE"], errors="coerce"),
        "status": df["OFFER_STATUS"],
        "order_status": df["ORDER_STATUS"].fillna("").str.strip(),
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
