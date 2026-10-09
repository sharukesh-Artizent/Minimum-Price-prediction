"""Locate and load the three source files, keeping only what the POC trusts."""
from __future__ import annotations

import glob
import os

import pandas as pd

from .config import OFFER_STATUSES

FILES = {
    "offers": "Offers.csv",
    "products": "Product Information.csv",
    "sales": "Sales Order&Line items.csv",
}
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
    cols = [
        "OFFER_ID",
        "BUYER_ID",
        "SKU",
        "WAREHOUSE",
        "OFFER_QUANTITY",
        "OFFER_PRICE",
        "OFFER_STATUS",
        "OFFER_CREATE_DATETIME",
    ]
    df = pd.read_csv(path, dtype=str, usecols=cols, encoding_errors="ignore")
    out = pd.DataFrame({
        "offer_id": df.OFFER_ID,
        "buyer_id": df.BUYER_ID,
        "sku": df.SKU,
        "wh": df.WAREHOUSE,
        "qty": pd.to_numeric(df.OFFER_QUANTITY, errors="coerce").astype(float),
        "price": pd.to_numeric(df.OFFER_PRICE, errors="coerce").astype(float),
        "status": df.OFFER_STATUS,
        "ts": pd.to_datetime(df.OFFER_CREATE_DATETIME, format=_TS, errors="coerce").astype("datetime64[ns]"),
    })
    out = out[out.status.isin(OFFER_STATUSES) & (out.price > 0) & out.ts.notna()].copy()
    out["day"] = out.ts.dt.normalize().astype("datetime64[ns]")
    out["accepted"] = out.status.eq("ACCEPTED")
    out["qty"] = out.qty.where(out.qty > 0, 1.0).astype(float)
    return out.reset_index(drop=True)


def load_sales(path: str) -> pd.DataFrame:
    cols = [
        "WAREHOUSE_CODE",
        "STATUS",
        "ORDER_TYPE",
        "DATE",
        "PRODUCT_ID",
        "UNIT_PRICE",
        "SALES_QTY",
        "QUANTITY_CANCELLED",
    ]
    df = pd.read_csv(path, dtype=str, usecols=cols, encoding_errors="ignore")
    units = (
        pd.to_numeric(df.SALES_QTY, errors="coerce").fillna(0)
        - pd.to_numeric(df.QUANTITY_CANCELLED, errors="coerce").fillna(0)
    ).astype(float)
    out = pd.DataFrame({
        "sku": df.PRODUCT_ID,
        "wh": df.WAREHOUSE_CODE,
        "status": df.STATUS,
        "order_type": df.ORDER_TYPE,
        "units": units,
        "unit_price": pd.to_numeric(df.UNIT_PRICE, errors="coerce").astype(float),
        "day": _day(df.DATE),
    })
    return out[(out.status != "CANCELED") & (out.units > 0) & out.day.notna()].reset_index(drop=True)


def load_products(path: str) -> pd.DataFrame:
    cols = {
        "SKU": "sku",
        "WAREHOUSE": "wh",
        "MANUFACTURER": "manufacturer",
        "MODEL": "model",
        "CAPACITY": "capacity",
        "GRADE": "grade",
        "COLOR": "color",
        "CARRIER": "carrier",
        "LOCK_STATUS": "lock_status",
        "CATEGORY": "category",
    }
    df = pd.read_csv(path, dtype=str, usecols=list(cols), encoding_errors="ignore")
    out = df.rename(columns=cols)[list(cols.values())].copy()
    for c in list(cols.values())[2:]:
        out[c] = out[c].fillna("unknown").str.strip().replace("", "unknown")
    out["family"] = out.manufacturer + "|" + out.model + "|" + out.capacity + "|" + out.grade
    return out.drop_duplicates(["sku", "wh"]).reset_index(drop=True)
