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


def _norm(name: str) -> str:
    """File name without extension, with case, spaces and punctuation ignored (uploads often drop '&')."""
    stem = os.path.splitext(name)[0]
    return "".join(ch for ch in stem.lower() if ch.isalnum())


def find_data_files(roots: list[str]) -> dict[str, str]:
    """First match of each required file under the given roots (searched recursively, in order).

    Accepts .csv or .xlsx (CSV preferred when both exist). Names are compared ignoring case,
    spaces and punctuation, so 'Sales Order Line items.csv' also matches 'Sales Order&Line items.csv'.
    Excel lock files ('~$...') are ignored.
    """
    wanted = {name: _norm(fname) for name, fname in FILES.items()}
    out: dict[str, str] = {}
    for root in roots:
        if not root or not os.path.isdir(root) or len(out) == len(FILES):
            continue
        for ext in ("csv", "xlsx"):
            paths = glob.glob(os.path.join(glob.escape(root), "**", f"*.{ext}"), recursive=True)
            for path in sorted(paths):
                base = os.path.basename(path)
                if base.startswith("~$"):
                    continue
                for name, key in wanted.items():
                    if name not in out and _norm(base) == key:
                        out[name] = path
    missing = [FILES[k] for k in FILES if k not in out]
    if missing:
        raise FileNotFoundError(f"could not find {missing} (.csv or .xlsx) under {roots}")
    return out


def _read(path, cols) -> pd.DataFrame:
    """CSV or Excel, every column as text."""
    if isinstance(path, str) and path.lower().endswith(".xlsx"):
        return pd.read_excel(path, dtype=str, usecols=cols)
    return pd.read_csv(path, dtype=str, usecols=cols, encoding_errors="ignore")


def _id(s: pd.Series) -> pd.Series:
    """Ids read from Excel can come back as '12345.0'; make them match the CSV form '12345'."""
    return s.str.strip().str.replace(r"\.0$", "", regex=True)


def _ts(s: pd.Series) -> pd.Series:
    """CSV timestamps are 'MM/DD/YYYY HH:MM'; Excel ones come back as ISO text. Accept both."""
    out = pd.to_datetime(s, format=_TS, errors="coerce")
    retry = out.isna() & s.notna()
    if retry.any():
        out[retry] = pd.to_datetime(s[retry], format="mixed", errors="coerce")
    return out.astype("datetime64[ns]")


def _day(s: pd.Series) -> pd.Series:
    return _ts(s).dt.normalize()


def load_offers(path) -> pd.DataFrame:
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
    df = _read(path, cols)
    out = pd.DataFrame({
        "offer_id": _id(df.OFFER_ID),
        "buyer_id": _id(df.BUYER_ID),
        "sku": _id(df.SKU),
        "wh": _id(df.WAREHOUSE),
        "qty": pd.to_numeric(df.OFFER_QUANTITY, errors="coerce").astype(float),
        "price": pd.to_numeric(df.OFFER_PRICE, errors="coerce").astype(float),
        "status": df.OFFER_STATUS,
        "ts": _ts(df.OFFER_CREATE_DATETIME),
    })
    out = out[out.status.isin(OFFER_STATUSES) & (out.price > 0) & out.ts.notna()].copy()
    out["day"] = out.ts.dt.normalize().astype("datetime64[ns]")
    out["accepted"] = out.status.eq("ACCEPTED")
    out["qty"] = out.qty.where(out.qty > 0, 1.0).astype(float)
    return out.reset_index(drop=True)


def load_sales(path) -> pd.DataFrame:
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
    df = _read(path, cols)
    units = (
        pd.to_numeric(df.SALES_QTY, errors="coerce").fillna(0)
        - pd.to_numeric(df.QUANTITY_CANCELLED, errors="coerce").fillna(0)
    ).astype(float)
    out = pd.DataFrame({
        "sku": _id(df.PRODUCT_ID),
        "wh": _id(df.WAREHOUSE_CODE),
        "status": df.STATUS,
        "order_type": df.ORDER_TYPE,
        "units": units,
        "unit_price": pd.to_numeric(df.UNIT_PRICE, errors="coerce").astype(float),
        "day": _day(df.DATE),
    })
    return out[(out.status != "CANCELED") & (out.units > 0) & out.day.notna()].reset_index(drop=True)


def load_products(path) -> pd.DataFrame:
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
    df = _read(path, list(cols))
    out = df.rename(columns=cols)[list(cols.values())].copy()
    out["sku"], out["wh"] = _id(out["sku"]), _id(out["wh"])
    for c in list(cols.values())[2:]:
        out[c] = out[c].fillna("unknown").str.strip().replace("", "unknown")
    out["family"] = out.manufacturer + "|" + out.model + "|" + out.capacity + "|" + out.grade
    return out.drop_duplicates(["sku", "wh"]).reset_index(drop=True)
