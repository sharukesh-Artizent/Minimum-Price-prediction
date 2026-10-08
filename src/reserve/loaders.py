"""Read the raw CSVs into clean, typed frames. ORDER_STATUS is never read."""
import pandas as pd

from reserve import config

_NA = ["NULL", ""]
_DT_FMT = "%m/%d/%Y %H:%M"


def _read(path, cols: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=cols, dtype=str, na_values=_NA,
                       keep_default_na=False, encoding="utf-8", encoding_errors="ignore")


def load_offers(path=config.OFFERS_CSV) -> pd.DataFrame:
    raw = _read(path, ["OFFER_ID", "BUYER_ID", "SKU", "WAREHOUSE", "OFFER_QUANTITY",
                       "OFFER_PRICE", "COUNTER_PRICE", "OFFER_STATUS", "OFFER_CREATE_DATETIME"])
    df = pd.DataFrame({
        "offer_id": raw["OFFER_ID"],
        "buyer_id": raw["BUYER_ID"],
        "sku": raw["SKU"],
        "wh": raw["WAREHOUSE"],
        "qty": pd.to_numeric(raw["OFFER_QUANTITY"], errors="coerce"),
        "offer_price": pd.to_numeric(raw["OFFER_PRICE"], errors="coerce"),
        "counter_price": pd.to_numeric(raw["COUNTER_PRICE"], errors="coerce"),
        "status": raw["OFFER_STATUS"],
        "ts": pd.to_datetime(raw["OFFER_CREATE_DATETIME"], format=_DT_FMT, errors="coerce"),
    })
    df = df.dropna(subset=["ts", "offer_price", "qty", "sku", "wh", "buyer_id", "status"])
    df = df[(df.offer_price > 0) & (df.qty > 0)].copy()
    df["day"] = df.ts.dt.normalize()
    return df.reset_index(drop=True)


def load_sales(path=config.SALES_CSV) -> pd.DataFrame:
    raw = _read(path, ["BUYER_ID", "WAREHOUSE_CODE", "STATUS", "ORDER_TYPE", "DATE",
                       "PRODUCT_ID", "UNIT_PRICE", "SALES_QTY", "QUANTITY_CANCELLED"])
    raw = raw[(raw.ORDER_TYPE == "STOCKLIST") & (raw.STATUS != "CANCELED")]
    units = (pd.to_numeric(raw.SALES_QTY, errors="coerce").fillna(0)
             - pd.to_numeric(raw.QUANTITY_CANCELLED, errors="coerce").fillna(0))
    df = pd.DataFrame({
        "buyer_id": raw.BUYER_ID,
        "sku": raw.PRODUCT_ID,
        "wh": raw.WAREHOUSE_CODE,
        "unit_price": pd.to_numeric(raw.UNIT_PRICE, errors="coerce"),
        "units": units,
        "ts": pd.to_datetime(raw.DATE, format=_DT_FMT, errors="coerce"),
    })
    df = df.dropna(subset=["ts", "unit_price", "sku", "wh"])
    df = df[(df.unit_price > 0) & (df.units > 0)].copy()
    df["day"] = df.ts.dt.normalize()
    return df.reset_index(drop=True)


def load_product(path=config.PRODUCT_CSV) -> pd.DataFrame:
    raw = _read(path, ["SKU", "CATEGORY", "MANUFACTURER", "MODEL", "CAPACITY", "GRADE"])
    raw = raw.rename(columns={"SKU": "sku", "CATEGORY": "category", "MANUFACTURER": "manufacturer",
                              "MODEL": "model", "CAPACITY": "capacity", "GRADE": "grade"})
    cols = ["sku", "category", "manufacturer", "model", "capacity", "grade"]
    df = raw[cols].dropna(subset=["sku"]).groupby("sku", as_index=False).first()
    return df[cols].fillna("UNK")


def load_all(use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, fn in [("offers", load_offers), ("sales", load_sales), ("product", load_product)]:
        f = config.INTERIM / f"{name}.parquet"
        if use_cache and f.exists():
            frames.append(pd.read_parquet(f))
        else:
            df = fn()
            df.to_parquet(f, index=False)
            frames.append(df)
    return tuple(frames)
