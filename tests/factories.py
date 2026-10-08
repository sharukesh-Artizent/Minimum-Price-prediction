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


def synthetic_world(seed: int = 0):
    """Known ground truth: floor = 0.9 * true value; counters at the floor taken 60% of the time."""
    rng = np.random.default_rng(seed)
    skus = [(f"S{i}", f"M{i % 3}", ["128GB", "256GB"][i % 2], ["A", "B"][(i // 2) % 2], 150 + 40 * i)
            for i in range(10)]
    product = product_df([dict(sku=s, model=mo, capacity=c, grade=g) for s, mo, c, g, _ in skus])
    days = pd.date_range("2025-10-10", "2026-04-09", freq="D")
    offers, sales = [], []
    for s, _, _, _, v in skus:
        for wh in ["W1", "W2"]:
            for d in days:
                true_v = v * (1 - 0.0005 * (d - days[0]).days)
                for _ in range(rng.poisson(3)):
                    buyer = f"b{rng.integers(5)}"
                    b = round(true_v * rng.uniform(0.75, 1.05), 2)
                    floor = 0.9 * true_v
                    qty = float(rng.integers(1, 4))
                    ts = d + pd.Timedelta(hours=int(rng.integers(8, 18)))
                    if b >= floor:
                        offers.append(dict(buyer_id=buyer, sku=s, wh=wh, day=d, ts=ts, qty=qty,
                                           offer_price=b, status="ACCEPTED"))
                        sales.append(dict(buyer_id=buyer, sku=s, wh=wh, day=d,
                                          ts=ts + pd.Timedelta(hours=1), unit_price=b, units=qty))
                    elif b >= 0.8 * true_v:
                        c = round(floor, 2)
                        offers.append(dict(buyer_id=buyer, sku=s, wh=wh, day=d, ts=ts, qty=qty,
                                           offer_price=b, counter_price=c, status="COUNTERED"))
                        if rng.random() < 0.6:
                            sales.append(dict(buyer_id=buyer, sku=s, wh=wh, day=d,
                                              ts=ts + pd.Timedelta(hours=2), unit_price=c, units=qty))
                    else:
                        offers.append(dict(buyer_id=buyer, sku=s, wh=wh, day=d, ts=ts, qty=qty,
                                           offer_price=b, status="NOT_ACCEPTED"))
    return offers_df(offers), sales_df(sales), product

