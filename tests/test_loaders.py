import csv

import numpy as np
import pandas as pd

from reserve.loaders import load_offers, load_product, load_sales

OFFER_HDR = ["OFFER_ID", "BUYER_ID", "SKU", "WAREHOUSE", "OFFER_QUANTITY", "OFFER_PRICE",
             "COUNTER_PRICE", "OFFER_STATUS", "ORDER_STATUS", "OFFER_CREATE_DATETIME",
             "TIMESTAMP", "SUBMITTED_SHOPPING_CART_ID"]
SALES_HDR = ["SALE_ID", "BUYER_ID", "WAREHOUSE_CODE", "CURRENCY_CODE", "STATUS", "ORDER_TYPE",
             "DATE", "SALE_COMPLETION_DATE", "SHOPPING_CART_ID", "PAYMENT_METHOD",
             "SHIPPING_CARRIER", "CPR_ORDER", "PX_CONNECT_ORDER", "PXN_BUYER_ID",
             "HYLA_ORDER_NUMBER", "HYLA_ORDER_STATUS", "LINE_ITEM_ID", "PRODUCT_ID", "LPN",
             "UNIT_PRICE", "SALES_QTY", "QUANTITY_SHIPPED", "QUANTITY_BACKORDERED",
             "QUANTITY_CANCELLED", "TOTAL_PRICE"]
PRODUCT_HDR = ["SKU", "WAREHOUSE", "ITEM_DESCRIPTION", "CATEGORY", "PRODUCT_FAMILY",
               "MANUFACTURER", "MODEL", "MODEL_NUMBER", "CAPACITY", "COLOR", "CARRIER",
               "LOCK_STATUS", "LAUNCH_DATE", "PROTOCOL", "CHANNEL", "INACTIVE", "DEACTIVATED",
               "UPDATE_DATETIME", "GRADE_STANDARD", "GRADE"]


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in header})


def test_load_offers_parses_and_drops_order_status(tmp_path):
    p = tmp_path / "o.csv"
    write_csv(p, OFFER_HDR, [
        dict(OFFER_ID="1", BUYER_ID="70", SKU="111", WAREHOUSE="W23", OFFER_QUANTITY="6",
             OFFER_PRICE="239", COUNTER_PRICE="NULL", OFFER_STATUS="ACCEPTED",
             ORDER_STATUS="REDEEMED", OFFER_CREATE_DATETIME="10/20/2025 13:08"),
        dict(OFFER_ID="2", BUYER_ID="70", SKU="111", WAREHOUSE="W23", OFFER_QUANTITY="2",
             OFFER_PRICE="200", COUNTER_PRICE="215", OFFER_STATUS="COUNTERED",
             ORDER_STATUS="NULL", OFFER_CREATE_DATETIME="1/5/2026 9:07"),
        dict(OFFER_ID="3", BUYER_ID="71", SKU="112", WAREHOUSE="TX1", OFFER_QUANTITY="1",
             OFFER_PRICE="0", COUNTER_PRICE="NULL", OFFER_STATUS="NOT_ACCEPTED",
             OFFER_CREATE_DATETIME="1/5/2026 9:07"),
        dict(OFFER_ID="4", BUYER_ID="71", SKU="112", WAREHOUSE="TX1", OFFER_QUANTITY="1",
             OFFER_PRICE="150", COUNTER_PRICE="NULL", OFFER_STATUS="NOT_ACCEPTED",
             OFFER_CREATE_DATETIME="garbage"),
    ])
    df = load_offers(p)
    assert list(df.offer_id) == ["1", "2"]
    assert not any("order" in c.lower() for c in df.columns)
    assert np.isnan(df.counter_price.iloc[0])
    assert df.counter_price.iloc[1] == 215.0
    assert df.day.iloc[1] == pd.Timestamp("2026-01-05")
    assert df.ts.iloc[1].hour == 9
    assert df.wh.iloc[0] == "W23"


def test_load_sales_keeps_only_stocklist_non_cancelled(tmp_path):
    p = tmp_path / "s.csv"
    base = dict(BUYER_ID="418", WAREHOUSE_CODE="TX1", DATE="10/19/2025 11:26",
                PRODUCT_ID="GLCP1", UNIT_PRICE="433")
    write_csv(p, SALES_HDR, [
        {**base, "STATUS": "INVOICED", "ORDER_TYPE": "STOCKLIST", "SALES_QTY": "6", "QUANTITY_CANCELLED": "0"},
        {**base, "STATUS": "INVOICED", "ORDER_TYPE": "AUCTION", "SALES_QTY": "6", "QUANTITY_CANCELLED": "0"},
        {**base, "STATUS": "CANCELED", "ORDER_TYPE": "STOCKLIST", "SALES_QTY": "6", "QUANTITY_CANCELLED": "0"},
        {**base, "STATUS": "INVOICED", "ORDER_TYPE": "STOCKLIST", "SALES_QTY": "2", "QUANTITY_CANCELLED": "2"},
    ])
    df = load_sales(p)
    assert len(df) == 1
    assert df.units.iloc[0] == 6.0
    assert df.sku.iloc[0] == "GLCP1"
    assert df.unit_price.iloc[0] == 433.0


def test_load_product_dedupes_and_fills(tmp_path):
    p = tmp_path / "p.csv"
    write_csv(p, PRODUCT_HDR, [
        dict(SKU="111", WAREHOUSE="W1", CATEGORY="Phones", MANUFACTURER="Apple", MODEL="iPhone 12", CAPACITY="128GB", GRADE=""),
        dict(SKU="111", WAREHOUSE="W2", CATEGORY="Phones", MANUFACTURER="Apple", MODEL="iPhone 12", CAPACITY="128GB", GRADE="A"),
        dict(SKU="112", WAREHOUSE="W1", CATEGORY="Phones", MANUFACTURER="Samsung", MODEL="S21", CAPACITY="256GB", GRADE=""),
    ])
    df = load_product(p).set_index("sku")
    assert len(df) == 2
    assert df.loc["111", "grade"] == "A"
    assert df.loc["112", "grade"] == "UNK"
    assert list(df.columns) == ["category", "manufacturer", "model", "capacity", "grade"]
