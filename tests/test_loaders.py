import numpy as np

from reserve.loaders import load_offers, load_products, load_sales


def test_load_offers_keeps_decided_positive_offers(data_dir):
    offers = load_offers(data_dir)
    assert sorted(offers["offer_id"]) == ["1", "2", "3", "6"]
    assert set(offers["status"]) == {"ACCEPTED", "COUNTERED", "NOT_ACCEPTED"}


def test_load_offers_derives_label_date_and_log_bid(data_dir):
    offers = load_offers(data_dir).set_index("offer_id")
    assert offers.loc["1", "accepted"] == 1.0
    assert offers.loc["2", "accepted"] == 0.0
    assert offers.loc["1", "date"].strftime("%Y-%m-%d") == "2025-10-20"
    assert np.isclose(offers.loc["1", "log_bid"], np.log(100))


def test_load_products_one_row_per_sku(data_dir):
    products = load_products(data_dir)
    assert products["sku"].is_unique
    assert list(products.columns) == [
        "sku", "manufacturer", "model", "carrier", "lock_status", "capacity", "grade"]


def test_load_sales_drops_zero_price(data_dir):
    sales = load_sales(data_dir)
    assert len(sales) == 2
    assert set(sales["sku"]) == {"100", "200"}
