import io
import pandas as pd
import pytest

from reserve_clm.data import find_data_files, load_offers, load_products, load_sales

OFFERS = """OFFER_ID,BUYER_ID,SKU,WAREHOUSE,OFFER_QUANTITY,OFFER_PRICE,COUNTER_PRICE,OFFER_STATUS,ORDER_STATUS,OFFER_CREATE_DATETIME,TIMESTAMP,SUBMITTED_SHOPPING_CART_ID
1,70,A,W1,6,239,NULL,ACCEPTED,REDEEMED,10/20/2025 13:08,10/20/2025 16:32,1
2,71,A,W1,2,200,210,COUNTERED,NULL,10/20/2025 14:00,10/20/2025 16:32,2
3,72,A,W1,1,150,NULL,NOT_ACCEPTED,NULL,10/21/2025 09:00,x,3
4,73,A,W1,1,150,NULL,PENDING_ACCEPTANCE,NULL,10/21/2025 09:00,x,4
5,74,A,W1,1,0,NULL,ACCEPTED,NULL,10/21/2025 09:00,x,5
6,75,A,W1,,100,NULL,ACCEPTED,NULL,10/21/2025 10:00,x,6
"""
SALES = """SALE_ID,BUYER_ID,WAREHOUSE_CODE,CURRENCY_CODE,STATUS,ORDER_TYPE,DATE,SALE_COMPLETION_DATE,SHOPPING_CART_ID,PAYMENT_METHOD,SHIPPING_CARRIER,CPR_ORDER,PX_CONNECT_ORDER,PXN_BUYER_ID,HYLA_ORDER_NUMBER,HYLA_ORDER_STATUS,LINE_ITEM_ID,PRODUCT_ID,LPN,UNIT_PRICE,SALES_QTY,QUANTITY_SHIPPED,QUANTITY_BACKORDERED,QUANTITY_CANCELLED,TOTAL_PRICE
1,418,W1,USD,INVOICED,STOCKLIST,10/19/2025 11:26,NULL,NULL,WIRE,FEDEX,0,0,NULL,1,NULL,1,A,,433,6,6,0,2,2598
2,418,W1,USD,CANCELED,STOCKLIST,10/19/2025 11:26,NULL,NULL,WIRE,FEDEX,0,0,NULL,1,NULL,2,A,,433,6,6,0,0,2598
3,418,W1,USD,INVOICED,AUCTION,10/20/2025 11:26,NULL,NULL,WIRE,FEDEX,0,0,NULL,1,NULL,3,A,,400,1,1,0,1,400
"""
PRODUCTS = """SKU,WAREHOUSE,ITEM_DESCRIPTION,CATEGORY,PRODUCT_FAMILY,MANUFACTURER,MODEL,MODEL_NUMBER,CAPACITY,COLOR,CARRIER,LOCK_STATUS,LAUNCH_DATE,PROTOCOL,CHANNEL,INACTIVE,DEACTIVATED,UPDATE_DATETIME,GRADE_STANDARD,GRADE
A,W1,desc,Phones,fam,Apple,iPhone 13,X1,128GB,Blue,OTH,UNLOCKED,,GSM,UNASSIGNED,0,0,2024-04-10,DLS,B
B,W1,desc,Phones,fam,Apple,iPhone 13,X1,128GB,,OTH,UNLOCKED,,GSM,UNASSIGNED,0,0,2024-04-10,DLS,B
A,W1,desc_dup,Phones,fam,Apple,iPhone 13,X1,128GB,Red,OTH,UNLOCKED,,GSM,UNASSIGNED,0,0,2024-04-10,DLS,B
"""


@pytest.fixture
def files(tmp_path):
    (tmp_path / "x" / "Offers").mkdir(parents=True)
    (tmp_path / "x" / "Offers" / "Offers.csv").write_text(OFFERS)
    (tmp_path / "x" / "Sales Order&Line items.csv").write_text(SALES)
    (tmp_path / "Product Information.csv").write_text(PRODUCTS)
    return tmp_path


def test_find_data_files(files):
    found = find_data_files(["", str(files / "missing"), str(files)])
    assert found["offers"].endswith("Offers.csv") and found["sales"].endswith("items.csv")
    with pytest.raises(FileNotFoundError):
        find_data_files([str(files / "x" / "Offers")])


def test_load_offers_filters_and_types(files):
    o = load_offers(str(files / "x" / "Offers" / "Offers.csv"))
    assert o.offer_id.tolist() == ["1", "2", "3", "6"]      # pending and zero price dropped
    assert o.accepted.tolist() == [True, False, False, True]
    assert o.day.iloc[2] == pd.Timestamp("2025-10-21") and str(o.day.dtype) == "datetime64[ns]"
    assert o.price.tolist() == [239.0, 200.0, 150.0, 100.0]
    assert o.qty.tolist() == [6.0, 2.0, 1.0, 1.0]           # missing qty filled with 1.0
    assert str(o.price.dtype).startswith("float")
    assert str(o.qty.dtype).startswith("float")

    import io
    csv = OFFERS + "7,76,A,W1,0,120,NULL,ACCEPTED,NULL,10/21/2025 11:00,x,7\n"
    o2 = load_offers(io.StringIO(csv))
    assert o2.loc[o2.offer_id == "7", "qty"].iloc[0] == 1.0


def test_load_sales_nets_cancelled(files):
    s = load_sales(str(files / "x" / "Sales Order&Line items.csv"))
    assert len(s) == 1                                      # CANCELED row and 1-1=0 unit row dropped
    assert s.units.iloc[0] == 4 and s.order_type.iloc[0] == "STOCKLIST"
    assert str(s.units.dtype).startswith("float")
    assert str(s.unit_price.dtype).startswith("float")
    assert str(s.day.dtype) == "datetime64[ns]"


def test_load_products_family(files):
    p = load_products(str(files / "Product Information.csv"))
    assert len(p) == 2                                      # duplicate (A, W1) dropped
    assert p.family.tolist() == ["Apple|iPhone 13|128GB|B"] * 2
    assert p.color.tolist() == ["Blue", "unknown"]


def test_find_data_files_tolerates_renamed_uploads(tmp_path):
    (tmp_path / "Offers.csv").write_text(OFFERS)
    (tmp_path / "Product_Information.csv").write_text(PRODUCTS)
    (tmp_path / "Sales Order Line items.csv").write_text(SALES)
    (tmp_path / "Sales Order& Line Items - real time.csv").write_text(SALES)   # must not match
    found = find_data_files([str(tmp_path)])
    assert found["sales"].endswith("Sales Order Line items.csv")
    assert found["products"].endswith("Product_Information.csv")


def test_xlsx_inputs_with_excel_types(tmp_path):
    """Excel stores numbers and dates as real types; ids and timestamps must still match the CSV path."""
    o = pd.read_csv(io.StringIO(OFFERS), dtype=str)
    o["SKU"] = 12345.0                                                  # numeric SKU as Excel float
    o["OFFER_CREATE_DATETIME"] = pd.to_datetime(o.OFFER_CREATE_DATETIME, format="%m/%d/%Y %H:%M")
    o["OFFER_PRICE"] = pd.to_numeric(o.OFFER_PRICE)
    o.to_excel(tmp_path / "Offers.xlsx", index=False)
    s = pd.read_csv(io.StringIO(SALES), dtype=str)
    s["DATE"] = pd.to_datetime(s.DATE, format="%m/%d/%Y %H:%M")
    s.to_excel(tmp_path / "Sales Order&Line items.xlsx", index=False)
    pd.read_csv(io.StringIO(PRODUCTS), dtype=str).to_excel(tmp_path / "Product Information.xlsx", index=False)
    (tmp_path / "~$Offers.xlsx").write_text("lock file")                # Excel lock file must be ignored
    found = find_data_files([str(tmp_path)])
    assert found["offers"].endswith("Offers.xlsx") and not found["offers"].endswith("~$Offers.xlsx")
    off = load_offers(found["offers"])
    assert off.sku.unique().tolist() == ["12345"] and off.offer_id.tolist() == ["1", "2", "3", "6"]
    assert off.day.iloc[0] == pd.Timestamp("2025-10-20")
    sal = load_sales(found["sales"])
    assert len(sal) == 1 and sal.day.iloc[0] == pd.Timestamp("2025-10-19")
    assert len(load_products(found["products"])) == 2


def test_csv_preferred_over_xlsx(tmp_path):
    (tmp_path / "Offers.csv").write_text(OFFERS)
    pd.read_csv(io.StringIO(OFFERS), dtype=str).to_excel(tmp_path / "Offers.xlsx", index=False)
    (tmp_path / "Product Information.csv").write_text(PRODUCTS)
    (tmp_path / "Sales Order Line items.csv").write_text(SALES)
    assert find_data_files([str(tmp_path)])["offers"].endswith("Offers.csv")
