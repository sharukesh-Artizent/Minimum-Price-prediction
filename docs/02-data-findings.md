# Data Findings

All figures come from the extracts in `Data/Data Files/` and were measured on 2026-10-08.

## Files

| File | Rows | Coverage | Notes |
|---|---|---|---|
| Offers/Offers.csv | 531,784 | 2025-10-10 → 2026-04-09 | 314 buyers, 41,295 SKUs |
| Product/Product Information.csv | 335,820 | — | one row per SKU × warehouse; every offered SKU is present |
| Sales/Historical Sales/Sales Order&Line items.csv | 645,101 | 2025-10-10 → 2026-04-09 | `PRODUCT_ID` uses the same IDs as product `SKU` (67,653 of 67,717 match) |
| Inventory/Inventory.csv | 1,048,575 | **2025-07-01 → 2025-08-27 only** | cut off at Excel's row limit; **does not overlap the offers** |
| Buyer Profile/Buyer Profile.csv | 2,597 | snapshot | used by Offer Clearance, not by the reserve price |

## Offer labels

| `OFFER_STATUS` | Count | Share |
|---|---|---|
| ACCEPTED | 284,754 | 53.5% |
| COUNTERED | 205,444 | 38.6% |
| NOT_ACCEPTED | 40,706 | 7.7% |
| PENDING_ACCEPTANCE | 879 | 0.2% (dropped) |

A counter means the bid was below what Assurant would accept. For floor estimation, COUNTERED
and NOT_ACCEPTED are both treated as "below floor".

## Assurant behaves as if it has a floor

On the 17,837 SKU × warehouse days that had both an accepted offer and a non-accepted one,
**97.8%** were consistent: every non-accepted bid was at or below every accepted bid. So a
latent floor exists. Phase 1 models it with a logistic threshold per node.

## Data is sparse per node

- There are 55,121 SKU × warehouse combinations with offers (55,070 after cleaning). The median
  combination has **3 offers in 6 months**; the 90th percentile has 28.
- On average there are **1.14 offers per SKU × warehouse per day**.
- 15,560 combinations (28%) account for 80% of offers. Only 5,084 have at least 30 offers, and
  they cover 48% of offers.

This is why per-node models fail and pooling across siblings (the ADMM graph) is needed.

## Graph structure available

- The 41,295 offered SKUs belong to **1,988 model families** (manufacturer, model, carrier, lock
  status). Each family has a median of 5 variants and a 90th percentile of 19.
- 22,268 SKUs are stocked in more than one warehouse.
- Grades: B+ (34K rows), C, B, A+, C+, R, D, L, C-, B-, A-, S, AA+, A, FTP. R, L, S and FTP
  have no confirmed rank.
- Capacities need normalising: 1TB, 1024GB and 1000GB are the same thing.
- The graph built for the POC has 55,070 nodes and 68,357 edges: 33,897 variant (colour),
  13,804 warehouse, 11,957 grade, 8,699 capacity.

## Known limitations that affect the POC

1. **No inventory overlaps the offers.** Phase 2 uses a stand-in for stock: the units Assurant
   actually sold per node in the evaluation window. Replace it once a full extract arrives.
2. **Counter outcomes are mostly redeemed.** 182,906 of 205,444 countered offers are
   `REDEEMED`. The replay backtest only accepts or rejects at the reserve, so it leaves out
   Assurant's revenue from counters. That understates the "Assurant actual" policy.
3. **Only 6 months of history.** That is not enough to learn yearly seasonality.
4. **Buyer Profile leaks the target.** `ACCEPTANCE_RATE` exactly equals accepted ÷ total over
   the same Offers file. This matters for Offer Clearance, not for this POC.
