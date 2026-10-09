# Reserve Price POC with CLM-8B fine-tuning — Design

**Date:** 2026-10-09
**Status:** agreed in chat with the project lead (decisions listed below)

## Goal

Predict a daily reserve (minimum) price band for each SKU × Warehouse × day by
fine-tuning CLM-8B (Qwen3-8B frozen reader + trained projection heads), and
measure the result against Assurant's historical behaviour. Deliver one
self-contained `.ipynb` that runs end to end on Colab or Kaggle.

## Decisions (from the project lead)

| # | Decision |
|---|---|
| D1 | No new data from Assurant. Use only the files in `Data/Data Files/`. |
| D2 | `OFFER_STATUS` is the only trusted outcome field. `ORDER_STATUS` (REDEEMED) is ignored. |
| D3 | Model = CLM-8B fine-tuning only. No zero-shot evaluation. Laya is a separate experiment. |
| D4 | Daily granularity (SKU × Warehouse × day). |
| D5 | Days with no SKU-level reference price are tested separately ("fallback" set). |
| D6 | Compute (GPU) is the lead's responsibility. The notebook must run on Colab or Kaggle. |
| D7 | A written plan precedes implementation; everything is tested locally first. |

## Data facts that shape the design (measured 2026-10-08/09)

- Offers: 531,783 rows, 2025-10-10 → 2026-04-09, 314 buyers, ~41K SKUs, 55K SKU × WH combos.
- Sales: same period, `PRODUCT_ID` = SKU (98.5% of offer SKUs match). Statuses include `CANCELED`.
- Product: one row per (SKU, WAREHOUSE); 100% of offer SKU × WH found. `LAUNCH_DATE` 97% empty (not used).
- Inventory: covers 2025-07-01 → 2025-08-27 only, no overlap with offers → **not used** (D1 prevents a re-extract).
- Buyer Profile: snapshot computed over the same offers (leaks labels) → **not used**.
- 17,453 SKU × WH × days have both an accepted and a non-accepted offer with a consistent floor (97.8% of two-sided days).
- Floor ÷ 30-day market reference: median 0.967, 5th–95th pct 0.85–1.05. Floor interval width: median 6% of price.

## Label

For each SKU × WH × day: `lo` = highest `NOT_ACCEPTED`/`COUNTERED` bid, `hi` = lowest
`ACCEPTED` bid. Keep days with both and `lo ≤ hi`. Assurant's floor lies in `[lo, hi]`.
The label therefore means "reproduce Assurant's current floor", not "optimal price".

## Reference price

Median (over the 30 days before the target day, exclusive) of daily median accepted
offer prices. SKU level if ≥ 3 accepted offers in the window, else product family
(manufacturer + model + capacity + grade) if ≥ 3, else none.
`ref_source ∈ {sku, family, none}`. `sku` = main set; `family` = fallback set (D5);
`none` = cannot be priced, counted only.

## Bands (CLM candidates)

14 bands of the ratio `reserve / reference`: `<0.80`, twelve 2.5% bands from 0.80 to 1.10,
`≥1.10`. Each band has an English description (the action-head text).
Training target = soft distribution over the bands the interval `[lo, hi]/ref` overlaps.

## State (CLM input)

Text rendered from features computed strictly before the target day:
product attributes; market reference + 7-vs-30-day trend; bids (count 7/30 days,
distinct bidders, units requested, median/top bid ÷ reference, accepted share);
sales (units 7/30 days, days since last sale, median stocklist sale price ÷ reference);
Assurant's previous floor at this SKU × WH (÷ reference, days ago); weekday.
Layout: context, blank line, question (same layout as `clm.schema.state_text`).

## Model

Qwen3-8B last-token embeddings (L2-normalised, tail-truncated to 2048 tokens, cached),
CLM projection heads warm-started from the released `CLM_v0.1-8B.pt`, trained with
CLM's bidirectional soft-target InfoNCE (same objective as `train/finetune.py --task choice`).
Early stopping on validation interval-hit rate. 3 seeds; report each, the mean, and the
seed-averaged ("ensemble") distribution.

Decoding: median band of the distribution → reserve = reference × band midpoint.
Also reported: expected ratio Σ p_k · mid_k.

## Splits

Train: day < 2026-02-01. Validation: February 2026. Test: day ≥ 2026-03-01.

## Baselines (rules, no training)

- B0 carry-forward: midpoint of the previous observed floor at this SKU × WH (÷ reference), else B1.
- B1 fixed ratio: reference × training median ratio.
- B2 majority band: reference × midpoint of the most frequent training band.

## Evaluation

1. Interval hit rate (reserve within `[lo, hi]`), too-high rate, too-low rate.
2. Price error vs interval midpoint (MAPE, median APE, signed bias).
3. Band accuracy: exact, within ±1 band, mean absolute band error.
4. Offer-level replay on sampled test-period SKU × WH × days: accept if bid ≥ reserve,
   compared with actual `OFFER_STATUS`: agreement, lost-sales rate, under-pricing rate,
   accepted value.
5. Calibration of the ensemble (confidence bins vs hit rate) and band ECE.
6. Breakdowns by brand, price tier, bid history.
All reported separately for the main (`sku`) and fallback (`family`) sets.

## Deliverable

`notebooks/reserve_price_clm.ipynb`, generated from tested modules in `reserve_clm/`
by `tools/build_notebook.py`; validated locally by executing it in smoke mode
(small CPU model, subsampled data).
