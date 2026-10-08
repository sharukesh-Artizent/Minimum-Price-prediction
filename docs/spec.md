# Reserve (Daily Minimum) Price POC — Design Spec

Date: 2026-10-08
Folder: `poc/Pareto`

## Goal

Recommend a daily reserve (minimum) price per SKU × Warehouse, and show with an offline backtest that it beats Assurant's current implicit floor on recovery value at equal or better sell-through.

## Why it is an optimization problem

The data has no reserve-price column. Two partial truths exist and are used:

1. **Assurant's current floor (baseline B0).** Each historical decision bounds Assurant's hidden floor for that SKU × Warehouse × day:
   - ACCEPTED at bid b → floor ≤ b
   - NOT_ACCEPTED at b → floor > b
   - COUNTERED at b with counter c → b < floor ≤ c

   In 97.8% of SKU-days that contain both accepted and non-accepted bids, the two sets do not overlap. The labels are intervals, so B0 is fitted with an XGBoost AFT model (`survival:aft`).
2. **Bids form a demand curve.** Every offer is a buyer saying "I'll pay at least b". Replaying past bids against a candidate reserve gives counterfactual revenue and sell-through.

## Data constraints (hard)

- Only local files under `Data/Data Files/` are available. There is no Snowflake access.
- `ORDER_STATUS` (REDEEMED) is **not used**. It is unreliable: NOT_ACCEPTED offers can also show REDEEMED.
- `OFFER_STATUS` and `COUNTER_PRICE` **are used**, as Assurant's decisions.
- Realized outcomes come from `Sales Order&Line items.csv`, filtered to `ORDER_TYPE == STOCKLIST` and `STATUS != CANCELED`.
- An offer counts as converted (`sale_match`) when the **same buyer bought the same SKU (PRODUCT_ID) within 3 days**, at a unit price within 1% of the offer price, or of the counter price for countered offers.
  - Observed rates: ACCEPTED 48.8%, COUNTERED 50.0%, NOT_ACCEPTED 1.0%.
  - The low NOT_ACCEPTED rate is the negative control that shows the link is real.
  - Match **recall** is below 100%. It is measured on ACCEPTED offers and divided out when estimating counter take-up.
- The inventory file covers only 2025-07-01 → 2025-08-27 and does not overlap the offer period. **Stock levels are not modelled.** Stock pressure is proxied by sales velocity and age.
- `LAUNCH_DATE` is about 97% empty. Age is proxied by days since the SKU × Warehouse was first seen.
- Offer data runs 2025-10-10 → 2026-04-09: 531,783 offers, 314 buyers, 41,295 SKUs, 55,121 SKU × Warehouse combinations, median 3 offers per combination.

## Approach

### Phase 1: baselines
- **B0**: Assurant's inferred floor, from the interval-censored AFT model. Its prediction is a ratio to market value.
- **B1**: `r = m̂ · (1 − d[age band, velocity band])`, with the discount table tuned on the inner-validation replay.

### Phase 2: ROI optimization (predict, then optimize)

| Component | What it does |
|---|---|
| M1 market value `m̂` | Hierarchical shrunk trailing mean of log sale price (naive), plus a LightGBM residual correction. LightGBM is used only if it beats naive on inner validation. |
| Velocity `τ` | Expected days to sell: `τ = 28 / (units sold in last 28d + 0.5)`, clipped to [1, 90]. |
| Depreciation `δ` | Median weekly log-price slope per (model, capacity, grade), converted to per day and clipped to [0, 0.01]. |
| Continuation value `W` | `max(m̂·ρ·(1−δ)^τ − h·m̂·τ, salvage·m̂)` |
| M3 bid distribution | Empirical quantiles (20) of `b / m̂` at the finest hierarchy level with at least 30 bids in the trailing 90 days. |
| M4 counter take-up `π(gap)` | Decreasing isotonic curve of `counter_taken` against `gap = (c − b)/b`, per warehouse (global fallback), divided by match recall. `π = 0` beyond the training P99 gap. |

Objective per unit, for candidate reserve `r = m̂·g`, `g ∈ [0.70, 1.15]` in steps of 0.01:

```
V(r) = E_b[ 1{b ≥ r}·b + 1{b < r}·(π(gap)·r + (1 − π(gap))·W) ]
```

Policy rules:
- **Counter price = reserve.** Counter-price optimization belongs to Offer Clearance, which is out of POC scope.
- **Constraints:** `g ≥ salvage (0.50)`, and the trust region `|r / B0 − 1| ≤ 0.15`.
- **Tie-break:** on equal value, take the higher price.
- **Sibling consistency:** isotonic projection so reserve is non-decreasing in capacity (same model, grade, warehouse, day) and along the grade ladder D < C < C+ < B < B+ < A < A+.

## Validation

**Time split.** Train runs from data start + 14-day warm-up to cutoff − 28 days. Inner validation is the last 28 days before cutoff. Test is the period from cutoff onward.
- Default cutoff: 2026-03-01, so the test period is 2026-03-01 → 2026-04-09.
- Optional weekly walk-forward over the test period.
- **No k-fold.**

**Layer 1: component checks**
- M1 MAPE against naive.
- M4 isotonic curve plotted.
- M3 coverage implicit in replay.

**Layer 2: replay of test offers under each policy**
- Policies: ACTUAL (Assurant's real decisions), B0, B1, P2, and ORACLE (hindsight best `g` per group, an upper bound).
- Metrics: revenue, units sold, sell-through (units sold / units bid), price realization (revenue / Σ sold·m̂), total value (revenue + unsold·W), and the accept / counter / reject mix.

**Layer 3: trust checks**
- Segment breakdown: Pareto A (SKU × Warehouse combinations covering the top 80% of train offers) vs BC, grade, warehouse.
- Deviation buckets `|r/B0 − 1|`: ≤ 5%, 5–10%, 10–15%, > 15%.
- Day-block bootstrap 95% CI of the P2 − B0 uplift.
- Sensitivity: h × {0.5, 1.5}, δ × {0.5, 1.5}.
- Frontier: revenue vs sell-through for h ∈ {0, 0.0005, 0.001, 0.002, 0.005, 0.01}.

## Success criteria

| ID | Check | Pass if |
|---|---|---|
| F1 | B0-implied decisions agree with actual OFFER_STATUS | ≥ 80% |
| F2 | Replay revenue B0 / ACTUAL | within ±5% |
| F3 | Replay ACTUAL revenue × recall vs matched-sale revenue on test | within ±10% |
| S1 | P2 total-value uplift over B0, segment A | 95% CI lower bound > 0 |
| S2 | P2 sell-through | not more than 2 points below B0 |
| S3 | M1 test MAPE | ≤ naive MAPE |
| S4 | Sign of P2 vs B0 uplift | unchanged under all four sensitivity settings |

## Assumptions to confirm with Assurant

- **Holding cost h = 0.1% of m̂ per unit per day.** This is a placeholder.
- **Salvage floor = 50% of m̂.**
- **Grade ladder** D < C < C+ < B < B+ < A < A+. The other grades (N, REC, R, L, UNK) are left unconstrained.
- **Accepted offers convert.** The replay assumes every accepted offer completes, for every policy alike.
- **Fixed bids.** Buyers' bids do not react to a different reserve. This is why the trust region and the deviation buckets exist.

## Deliverables

- `output/report.md` with the tables and charts above, copied to `docs/results-<date>.md`.
- `output/reserve_<date>.csv`: one reserve per active SKU × Warehouse, with B0, m̂, W, P(sell) and a plain-English reason.
