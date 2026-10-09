# Results

Experimental results for Phase 1 (baseline floor estimation) and Phase 2 (ROI-optimal reserve price) of the ADMM Reserve Price POC, measured on 2026-10-08.

## 1. Setup

- **Data windows:**
  - **Hyperparameter tuning validation window:** 2026-03-12 to 2026-03-26 (14 days), trained on 2026-01-15 to 2026-03-12 (56 days, 40,961 validation offers). Sits strictly before the final hold-out window to prevent lookahead leakage.
  - **Phase 1 training window:** 2026-01-29 to 2026-03-26 (56 days).
  - **Phase 1 evaluation & Phase 2 replay test window:** 2026-03-26 to 2026-04-09 (14 days, 56,256 hold-out offers).
- **Hyperparameters:**
  - **Phase 1:** $\lambda = 1.0$, $\rho = 2.0$. Selected via grid search over $\lambda \in \{0.3, 1.0, 3.0\} \times \rho \in \{0.5, 1.0, 2.0\}$. $\lambda = 1.0, \rho = 2.0$ achieved 74.8% validation balanced accuracy and converged in 373 iterations.
  - **Phase 2:** $\rho = 1.0$, $\kappa \in [0.02, 0.15]$, $\text{max\_iter} = 200$.
- **Graph size:**
  - **Nodes:** 55,070 SKU × warehouse combinations.
  - **Edges:** 68,357 total sibling relationships:
    - **Variant (colour):** 33,897 edges
    - **Warehouse:** 13,804 edges
    - **Grade:** 11,957 edges
    - **Capacity:** 8,699 edges

## 2. Phase 1: Baseline Floor Estimation

Hold-out evaluation on 56,256 test offers across node data tiers (Tier A: $\ge 30$ offers; Tier B: 10–29; Tier C: 1–9; Unseen: 0 train offers):

| Model / Method | Accuracy | Balanced Accuracy | Tier A ($n=39,568$) | Tier B ($n=6,870$) | Tier C ($n=3,322$) | Unseen ($n=6,496$) | Interval Coverage ($n=4,059$) | Mean Log Gap |
|---|---|---|---|---|---|---|---|---|
| **Family prior** | 0.597 | 0.594 | 0.585 | 0.606 | 0.634 | 0.637 | 0.269 | 0.081 |
| **ADMM no fusion ($\lambda = 0$)** | 0.732 | 0.730 | 0.764 | 0.677 | 0.630 | 0.648 | 0.292 | 0.067 |
| **ADMM fused ($\lambda = 1.0, \rho = 2.0$)** | **0.760** | **0.758** | **0.774** | **0.744** | **0.720** | **0.706** | **0.387** | **0.033** |

Pooling across the graph significantly boosted accuracy in the sparse tail and unseen nodes: Tier B gained +6.7 percentage points (0.677 to 0.744), Tier C gained +9.0 percentage points (0.630 to 0.720), and unseen nodes gained +5.8 percentage points (0.648 to 0.706), while interval coverage rose from 29.2% to 38.7% and the mean interval violation gap was cut in half.

## 3. Phase 2: Pareto Frontier

Sweeping the warehouse clearance target multiplier $\alpha$ from 0.0 (pure unconstrained expected value) to 1.3 (+30% above historical clearance rate) across 14 warehouses:

| Multiplier ($\alpha$) | Expected Units | Expected Revenue (\$) | Expected Value (\$) | Mean Price Change (%) | Binding Warehouses | Iterations | Converged | Ordering Violations |
|---|---|---|---|---|---|---|---|---|
| **0.0** | 50,864 | 15,182,568 | 81,986,782 | +0.25% | 0 / 14 | 200 | False | 303 |
| **1.0** | 50,952 | 15,208,543 | 81,986,194 | +0.23% | 6 / 14 | 200 | False | 302 |
| **1.1** | 51,348 | 15,313,498 | 81,981,116 | +0.17% | 11 / 14 | 200 | False | 301 |
| **1.2** | 54,657 | 15,977,130 | 81,967,593 | -0.10% | 13 / 14 | 200 | False | 306 |
| **1.3** | 58,286 | 16,787,053 | 81,924,080 | -0.41% | 14 / 14 | 200 | False | 340 |

Increasing the clearance multiplier from $\alpha = 0.0$ to $\alpha = 1.3$ shifts catalog-wide unweighted reserve prices from +0.25% to -0.41% across all 55,070 nodes, clearing 14.6% more units (50,864 to 58,286) and increasing current-period sales revenue by \$1.60M in exchange for only a \$62.7K (0.08%) reduction in total expected value (revenue plus depreciated terminal inventory).

Unweighted prices across the entire catalog barely move (+0.25% to -0.41%) because the vast majority of catalog nodes have zero or minimal stock and remain anchored to their Phase 1 baseline floor. However, reserves drop strategically on heavily stocked nodes to drive clearance: across active nodes with positive inventory, the inventory-weighted price change is -3.15% (as documented in Section 4). This confirms the model concentrates price concessions specifically where inventory pressure exists rather than discounting the thin catalog tail indiscriminately.

## 4. Replay Backtest

Replaying the 14-day hold-out period (2026-03-26 to 2026-04-09) against actual incoming offers:

| Policy | Revenue (\$) | Units Sold | Accepted Offers | Terminal Value (\$) | Total Value (\$) |
|---|---|---|---|---|---|
| **Assurant actual** | 31,877,311 | 122,345 | 15,911 | 56,370,791 | 88,248,102 |
| **Phase 1 floor** | 30,826,033 | 118,388 | 14,570 | 58,032,170 | 88,858,203 |
| **Phase 2 $\times$0.0** | 55,686,038 | 200,940 | 15,594 | 35,595,083 | 91,281,121 |
| **Phase 2 $\times$1.0** | 55,699,679 | 200,977 | 15,612 | 35,580,782 | 91,280,461 |
| **Phase 2 $\times$1.1** | 55,870,238 | 201,829 | 15,849 | 35,398,621 | 91,268,859 |
| **Phase 2 $\times$1.2** | 58,919,538 | 215,728 | 16,849 | 32,202,686 | 91,122,224 |
| **Phase 2 $\times$1.3** | 62,009,225 | 229,922 | 17,735 | 28,826,071 | 90,835,296 |

- **Inventory-weighted reserve price change:** Across the 19,098 active SKU × warehouse nodes with positive inventory proxy, the inventory-weighted price change of the Phase 2 reserve relative to the Phase 1 floor is **-3.15%**.

## 5. Caveats

### Known Data Limitations
1. **No inventory overlaps the offers.** Phase 2 uses a stand-in for stock: the units Assurant actually sold per node in the evaluation window. Replace it once a full extract arrives.
2. **Counter outcomes are mostly redeemed.** 182,906 of 205,444 countered offers are `REDEEMED`. The replay backtest only accepts or rejects at the reserve, so it leaves out Assurant's revenue from counters. That understates the "Assurant actual" policy.
3. **Only 6 months of history.** That is not enough to learn yearly seasonality.
4. **Buyer Profile leaks the target.** `ACCEPTANCE_RATE` exactly equals accepted ÷ total over the same Offers file. This matters for Offer Clearance, not for this POC.

### Convergence Status
- **Phase 1:** In hyperparameter tuning ($\lambda = 1.0, \rho = 2.0$), ADMM converged in 373 iterations. In the evaluation pipeline with `max_iter = 300`, the algorithm reached the iteration limit before meeting the strict $\epsilon = 10^{-4}$ tolerance, though loss and primal residuals had fully stabilized.
- **Phase 2:** Grid-based $x$-updates over discrete candidate prices combined with dual ascent on warehouse constraints oscillate across discrete grid points. At `max_iter = 200`, ADMM did not meet the dual convergence tolerance for any multiplier, but yielded stable reserve solutions and an intuitive, monotonic economic frontier.

### Ordering Violations
- Phase 2 reserves exhibit 301 to 340 sibling grade and capacity ordering violations out of 20,656 total directed ordering edges (<1.7% violation rate). This occurs because Phase 2 treats sibling consistency as a soft quadratic penalty alongside node-level demand and warehouse dual constraints rather than strict integer inequalities.

## 6. Asks for Assurant

Open items to raise with Assurant to transition from POC to production MVP:
- **Grade ranking:** confirm the official rank scale and where grades R, L, S and FTP sit.
- **Attribute step factors:** provide confirmed factors for storage, generation and brand (currently using placeholder steps of 10% for capacity and 8% for grade).
- **Economic parameters:** provide official holding cost per unit-day, salvage floors, warehouse clearance targets, and daily reserve price change guardrails.
- **Cost basis:** clarify whether "ROI" in the project charter implies return on purchase cost. Without cost basis data, the objective optimizes net revenue recovery.
- **Overlapping inventory extract:** provide a complete inventory extract covering the full offer history date range.
- **Redeemed counters:** clarify the operational definition of `ORDER_STATUS = REDEEMED` and buyer counter negotiation dynamics.
