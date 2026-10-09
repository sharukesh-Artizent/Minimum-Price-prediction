# Results

Experimental results for Phase 1 (baseline floor estimation) and Phase 2 (ROI-optimal reserve price) of the ADMM Reserve Price POC, measured on 2026-10-08.

## 1. Setup

- **Data windows:**
  - **Hyperparameter tuning validation window:** 2026-03-12 to 2026-03-26 (14 days), trained on 2026-01-15 to 2026-03-12 (56 days, 40,961 validation offers). Sits strictly before the final hold-out window to prevent lookahead leakage.
  - **Phase 1 training window:** 2026-01-29 to 2026-03-26 (56 days).
  - **Phase 1 evaluation & Phase 2 replay test window:** 2026-03-26 to 2026-04-09 (14 days, 56,256 hold-out offers).
- **Hyperparameters:**
  - **Phase 1:** $\lambda = 1.0$, $\rho = 2.0$. Selected via grid search over $\lambda \in \{0.3, 1.0, 3.0\} \times \rho \in \{0.5, 1.0, 2.0\}$. With `max_iter = 600`, Phase 1 ADMM converges in 374 iterations ($\epsilon = 10^{-4}$ tolerance). On the separate tuning validation window, it converged in 373 iterations.
  - **Phase 2:** $\rho = 1.0$, $\kappa \in [0.02, 0.15]$, $\text{max\_iter} = 200$.
- **Graph size:**
  - **Nodes:** 55,070 SKU × warehouse combinations.
  - **Edges:** 68,357 total sibling relationships:
    - **Variant (colour):** 33,897 edges
    - **Warehouse:** 13,804 edges
    - **Grade:** 11,957 edges
    - **Capacity:** 8,699 edges

## 2. Phase 1: Baseline Floor Estimation

Hold-out evaluation on 56,256 test offers across node data tiers. Tiers are defined by cumulative offer volume share within the 56-day training window (Tier A: top 80% volume, covering 3 to 100+ offers; Tier B: next 15%, covering 1 to 3 offers; Tier C: tail 5%, covering 1 offer; Unseen: 0 offers in the train window):

| Model / Method | Accuracy | Balanced Accuracy | Tier A ($n=39,568$) | Tier B ($n=6,870$) | Tier C ($n=3,322$) | Unseen ($n=6,496$) | Interval Coverage ($n=4,059$) | Mean Log Gap |
|---|---|---|---|---|---|---|---|---|
| **Family prior** | 0.597 | 0.594 | 0.585 | 0.606 | 0.634 | 0.637 | 0.269 | 0.081 |
| **ADMM no fusion ($\lambda = 0$)** | 0.732 | 0.730 | 0.764 | 0.677 | 0.630 | 0.648 | 0.292 | 0.067 |
| **ADMM fused ($\lambda = 1.0, \rho = 2.0$)** | **0.760** | **0.758** | **0.774** | **0.744** | **0.720** | **0.706** | **0.388** | **0.033** |

Pooling across the graph significantly boosted accuracy in the sparse tail and unseen nodes: Tier B gained +6.7 percentage points (0.677 to 0.744), Tier C gained +9.0 percentage points (0.630 to 0.720), and unseen nodes gained +5.8 percentage points (0.648 to 0.706), while interval coverage rose from 29.2% to 38.8% and the mean interval violation gap was cut in half.

## 3. Phase 2: Pareto Frontier

Sweeping the warehouse clearance target multiplier $\alpha$ from 0.0 (pure unconstrained expected value) to 1.3 (+30% above the model's expected clearance units at the Phase-1 floor) across the 17 warehouses (with up to 14 binding at $\alpha = 1.3$):

| Multiplier ($\alpha$) | Expected Units | Expected Revenue (\$) | Expected Value (\$) | Mean Price Change (%) | Binding Warehouses | Iterations | Converged | Ordering Violations |
|---|---|---|---|---|---|---|---|---|
| **0.0** | 50,864 | 15,182,568 | 81,986,782 | +0.25% | 0 / 17 | 200 | False | 303 |
| **1.0** | 50,952 | 15,208,543 | 81,986,194 | +0.23% | 6 / 17 | 200 | False | 302 |
| **1.1** | 51,348 | 15,313,498 | 81,981,116 | +0.17% | 11 / 17 | 200 | False | 301 |
| **1.2** | 54,657 | 15,977,130 | 81,967,593 | -0.10% | 13 / 17 | 200 | False | 306 |
| **1.3** | 58,286 | 16,787,053 | 81,924,080 | -0.41% | 14 / 17 | 200 | False | 340 |

Increasing the clearance multiplier from $\alpha = 0.0$ to $\alpha = 1.3$ shifts catalog-wide unweighted reserve prices from +0.25% to -0.41% across all 55,070 nodes, clearing 14.6% more units (50,864 to 58,286) and increasing current-period sales revenue by \$1.60M in exchange for only a \$62.7K (0.08%) reduction in total expected value (revenue plus depreciated terminal inventory).

Unweighted prices across the entire catalog barely move (+0.25% to -0.41%) because the vast majority of catalog nodes have zero or minimal stock and remain anchored to their Phase 1 baseline floor. However, reserves drop strategically on heavily stocked nodes to drive clearance: across active nodes with positive inventory, the inventory-weighted price change is -3.15% (as documented in Section 4). This confirms the model concentrates price concessions specifically where inventory pressure exists rather than discounting the thin catalog tail indiscriminately.

Note that the flatness of the frontier (expected value drops only 0.08% for a 14.6% volume lift) is largely driven by the placeholder holding cost (0.1%/day) and depreciation (~0.2%/day), which make waiting relatively inexpensive. The shape of the frontier reflects these assumptions rather than calibrated empirical holding costs.

## 4. Replay Backtest

### Baseline Replay (Unscaled Historical Sales Proxy)

Replaying the 14-day hold-out period (2026-03-26 to 2026-04-09) against actual incoming offers using total historical sales (329,598 units) as an unscaled inventory proxy:

| Policy | Revenue (\$) | Units Sold | Accepted Offers | Terminal Value (\$) | Total Value (\$) |
|---|---|---|---|---|---|
| **Assurant actual** | 31,877,311 | 122,345 | 15,911 | 56,362,746 | 88,240,057 |
| **Phase 1 floor** | 30,828,698 | 118,386 | 14,574 | 58,020,144 | 88,848,842 |
| **Phase 2 $\times$0.0** | 55,677,057 | 200,985 | 15,582 | 35,591,927 | 91,268,984 |
| **Phase 2 $\times$1.0** | 55,692,690 | 201,028 | 15,602 | 35,575,453 | 91,268,143 |
| **Phase 2 $\times$1.1** | 55,855,469 | 201,851 | 15,831 | 35,401,194 | 91,256,663 |
| **Phase 2 $\times$1.2** | 58,871,566 | 215,649 | 16,840 | 32,244,343 | 91,115,909 |
| **Phase 2 $\times$1.3** | 61,934,848 | 229,711 | 17,717 | 28,895,665 | 90,830,513 |

- **Inventory-weighted reserve price change:** Across the 19,098 active SKU × warehouse nodes with positive inventory proxy, the inventory-weighted price change of the Phase 2 reserve relative to the Phase 1 floor is **-3.15%**.

#### Why the Baseline Numbers are Artificially Inflated
The apparent headline revenue gain ($55.7M vs $31.9M) reflects three structural artifacts of the baseline setup:
1. **Stock stand-in is overstated:** Inventory was proxied by total historical sales (329K units), which includes non-offer channels (enterprise contracts, direct retail). Assurant's accepted offer volume in this window was only 122K units. The unscaled replay allowed Phase 2 to "sell" inventory the offer channel never possessed.
2. **Counters are ignored:** The baseline credited Assurant only for bids marked `ACCEPTED`. In reality, Assurant countered heavily, and ~89% of those counter-offers were redeemed at an average +4.1% markup over the buyer's bid. The baseline thus discarded substantial revenue Assurant actually earned.
3. **Leftover stock valuation:** Unsold units were marked at depreciated expected bid value, meaning artificial inventory volume drove terminal value.

---

### Fair Replay Comparison (Counters Credited & Stock Scaled)

To investigate performance under tighter constraints, a fair backtest was constructed:
1. **Crediting redeemed counters:** Assurant's decisions credit offers where `status == "COUNTERED"` and `order_status == "REDEEMED"` at the transacted counter price (`COUNTER_PRICE`).
2. **Stock scaling factor (37.12%):** Available stock per node was scaled by 37.12% (the ratio of the unscaled baseline replay's accepted units to total sales proxy, 122,345 ÷ 329,598). **Important:** This is a heuristic scaling factor derived from the baseline replay itself to bring total stock to the scale of observed offer activity; it is *not* an empirical measurement of the offer channel's true share of company sales. The reported \$31.01M for "Assurant actual" is performance under this shrunken stock constraint, not Assurant's true offer revenue.

| Policy | Revenue (\$) | Units Sold | Accepted Offers | Terminal Value (\$) | Total Value (\$) |
|---|---|---|---|---|---|
| **Assurant actual (fair)** | 31,006,547 | 107,088 | 18,088 | 2,684,991 | 33,691,538 |
| **Phase 1 floor** | 13,799,951 | 52,165 | 11,878 | 19,314,096 | 33,114,047 |
| **Phase 2 $\times$0.0** | 23,054,274 | 82,971 | 11,917 | 10,969,457 | 34,023,730 |
| **Phase 2 $\times$1.0** | 23,057,103 | 82,980 | 11,928 | 10,965,823 | 34,022,926 |
| **Phase 2 $\times$1.1** | 23,118,783 | 83,288 | 12,088 | 10,899,315 | 34,018,098 |
| **Phase 2 $\times$1.2** | 24,133,853 | 88,033 | 12,745 | 9,822,256 | 33,956,110 |
| **Phase 2 $\times$1.3** | 25,181,339 | 92,661 | 13,326 | 8,657,207 | 33,838,545 |

---

### Leftover Stock Valuation Sensitivity: The "Lift" Disappears

The apparent total recovery of Phase 2 (\$34.02M vs Assurant's \$33.69M) is fragile and entirely dependent on the assumption that unsold inventory retains 100% of its depreciated expected bid value two weeks later.

Testing total value recovery across salvage valuation discounts:

| Leftover Stock Valued At | Assurant Actual | Phase 1 Floor | Phase 2 $\times$0.0 |
|---|---|---|---|
| **100% of mean bid (current)** | \$33.69M | \$33.11M | **\$34.02M** |
| **90% of mean bid** | **\$33.42M** | \$31.18M | \$32.93M |
| **80% of mean bid** | **\$33.15M** | \$29.25M | \$31.83M |

#### Unit Economics and Honest Assessment:
1. **Realization and volume:** Assurant actually sells **107,088 units** at an average realization of **\$289.54 per unit** (\$31.01M revenue). In contrast, Phase 2 ($\times$0.0) sells only **82,971 units** at an average of **\$277.86 per unit** (\$23.05M revenue).
2. **Leftover inventory overhang:** Phase 2 leaves behind **24,117 more unsold units** than Assurant.
3. **Valuation artifact:** Phase 2 only "wins" at 100% salvage because every unsold unit is marked at the full average bid. As soon as a realistic 10% salvage/liquidation discount is applied to unsold stock, Assurant clearly beats Phase 2 (\$33.42M vs \$32.93M). At an 80% valuation, Assurant outperforms Phase 2 by \$1.32M.
4. **Conclusion:** **Phase 2 does not beat Assurant on this hold-out test.** The POC proves the algorithmic mechanics—the ADMM formulation solves at scale, shifts reserves directionally by inventory depth, and provides a controllable clearance knob—but it does *not* demonstrate an economic dollar lift over Assurant's actual decision process.

---

## 5. Caveats

### Known Data Limitations
1. **No inventory overlaps the offers.** Phase 2 uses a stand-in for stock: the units Assurant actually sold per node in the evaluation window. Replace it once a full extract arrives.
2. **Counter outcomes are mostly redeemed.** 182,906 of 205,444 countered offers are `REDEEMED`. The replay backtest only accepts or rejects at the reserve, so it leaves out Assurant's revenue from counters. That understates the "Assurant actual" policy.
3. **Only 6 months of history.** That is not enough to learn yearly seasonality.
4. **Buyer Profile leaks the target.** `ACCEPTANCE_RATE` exactly equals accepted ÷ total over the same Offers file. This matters for Offer Clearance, not for this POC.

### Convergence Status
- **Phase 1:** With `max_iter = 600`, ADMM converges in 374 iterations on the hold-out evaluation dataset ($\epsilon = 10^{-4}$ tolerance), and in 373 iterations on the earlier tuning validation window. (When previously capped at 300 iterations, the algorithm stopped before convergence with primal residuals at 0.061 vs 0.037 tolerance).
- **Phase 2:** Grid-based $x$-updates over discrete candidate prices combined with dual ascent on warehouse constraints oscillate across discrete grid points. At `max_iter = 200`, ADMM did not meet the dual convergence tolerance for any multiplier, but yielded stable reserve solutions and an intuitive, monotonic economic frontier.

### Ordering Violations
- Phase 2 reserves exhibit 301 to 340 sibling grade and capacity ordering violations out of 20,656 total directed ordering edges (<1.7% violation rate). Ordering is enforced as a hard proximal projection constraint ($z_{src} \ge z_{dst}$ via `edge_prox`), but because Phase 2 does not converge to full tolerance within 200 iterations under discrete grid updates and dual ascent, mild residual violations remain in the unconverged state.

## 6. Asks for Assurant

Open items to raise with Assurant to transition from POC to production MVP:
- **Aligned stock per node for a defensible comparison:** For a defensible benchmark, every policy must be tested against the exact same stock: the units Assurant actually sold through the offer channel per SKU $\times$ warehouse.
- **Confirmed salvage discount rate & holding cost:** Official liquidation discount percentage applied to unsold stock at horizon end, plus confirmed daily holding cost per unit. Relative policy rankings are decisively sensitive to leftover stock valuation.
- **Counter negotiation dynamics:** Clarify buyer drop-off curves on counter offers to model active counter-pricing rather than treating the offer channel as a pure accept/reject hurdle.
- **Grade ranking:** Confirm the official rank scale and where grades R, L, S and FTP sit.
- **Attribute step factors:** Provide confirmed factors for storage, generation and brand (currently using placeholder steps of 10% for capacity and 8% for grade).
- **Economic parameters:** Confirm warehouse clearance targets and daily reserve price change guardrails.
- **Cost basis:** Clarify whether "ROI" in the project charter implies return on purchase cost. Without cost basis data, the objective optimizes net revenue recovery.
- **Overlapping inventory extract:** Provide a complete inventory extract covering the full offer history date range.

