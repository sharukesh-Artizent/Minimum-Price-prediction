# ADMM Reserve Price — Design (Spec)

## 1. Problem

Every day, recommend a reserve price `r_i` for each node `i` = (SKU, warehouse). The data has
no ground-truth reserve, so the work is split into two phases:

- **Phase 1 (baseline):** estimate the floor Assurant currently uses, pooled across siblings.
- **Phase 2 (ROI):** optimise the reserve for expected net recovery under business constraints.

All prices are in **log space**: `x_i = log r_i`. Multiplicative factors (for example, 256GB
worth 1.10 × 128GB) then become additive offsets `d = log 1.10`.

ADMM is the solver in both phases. It suits this problem because the objective is a sum of
small per-node terms joined by edge terms (sibling links) and group terms (warehouse targets).

## 2. The SKU graph (shared by both phases)

Nodes are the SKU × warehouse pairs that appear in the offers. Product attributes are joined by
SKU. `family = manufacturer|model|carrier|lock_status`.

| Edge kind | Connects | Offset `d` (src − dst) | Ordered (`x_src ≥ x_dst`) |
|---|---|---|---|
| warehouse | same SKU, neighbouring warehouses (chain sorted by code) | 0 | no |
| variant | same family, grade, warehouse and capacity (colour variants, chain sorted by SKU) | 0 | no |
| capacity | same family, grade and warehouse; neighbouring capacities | `capacity_step_log × log2(cap_hi / cap_lo)` | yes, src = higher capacity |
| grade | same family, capacity and warehouse; neighbouring known grade ranks | `grade_step_log × (rank_hi − rank_lo)` | yes, src = better grade |

Placeholder factors are `capacity_step_log = log 1.10` per doubling and
`grade_step_log = log 1.08` per rank. Assurant will supply the real values. Grades R, L, S and
FTP get no grade edges until Assurant ranks them. Generation edges are left out of the POC
because there is no generation mapping yet.

## 3. Phase 1 — baseline floor (network lasso)

### Objective

```
minimize_x   Σ_i w_i · L_i(x_i)  +  (γ/2) Σ_i (x_i − p_i)²
           + λ Σ_e a_e · | x_src(e) − x_dst(e) − d_e |
subject to   x_src(e) ≥ x_dst(e)   for ordered edges
```

- **Data term `L_i`:** the recency-weighted *mean* logistic log-loss of Assurant's decisions at
  node i. The model is `P(accept | bid b) = σ((log b − x_i) / T)`, with temperature `T = 0.03`
  (about 3% in price). Offer weight is `exp(−age_days / τ)`, with `τ = 14`. The loss is convex
  in `x_i`.
- **Flexible Pareto weight:** `w_i = n_i / (n_i + k)`, with `k = 10` and `n_i` the number of
  training offers. Head nodes trust their own data. Tail nodes (small `w_i`) lean on their
  edges. Pareto tiers A/B/C (cumulative offer share at 80% / 95%) plus "unseen" are reported for
  evaluation.
- **Prior `p_i`:** the family median log price of accepted bids, falling back to the global
  median. `γ = 0.01` only keeps nodes with no data and no edges well-defined.

### ADMM updates

Each edge holds copies `z_src(e)` and `z_dst(e)` of its endpoints, with scaled duals `u`.

1. **x-update (per node, in parallel):** minimise
   `w_i L_i(x) + (γ/2)(x − p_i)² + (ρ/2) Σ_{e∋i} (x − z_ie + u_ie)²`.
   This is one-dimensional and convex, and is solved by vectorised bisection on the derivative.
2. **z-update (per edge, closed form):** with `p = x_src + u_src`, `q = x_dst + u_dst`,
   `δ = p − q − d` and `θ = 2 λ a / ρ`:
   `t = soft(δ, θ)`, then `t = max(t, −d)` if the edge is ordered,
   `z_src = p − (δ − t)/2`, `z_dst = q + (δ − t)/2`.
3. **Dual update:** `u += x − z`.

The loop stops when the primal and dual residuals are both below `eps · √(2E)`, or at
`max_iter`.

### Evaluation (time-based hold-out, never k-fold)

- Train on the 56 days before `train_end` and test on the following 14 days.
- **Decision agreement:** predict "accept" when `log bid ≥ x_i`. Report accuracy, balanced
  accuracy and log-loss, overall and by Pareto tier.
- **Interval coverage:** on test node-days with both an accept and a non-accept, the share
  where `max non-accepted ≤ floor ≤ min accepted`, plus the mean log gap when outside.
- **Ablations:** family prior only; ADMM with `λ = 0` (no pooling); ADMM fused. The gap between
  fused and `λ = 0` on tiers B, C and unseen is the evidence that ADMM pooling adds value.
- Hyperparameters (λ, ρ) are tuned on an earlier validation window (`train_end = 2026-03-12`)
  so the final test window stays unseen.

Prototype result (train_end 2026-03-26, λ = 1, ρ = 1, 300 iterations, not converged):

| Policy | Accuracy | Tier B | Tier C | Unseen | Interval coverage |
|---|---|---|---|---|---|
| family prior | 0.597 | 0.606 | 0.634 | 0.637 | 0.269 |
| ADMM λ = 0 | 0.732 | 0.677 | 0.629 | 0.648 | 0.291 |
| ADMM fused | **0.760** | **0.745** | **0.720** | **0.706** | **0.388** |

## 4. Phase 2 — ROI-optimal reserve

### Demand inputs per node

- `μ_i`: location of the log bids, fitted with the same network-lasso ADMM using a squared loss.
  This pools sparse nodes.
- `σ_i`: family-level standard deviation of log-bid residuals, with a floor of 0.02.
- `rate_i`: offers per day, shrunk toward the family rate per node with `c = 7` pseudo-days.
- `qty_i`: units per offer, shrunk the same way.
- `δ_i`: daily depreciation per family, from a within-node regression of log sale price on day.
  Clipped to [0, 0.01]; the default is 0.002 when a family has fewer than 30 sales.
- `I_i`: stock to clear. Until real inventory arrives, this is a **proxy**: units Assurant sold
  in the evaluation window.

### Node objective (one-day look-ahead)

Log bids follow `N(μ, σ²)` and offers arrive as `Poisson(rate)` with `qty` units each. Bids at
or above `r` are accepted.

```
survive(r) = 1 − Φ(z),  z = (log r − μ) / σ
sold(r)    = min(I, rate · qty · survive(r))
revenue(r) = e^{μ + σ²/2} · Φ(σ − z) · rate · qty · sold / demand
f_i(r)     = revenue(r) + (I − sold(r)) · C_i
C_i        = value_i · (1 − δ_i − h)^{D_i}        value_i = e^{μ + σ²/2}
D_i        = clip( I_i / (rate · qty · survive(baseline)), 1, 365 )
```

`C_i` is the value of holding a unit one more day. When stock is high, `D_i` is long, so `C_i`
is low and the reserve drops. This is how stock pressure enters the model. Without other
constraints, the best reserve is about `C_i`: sell only when a bid beats the value of waiting.
`h = 0.001` is the holding cost per unit-day as a fraction of value. Assurant will supply the
real figure.

### Joint problem

```
maximize   Σ_i f_i(r_i)
subject to Σ_{i∈w} sold_i(r_i) ≥ T_w                 for each warehouse w
           y_src ≥ y_dst                             ordered sibling edges (y = log r)
           y_i ∈ [x_i − κ_i, x_i + κ_i]               trust region around the phase-1 floor
           κ_i = κ_min + (κ_max − κ_min) · w_i         0.02 … 0.15
```

`T_w = multiplier × Σ_{i∈w} sold_i(phase-1 floor)`.

### ADMM updates

1. **x-update (per node):** choose y on a 41-point grid inside the trust box to maximise
   `(f_i(y) + μ_w · sold_i(y)) / (value_i · max(I_i, 1)) − (ρ/2) Σ_{e∋i} (y − z_ie + u_ie)²`.
   `f_i` is not convex, so the grid gives the exact best grid point. Nodes with no stock keep
   the phase-1 floor.
2. **Warehouse dual:** `μ_w ← max(0, μ_w + η (T_w − Σ sold) / T_w)`, with `η = 5` dollars per
   unit per 100% shortfall. `μ_w` is the price of clearing one more unit in warehouse w.
3. **z-update:** the ordered-edge projection from Phase 1 with `θ = 0`.
4. **Dual update:** `u += y − z`.

### Pareto frontier

Solve for multipliers {0 (unconstrained), 1.0, 1.1, 1.2, 1.3}. Each solve gives one point of
expected units cleared versus expected value. The business chooses its operating point.

### Evaluation — replay backtest

Replay the 14-day hold-out offers in order of date, node, then highest bid. Accept an offer if
`bid ≥ reserve` (or, for "Assurant actual", if it was actually accepted), up to the remaining
proxy stock. Revenue is the bid times units. Leftover stock is marked at
`value · (1 − δ)^14`. Report revenue, units, accepted offers, terminal value and total value
for these policies: Assurant actual, Phase-1 floor, and Phase 2 at each multiplier.

Prototype result (same window; 200 iterations, not converged, 365 ordering violations):

| Policy | Revenue | Units | Total value |
|---|---|---|---|
| Assurant actual | 31.9M | 122K | 88.2M |
| Phase-1 floor | 30.8M | 118K | 88.8M |
| Phase 2 (× 0) | 55.7M | 201K | 91.3M |
| Phase 2 (× 1.3) | 61.9M | 230K | 90.8M |

On stocked nodes, Phase 2 moves reserves −3.2% on average weighted by inventory: it cuts where
stock is heavy and raises where it is thin. Interpret this with care. The replay leaves out
counters (understating Assurant), the stock proxy comes from total sales (including non-offer
channels), and the terminal mark-to-market drives total value.

### Explainability

Each node's reserve can be explained by:
- its Phase-1 floor and Pareto weight
- its continuation value `C_i` and days-to-clear `D_i`
- whether the trust-box edge is binding
- the warehouse dual `μ_w` (the clearance target is binding)
- active ordering edges (held above a cheaper sibling)

## 5. Out of scope for the POC

Generation edges, real inventory, counter modelling in the replay, multi-day dynamic
programming, daily re-solve with limits on day-to-day changes, and Azure ML packaging.
