# Daily Reserve Price POC with CLM-8B

This folder contains a proof of concept for the first half of the Offer Clearance project:
predicting a **daily reserve (minimum) price for every SKU × warehouse** using a fine-tuned
**CLM-8B** model, and measuring it against Assurant's own historical pricing behaviour.

| What | Where |
|---|---|
| Runnable notebook (server / JupyterLab) | `notebooks/reserve_price_clm.ipynb` |
| Source modules (tested, inlined into the notebook) | `reserve_clm/` |
| Tests (51) | `tests/` |
| Design spec and implementation plan | `docs/` |
| Follow-up experiments (significance, LightGBM, routing) | `notebooks/followup_cells.py` |
| Outputs on the server | `/dist_home/suryansh/sharukesh/Work/reserve_clm_run/` |

---

## 1. The problem in one paragraph

The data has no "correct reserve price" column, so there is no direct ground truth. What it does
have is Assurant's decision on every offer (`OFFER_STATUS`: accepted, countered or not accepted).
On any day where the same SKU at the same warehouse received **both** an accepted offer and a
turned-down offer, Assurant's hidden floor must lie between the **highest bid it turned down** and
the **lowest bid it accepted**. That range is our label. The model learns to place today's reserve
inside it.

> A perfect score therefore means *"reproduces Assurant's current floor"*, not *"finds the optimal
> price"*. Optimising for profit is a later step.

---

## 2. What is CLM?

**CLM (Contrastive Language Model)**, from Stanford and Nvidia, treats a decision as **ranking a
set of candidate answers** instead of generating text. It has two parts:

1. **A frozen reader: Qwen3-8B.** It turns any piece of text into one vector of 4,096 numbers
   (the hidden state of the last token). It is never trained here.
2. **Two small trainable "heads"** (about 19M parameters together):
   - the **state head** maps the vector of the *situation* (our SKU × warehouse × day text) to a
     512-number vector,
   - the **action head** maps the vector of each *candidate answer* (our price buckets) to a
     512-number vector.

The score of each candidate is the cosine similarity between the two 512-number vectors, scaled
and passed through a softmax, giving a probability for every bucket. Training only adjusts the two
heads, so it is fast (minutes) even though the reader has 8 billion parameters.

```
situation text ──► Qwen3-8B (frozen) ──► 4096 numbers ──► state head ──► s (512)
bucket text    ──► Qwen3-8B (frozen) ──► 4096 numbers ──► action head ─► b_k (512)   one per bucket

score_k = similarity(s, b_k)  →  softmax  →  P(bucket k)
```

We start from the **released CLM head** (`CLM_v0.1-8B.pt`) and fine-tune it on our data
(warm start). Code: `reserve_clm/embed.py` (reader) and `reserve_clm/train.py` (heads).

---

## 3. The data we used

Only three files, all from `Data/Data Files/` (CSV or Excel both work):

| File | Rows | Used for |
|---|---|---|
| `Offers.csv` | 531,783 (Oct 10 2025 – Apr 9 2026) | labels, market price, bid features |
| `Sales Order&Line items.csv` | 645,100 lines | sales velocity, sale prices |
| `Product Information.csv` | 335,819 | product attributes (brand, model, capacity, grade…) |

Not used: **Inventory** (covers Jul–Aug 2025 only, no overlap with the offers, and cut off at the
Excel row limit), **Buyer Profile** (its numbers are computed from the same offers, which would leak
the answer), and `ORDER_STATUS` (unreliable).

From 531K offers to training rows:

| Step | Count |
|---|---|
| Offers kept (pending removed) | 530,904 |
| SKU × warehouse × days with any offer | 465,348 |
| Days with **both** an accepted and a turned-down offer (= labelled days) | 17,453 (3.8%) |
| Labelled days with a SKU-level market price, **train** (before Feb 2026) | **3,979** |
| Validation (Feb 2026) / Test (Mar–Apr 2026) | 760 / 3,306 |
| Fallback test set (market price taken from similar products) | 3,170 |

---

## 4. The market price (the anchor everything is multiplied by)

For every SKU × warehouse × day we compute a **market price** from the 30 days *before* that day:

- take every **accepted** offer for that SKU (any warehouse) in the window,
- take the median price of each day, then the median of those daily values,
- this needs at least 3 accepted offers; otherwise the same calculation is done for the **product
  family** (same manufacturer, model, capacity and grade).

`ref_source = sku` → **main set**; `ref_source = family` → **fallback set** (reported separately,
never trained on); no price at all → the day cannot be priced and is only counted.

---

## 5. The buckets

### They are multipliers of the market price, not dollar prices

The model never predicts dollars. It predicts **which multiple of today's market price** the
reserve should be. The final reserve is:

```
reserve = market price (that SKU, that day) × midpoint of the chosen bucket
```

Because the market price changes for every SKU and every day, the same bucket maps to different
dollar values. Example from the test set, market price **$373.50**:

| Bucket | Dollar range that day |
|---|---|
| 0.925–0.950× | $345.49 – $354.83 |
| **0.950–0.975×** | **$354.83 – $364.16**, midpoint **$359.49** = reserve |
| 0.975–1.000× | $364.16 – $373.50 |

### 14 buckets: 12 of 2.5% between 0.80× and 1.10×, plus two tails

| Bucket | Multiplier of market price |
|---|---|
| b00 | below 0.80× (tail) |
| b01 – b12 | 0.800–0.825×, 0.825–0.850×, … , 1.075–1.100× (2.5% each) |
| b13 | 1.10× and above (tail) |

### How we chose them

Before building anything we divided Assurant's floor (the middle of its range) by that day's market
price on every labelled day:

- median **0.967** (Assurant's floor typically sits about 3% below market),
- **90% of days fall between 0.85× and 1.05×**, extremes run from 0.45× to 1.73×.

So **0.80× – 1.10×** covers the central 90% with margin on both sides, and the two tail buckets
catch the rare extremes. The width of **2.5%** was chosen because Assurant's floor is itself only
known to about ±3% (the median range is 6% wide), so finer buckets would mostly split noise. The
range is a design choice informed by the data, not learned; it lives in `reserve_clm/bands.py`.

### What each bucket "says" (the text CLM reads)

Each bucket has an English description, e.g.:

> *"Slight discount reserve: between 0.950x and 0.975x the market price. Use when demand is steady,
> several buyers bid close to market and prices are flat."*

The descriptions group the buckets into tiers (deep clearance, heavy discount, moderate discount,
slight discount, at-market, premium, strong premium).

---

## 6. Input values (what the model sees for each SKU × warehouse × day)

All features use **only days strictly before** the target day (a test checks this against the raw
offers), so the model never sees the bids it is being scored on.

| Group | Features |
|---|---|
| Product | manufacturer, model, capacity, grade, colour, carrier, lock status, warehouse |
| Market price | 30-day market price, where it came from (this SKU or similar products), number of accepted offers behind it, 7-day vs 30-day trend |
| Bids | bids in last 7 and 30 days, distinct buyers, units requested, median bid ÷ market, top bid ÷ market, share of recent bids accepted |
| Sales | units sold in last 7 and 30 days, days since last sale, median stocklist sale price ÷ market |
| Assurant's previous floor | last observed floor range at this SKU × warehouse (÷ market) and how many days ago |
| Calendar | weekday |

### How the inputs become text

The features are written as a short paragraph, with numbers plus plain words (text readers handle
words better than raw numbers), followed by the question, in the same layout CLM was trained on:

```
Product: Apple iPhone XR, 128GB, grade B+, colour White, carrier ATT, UNLOCKED. Warehouse W23.
Market price: $107 from this SKU, last 30 days, 3 accepted offers. Trend last 7 days vs 30 days: flat (+0.0%).
Bids last 7 days: 2 (low demand). Bids last 30 days: 4 from 3 buyers for 370 units.
Median bid 1.00x market, top bid 1.01x market. Share of recent bids accepted: 75%.
Units sold last 7 days: 12, last 30 days: 32 (fast). Last sale 3 days ago. Median sale price 1.00x market.
No previous reserve observed for this SKU at this warehouse.
Day: Thursday.

Which reserve (minimum) price band should be set today for this SKU at this warehouse? Bands are multiples of the SKU's recent market price.
```

---

## 7. Embedding the inputs and the buckets

- Every state text (~29K: train, validation, both test sets and the replay sample) and all 14
  bucket descriptions go through **Qwen3-8B once**. Each becomes a 4,096-number vector taken from
  the last token, normalised to length 1 (the same recipe as CLM's own training code).
- Texts are processed in length-sorted batches; padding does not change the vectors.
- The vectors are **cached to disk** (`embed_cache_Qwen_Qwen3-8B_2048.npz`, ~250 MB). A re-run with
  the same texts skips Qwen3-8B entirely.
- Training then only works on these cached vectors: the state head reads the 29K situation vectors,
  the action head reads the 14 bucket vectors.

---

## 8. How we trained it

| Setting | Value |
|---|---|
| Start point | released CLM head `CLM_v0.1-8B.pt` (width 1536, depth 3, 512-d projection) |
| Objective | CLM's bidirectional **soft-target InfoNCE** (same as CLM's `finetune.py --task choice`) |
| Target | a **soft** distribution: weight on every bucket the floor range overlaps, in proportion to the overlap |
| Optimiser | AdamW, learning rate 5e-4, one-cycle schedule, batch 256, up to 60 epochs |
| Early stopping | on validation (February) interval hit rate, patience 10 |
| Seeds | 3 (1234, 1, 2); we report each seed, their mean, and the average of their probabilities ("ensemble") |
| Decoding | the **median** bucket of the probability distribution → reserve = market × bucket midpoint (also reported: the probability-weighted "expected ratio") |
| Training data | 3,979 main-set days before Feb 2026 |

---

## 9. Baselines (simple rules, no learning)

All three also produce `reserve = market price × ratio`; only the ratio differs. They exist to
show whether CLM learned anything a spreadsheet formula would not.

| Baseline | Rule | Example (market $250) |
|---|---|---|
| **B0 carry-forward** | reuse the middle of Assurant's **previous** floor at this SKU × warehouse; if none, use B1 | last floor 0.95–0.97× → **$240** |
| **B1 fixed ratio** | always the training median ratio, **0.965×** | **$241.25** |
| **B2 majority band** | always the midpoint of the most common training bucket (0.975–1.000×), **0.9875×** | **$246.88** |

---

## 10. Results (test set: March – April 2026)

### 10.1 Main test set (3,306 days, SKU-level market price)

| Method | Hit rate | MAPE | Within ±1 bucket | Within ±3 | Within ±5 |
|---|---|---|---|---|---|
| B0 carry-forward | 50.3% | 4.77% | 57.4% | 87.7% | 96.0% |
| B1 fixed ratio | 51.8% | 4.33% | 61.2% | 90.8% | 96.0% |
| B2 majority band | 52.9% | 4.67% | 60.6% | 88.6% | 96.2% |
| CLM seeds (mean of 3) | 54.9% (±0.6) | 4.12% | 64.4% | 91.8% | 97.2% |
| **CLM ensemble** | **55.7%** | **4.04%** | **65.6%** | **92.3%** | **97.4%** |

- **Hit rate** = share of days where the reserve landed inside Assurant's floor range
  [highest bid turned down, lowest bid accepted].
- **MAPE** = average % distance between the reserve and the middle of that range.
- **Within ±k** = predicted bucket at most k buckets away from the bucket holding the middle of
  the range (±1 ≈ ±2.5% of price, ±3 ≈ ±7.5%, ±5 ≈ ±12.5%).

CLM is best on every measure, by about 3 points of hit rate, and the result repeated across two
full runs (55.2% and 55.7%). The ±3 and ±5 windows barely separate the methods, because Assurant's
floor almost always sits between 0.85× and 1.05× of market, so nobody is ever far off.

### 10.2 Why a 4% MAPE still gives only ~55% hit rate: the target is narrower than 4%

A 4% average error sounds good, but two facts explain the gap:

1. **4% is easy here.** The market price is already an excellent anchor: B1, which applies one
   fixed multiplier to every SKU, already reaches 4.3% MAPE. CLM's own contribution is moving
   4.3% → 4.0%.
2. **The hit window is usually narrower than 4%.** Half-width of Assurant's floor range on test days:

   | Half-width (± around the middle) | Share of days | B1 hit rate on those days |
   |---|---|---|
   | under ±1% | 16.6% | 5.6% |
   | ±1–2% | 21.7% | 27.4% |
   | ±2–4% | 27.1% | 61.6% |
   | over ±4% | 34.6% | 81.7% |

   The median window is **±2.7%**. If Assurant turned down $236 and accepted $240 on the same day,
   the target is a $4 gap: a reserve of $243, only 1.5% off, counts as a miss.

3. **Even a perfect bucket picker tops out at 88.8%.** With 2.5% buckets, 11% of days have a range
   so narrow that no bucket midpoint falls inside it. CLM's 55.7% should be read against that ceiling.

**Would smaller buckets help?** Rounding a prediction to a 2.5% bucket moves it by at most 1.25%,
far less than the ~4% model error. Rounding a continuous model's predictions to 1% buckets raised
its hit rate by under 1 point (57.1% → 58.0%), and CLM's continuous "expected ratio" output scored
55.1% vs 55.7% for the bucket output. Smaller buckets would also leave fewer training examples per
bucket and near-identical bucket texts. Expected effect for CLM: 0 to +1 point.

### 10.3 Daily acceptance: the predicted reserve as the day's base price

For each SKU × warehouse × day, accept **every** incoming offer whose bid is at or above the
predicted reserve, then compare with what Assurant actually did with those offers.

**Labelled test days (main set, 3,306 days, 7,887 offers):**

| Method | Agreement with Assurant | Precision | Recall | Whole day exactly right | Accepted / day (Assurant: 1.14) | Lost sales | Under-pricing |
|---|---|---|---|---|---|---|---|
| B0 carry-forward | 75.8% | 72.7% | 79.2% | 49.5% | 1.24 | 10.0% | 14.2% |
| B1 fixed ratio | 77.1% | 71.4% | 86.9% | 51.8% | 1.39 | 6.3% | 16.6% |
| B2 majority band | 77.3% | 79.1% | 71.5% | 52.9% | 1.03 | 13.6% | 9.0% |
| **CLM ensemble** | **79.2%** | 75.5% | 83.6% | **55.7%** | 1.26 | 7.9% | 13.0% |

- **Agreement**: share of offers where our accept/reject matches Assurant's.
- **Precision**: of the offers we accept, share Assurant also accepted. **Recall**: of the offers
  Assurant accepted, share we also accept.
- **Lost sales**: we would reject an offer Assurant accepted. **Under-pricing**: we would accept an
  offer Assurant turned down.

CLM matches Assurant on about **79 of every 100 offers** (vs ~77 for the best rule). It leans
toward accepting: it accepts ~11% more offers per day than Assurant did, so its mistakes are more
often under-pricing than lost sales.

**All sampled test-period days** (8,914 days, 11,337 offers, including days where every offer was
accepted or none was): CLM agreement **71.5%** vs 69.5% for B0; whole-day exact match 67.8% vs 66.0%.
CLM accepts 59% of offers vs Assurant's 52.5%.

### 10.4 Fallback set (3,170 days, market price from similar products)

| Method | Hit rate | MAPE | Agreement (daily acceptance) |
|---|---|---|---|
| B0 carry-forward | **35.9%** | 8.69% | **69.2%** |
| CLM ensemble | 35.0% | 8.28% | 68.2% |

CLM is **not usable here yet**: no better than carry-forward, the three seeds disagree (30.5% –
34.1%), and it prices too low (accepts 64% of offers vs Assurant's 47%). It never trained on
family-priced rows. Use B0 for these SKUs for now.

### 10.5 Other observations

- **Confidence is meaningful:** calibration error 0.015; hit rate rises from 52% in the least
  confident fifth of predictions to 61% in the most confident fifth.
- **By segment** CLM beats B0 for Apple (+6 points), Samsung (+5), phones under $150 (+8) and the
  most-traded SKUs (+8); the only exception is Google (155 rows).
- **The model is unsure:** it typically puts only 15–33% on its top bucket, spread across three
  neighbouring buckets in 0.925–1.025×. It behaves like a smarter fixed ratio.
- **Reference point:** a LightGBM model on the same features reached 57.9% hit rate on the main
  set and 44.1% on the fallback set (see `notebooks/followup_cells.py`).

---

## 11. Limitations

- The label is Assurant's **current** behaviour, so the model can at best imitate it.
- Only 3,979 training days; the median SKU × warehouse has a single labelled day.
- No inventory data overlaps the offer period, so stock level and ageing are missing.
- Offer quantities overstate real sales about 3×, so dollar "value" from offers is not revenue.
- The data holds ~2,900 offers/day, far below the contract's ~140K/day target.

## 12. Next steps

1. Train on the 4,510 family-priced rows too (roughly doubles the data, should fix the fallback set).
2. Choose the epoch by validation log-loss or average the last epochs' weights (removes seed noise).
3. Nudge reserves slightly up to correct the lean toward accepting.
4. Route: use CLM when it is confident, a rule otherwise.
5. Run the follow-up cells for paired significance tests and the LightGBM comparison in the same run.

## 13. How to run

1. Put `Offers`, `Product Information` and `Sales Order&Line items` (CSV or XLSX) in
   `/dist_home/suryansh/sharukesh/Work`.
2. Install: `pip install "transformers>=4.51" pandas pyarrow openpyxl huggingface_hub numpy requests`
   and `pip install --no-deps "contrastive-lm @ git+https://github.com/Contrastive-LM/CLM@d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6"`.
3. Optional pre-download: `hf download Qwen/Qwen3-8B` and
   `hf download Contrastive-LM/CLM-v0.1-8B CLM_v0.1-8B.pt --local-dir ~/.cache/clm`.
4. Open `notebooks/reserve_price_clm.ipynb`, set `GPU = 1` (cell 4) if needed, Run All.
   Results are written to `Work/reserve_clm_run/results/`.

Local tests: `.venv/Scripts/python -m pytest` (51 tests) and `python tools/run_smoke.py` (runs the
whole notebook on CPU with a small model and a data sample).
