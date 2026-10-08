# Project Context

Sources: `Doc/Assurant-WO Artizent--OfferClearance_Assur-26-27-032N_v1.4.docx` (SOW),
`Doc/Price Prediction & Offer Clearance Business Context 1.docx`,
`Doc/Price Prediction- Offer Clearence.pptx` (solution deck).

## The engagement

Assurant / HYLA resells used devices. Artizent is building a standalone decision engine with
two parts:

1. **Daily Minimum (Reserve) Price** per SKU × Warehouse. It is predicted every day, then a
   business user reviews it, may override it, and approves it. Once approved, it becomes the
   active floor for that day.
2. **Offer Clearance.** Buyer offers arrive from source systems (real-time API or batch) and
   receive Accept / Reject / Counter Price / Counter Price + Quantity, with a rationale. The
   final outcome is sent back by Offer ID and is used to improve the models.

Commercials: fixed price, USD 491K, 28 weeks (16-week MVP, then 12 weeks of production
hardening). The target volume is about 140K offers per day. The stack is Azure: .NET 10 APIs,
an AKS orchestrator, Azure ML, Azure Managed Redis, Azure SQL, Service Bus, with Snowflake as
the data source.

## What the documents say about the reserve price

- The business doc rules out the median, average or most common offer. The system should try
  candidate prices, predict the outcome at each one (chance of selling, time to sell, value
  realised, risk of waiting), and pick the best balance.
- The candidate signals it lists are product attributes (model, storage, colour, grade,
  battery), inventory (stock, days on hand, ageing, inflow and outflow), market activity
  (offers, accepted offers, counters, final prices) and demand (buyer participation, offer
  concentration, offer velocity, comparable products).
- **No model has been chosen.** The SOW says the *"final modelling approach features and
  decision signals will be established through data analysis"*. The deck covers only Offer
  Clearance.
- In the SOW, the approved reserve price acts as a hard guardrail for Offer Clearance. Human
  approval and override apply only to the reserve price.

## Why this is an optimisation problem

The data contains no "correct reserve price" to learn from. What it does contain:

- **Assurant's current behaviour:** each accept / counter / reject decision brackets the floor
  Assurant was effectively using. Phase 1 learns this as the baseline to beat.
- **Buyer bids:** these reveal demand at each price, so we can estimate what a different
  reserve would have sold and earned. Phase 2 optimises against this.

## Open items to raise with Assurant

- Grade ranking, including where R, L, S and FTP sit.
- Storage, generation and brand factors. The deck says Assurant sets these.
- Holding cost per unit-day, salvage floors, warehouse clearance targets, and a limit on daily
  price changes.
- Cost basis, if "ROI" means return on purchase cost. Without it, we optimise net recovery.
- A complete inventory extract that overlaps the offer history (see 02-data-findings).
- What `ORDER_STATUS = REDEEMED` means.
