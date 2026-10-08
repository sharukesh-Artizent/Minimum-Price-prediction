# Reserve Price POC — ADMM

Proof of concept for the **Daily Minimum (Reserve) Price** at SKU × Warehouse level for the
Assurant / HYLA Offer Clearance engagement, using ADMM over a graph of related SKUs.

- **Phase 1 — baseline:** learn Assurant's implied floor from historical accept / counter /
  reject decisions, pooled across sibling SKUs with a network-lasso ADMM and a flexible
  Pareto weight per node.
- **Phase 2 — ROI:** choose the reserve that maximises expected net recovery (sales today plus
  the depreciated value of unsold stock), subject to warehouse clearance targets and sibling
  ordering, solved with ADMM; sweeping the clearance target traces a Pareto frontier.

## Docs

| File | What it covers |
|---|---|
| [docs/01-project-context.md](docs/01-project-context.md) | What the engagement is and where the reserve price fits |
| [docs/02-data-findings.md](docs/02-data-findings.md) | What the data extracts contain, and their limits |
| [docs/03-admm-design.md](docs/03-admm-design.md) | Phase 1 and Phase 2 formulations (the spec) |
| [docs/superpowers/plans/2026-10-08-admm-reserve-price-poc.md](docs/superpowers/plans/2026-10-08-admm-reserve-price-poc.md) | Task-by-task implementation plan |

## Data

The code reads the CSV extracts from `../../Data/Data Files` (relative to this repo).
Override with the `RESERVE_DATA_DIR` environment variable. Data is never copied into the repo.

## Quick start (after the plan is executed)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest
.venv/Scripts/python -m reserve.cli_phase1
.venv/Scripts/python -m reserve.cli_phase2
```

Outputs go to `outputs/` (git-ignored).
