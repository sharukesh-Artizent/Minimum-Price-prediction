# Minimum Price Prediction - POC Approaches

This repository contains Proof of Concept (POC) implementations for **Offer Clearance / Minimum Price Prediction**, comparing different mathematical and algorithmic modeling strategies.

Each POC implementation is maintained in its own dedicated branch:

## Branches

| Branch | Approach | Key Techniques & Description |
|---|---|---|
| [`admm`](https://github.com/sharukesh-Artizent/Minimum-Price-prediction/tree/admm) | **ADMM (Alternating Direction Method of Multipliers)** | Convex optimization framework decomposing reserve price optimization into distributed consensus subproblems with clearance rate constraints. |
| [`cml`](https://github.com/sharukesh-Artizent/Minimum-Price-prediction/tree/cml) | **CML (Constrained / Contrastive Machine Learning)** | Deep neural representations with text embeddings (Qwen), calibrated interval prediction, and business rule constrained training heads. |
| [`pareto`](https://github.com/sharukesh-Artizent/Minimum-Price-prediction/tree/pareto) | **Pareto Multi-Objective Optimization** | Non-dominated sorting / Pareto frontier search balancing clearance rate vs. revenue/margin trade-offs. |

## Quick Start

To explore any specific POC, clone the repository and checkout the desired branch:

```bash
# Clone the repository
git clone https://github.com/sharukesh-Artizent/Minimum-Price-prediction.git
cd Minimum-Price-prediction

# Switch to the ADMM approach
git checkout admm

# Switch to the CML approach
git checkout cml

# Switch to the Pareto approach
git checkout pareto
```
