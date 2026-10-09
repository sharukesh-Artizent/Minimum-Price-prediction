"""Project-wide constants for the reserve-price CLM POC."""
import pandas as pd

OFFER_STATUSES = ("ACCEPTED", "COUNTERED", "NOT_ACCEPTED")
TRAIN_END = pd.Timestamp("2026-02-01")   # train: day < TRAIN_END
VAL_END = pd.Timestamp("2026-03-01")     # val: TRAIN_END <= day < VAL_END; test: day >= VAL_END
REF_WINDOW_DAYS = 30
TREND_WINDOW_DAYS = 7
MIN_REF_ACCEPTS = 3                      # accepted offers needed in the window for a reference price
CLM_REPO_COMMIT = "d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6"
