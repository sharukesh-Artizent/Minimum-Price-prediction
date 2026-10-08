"""Point-in-time windowed sums: the value for day d uses events in [d - window, d)."""
import pandas as pd


def trailing_sum(events: pd.DataFrame, keys: list[str], value_cols: list[str],
                 queries: pd.DataFrame, window_days: int) -> pd.DataFrame:
    ev = events.groupby(keys + ["day"], as_index=False)[value_cols].sum()
    q = queries[keys + ["day"]].drop_duplicates()
    zeros = q.assign(**{c: 0.0 for c in value_cols})
    allr = (pd.concat([ev, zeros], ignore_index=True)
            .groupby(keys + ["day"], as_index=False)[value_cols].sum()
            .sort_values(keys + ["day"]))
    rolled = (allr.set_index("day").groupby(keys)[value_cols]
              .rolling(f"{window_days}D", closed="left").sum()
              .reset_index())
    out = q.merge(rolled, on=keys + ["day"], how="left")
    out[value_cols] = out[value_cols].fillna(0.0)
    return out
