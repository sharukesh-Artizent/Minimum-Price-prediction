"""M3: empirical distribution of bid / m-hat at the finest hierarchy level with enough bids."""
import numpy as np
import pandas as pd

from reserve.config import LEVELS, Params


class BidDist:
    def __init__(self, levels, qlevels, tables, global_q, min_obs):
        self.levels, self.qlevels, self.tables = levels, qlevels, tables
        self.global_q, self.min_obs = global_q, min_obs
        self.qcols = [f"q{i}" for i in range(len(qlevels))]

    @classmethod
    def fit(cls, offers_f: pd.DataFrame, cutoff: pd.Timestamp, p: Params,
            levels: list[list[str]] = LEVELS) -> "BidDist":
        start = cutoff - pd.Timedelta(days=p.bid_window_days)
        o = offers_f[(offers_f.day < cutoff) & (offers_f.day >= start)]
        o = o.assign(x=(o.offer_price / o.m).clip(0.2, 3.0))
        ql = (np.arange(p.n_bid_quantiles) + 0.5) / p.n_bid_quantiles
        qcols = [f"q{i}" for i in range(len(ql))]
        tables = []
        for keys in levels:
            t = o.groupby(keys).x.quantile(ql).unstack()
            t.columns = qcols
            t["n"] = o.groupby(keys).size()
            tables.append(t.reset_index())
        return cls(levels, ql, tables, np.quantile(o.x, ql), p.min_group_obs)

    def lookup(self, df: pd.DataFrame) -> np.ndarray:
        out = np.tile(self.global_q, (len(df), 1))
        for keys, t in zip(self.levels, self.tables):
            t = t[t.n >= self.min_obs]
            j = df[keys].reset_index(drop=True).merge(t, on=keys, how="left")
            ok = j["n"].notna().to_numpy()
            out[ok] = j[self.qcols].to_numpy()[ok]
        return out
