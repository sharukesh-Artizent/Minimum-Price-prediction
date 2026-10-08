"""B1: transparent rule r = m * (1 - d[age band, velocity band]), d tuned by replay."""
import numpy as np
import pandas as pd

from reserve.counter_curve import CounterCurve
from reserve.replay import replay_policy

AGE_EDGES = [-np.inf, 30, 60, 90, np.inf]
D_GRID = np.round(np.arange(0.0, 0.26, 0.01), 2)


def tau_edges_from(df: pd.DataFrame) -> np.ndarray:
    return np.quantile(df.tau, [1 / 3, 2 / 3])


def bands(df: pd.DataFrame, tau_edges) -> tuple[np.ndarray, np.ndarray]:
    age = pd.cut(df.age_days, AGE_EDGES, labels=False, right=False).to_numpy().astype(int)
    tau = np.digitize(df.tau.to_numpy(float), tau_edges)
    return age, tau


def tune_rule(ev: pd.DataFrame, curve: CounterCurve, tau_edges, d_grid=None) -> dict:
    d_grid = D_GRID if d_grid is None else d_grid
    a, t = bands(ev, tau_edges)
    table = {}
    for (ai, ti), cell in ev.assign(_a=a, _t=t).groupby(["_a", "_t"]):
        best_d, best_v = None, -np.inf
        for d in d_grid:  # ascending: ties keep the smaller discount
            v = replay_policy(cell, cell.m.to_numpy(float) * (1 - d), curve).value.sum()
            if v > best_v:
                best_d, best_v = float(d), v
        table[(int(ai), int(ti))] = best_d
    return table


def apply_rule(df: pd.DataFrame, table: dict, tau_edges, default: float = 0.05) -> np.ndarray:
    a, t = bands(df, tau_edges)
    d = np.array([table.get((int(x), int(y)), default) for x, y in zip(a, t)])
    return df.m.to_numpy(float) * (1 - d)
