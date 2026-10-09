"""Reserve bands: multiples of the reference (market) price that CLM chooses between."""
import numpy as np

INNER_EDGES = np.round(np.arange(0.80, 1.10001, 0.025), 3)       # 0.800 ... 1.100 (13 edges)
EDGES = np.concatenate([[0.70], INNER_EDGES, [1.20]])             # outer edges bound the tails
N_BANDS = len(EDGES) - 1                                          # 14
BAND_KEYS = [f"b{i:02d}" for i in range(N_BANDS)]
BAND_LABELS = (["<0.80x"] + [f"{EDGES[i]:.3f}-{EDGES[i + 1]:.3f}x" for i in range(1, N_BANDS - 1)]
               + [">=1.10x"])
BAND_MIDS = (EDGES[:-1] + EDGES[1:]) / 2


def band_of(ratio) -> np.ndarray:
    """Band index (0..13) of each ratio; anything below 0.80 is band 0, from 1.10 up band 13."""
    r = np.asarray(ratio, dtype=float)
    return np.searchsorted(EDGES[1:-1], r, side="right")


def soft_targets(lo, hi) -> np.ndarray:
    """Distribution over bands proportional to the overlap with [lo, hi] (ratios).

    A zero-width interval puts all mass on the band containing it.
    """
    lo = np.clip(np.asarray(lo, dtype=float), EDGES[0], EDGES[-1])
    hi = np.clip(np.asarray(hi, dtype=float), EDGES[0], EDGES[-1])
    a, b = EDGES[:-1][None, :], EDGES[1:][None, :]
    overlap = np.clip(np.minimum(hi[:, None], b) - np.maximum(lo[:, None], a), 0.0, None)
    width = overlap.sum(1)
    out = np.zeros_like(overlap)
    wide = width > 1e-12
    out[wide] = overlap[wide] / width[wide, None]
    point = ~wide
    out[point, band_of((lo[point] + hi[point]) / 2)] = 1.0
    return out
