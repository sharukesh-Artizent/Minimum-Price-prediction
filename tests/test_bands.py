import numpy as np

from reserve_clm.bands import BAND_KEYS, BAND_LABELS, BAND_MIDS, EDGES, N_BANDS, band_of, soft_targets


def test_band_layout():
    assert N_BANDS == 14 and len(BAND_KEYS) == 14 and len(BAND_LABELS) == 14 and len(BAND_MIDS) == 14
    assert EDGES[0] == 0.70 and EDGES[1] == 0.80 and EDGES[-2] == 1.10 and EDGES[-1] == 1.20
    assert BAND_KEYS[0] == "b00" and BAND_KEYS[-1] == "b13"
    assert BAND_LABELS[0] == "<0.80x" and BAND_LABELS[-1] == ">=1.10x"


def test_band_of_edges():
    assert band_of([0.5, 0.799, 0.80, 0.824, 0.825, 1.099, 1.10, 3.0]).tolist() == [0, 0, 1, 1, 2, 12, 13, 13]


def test_soft_targets_point_mass():
    t = soft_targets([0.96], [0.96])
    assert t.shape == (1, 14) and t[0, band_of([0.96])[0]] == 1.0 and t.sum() == 1.0


def test_soft_targets_overlap_is_proportional():
    # 0.94..0.99 overlaps band [0.925,0.95) by 0.01, [0.95,0.975) by 0.025, [0.975,1.0) by 0.015
    t = soft_targets([0.94], [0.99])[0]
    np.testing.assert_allclose(t[band_of([0.94])[0]], 0.2)
    np.testing.assert_allclose(t[band_of([0.96])[0]], 0.5)
    np.testing.assert_allclose(t[band_of([0.98])[0]], 0.3)
    np.testing.assert_allclose(t.sum(), 1.0)


def test_soft_targets_clip_tails():
    t = soft_targets([1.5], [2.0])[0]
    assert t[13] == 1.0
