import numpy as np
import pandas as pd

from reserve.pareto import node_tiers, pareto_tiers, pareto_weight


def test_pareto_tiers_include_item_that_crosses_threshold():
    counts = pd.Series([5, 50, 30, 5, 10], index=["e", "a", "b", "d", "c"])
    tiers = pareto_tiers(counts)
    assert tiers.loc[["a", "b"], "tier"].tolist() == ["A", "A"]
    assert tiers.loc[["c", "e"], "tier"].tolist() == ["B", "B"]
    assert tiers.loc["d", "tier"] == "C"


def test_node_tiers_marks_zero_count_nodes_unseen():
    tiers = node_tiers(np.array([90, 0, 10]))
    assert tiers.tolist() == ["A", "unseen", "B"]


def test_pareto_weight_saturates():
    w = pareto_weight([0, 10, 1000], k=10)
    assert np.allclose(w, [0.0, 0.5, 1000 / 1010])
