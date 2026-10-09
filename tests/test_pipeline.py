import numpy as np
import pandas as pd
import pytest

from conftest import DATA_DIR
from reserve_clm.bands import N_BANDS
from reserve_clm.pipeline import RunConfig, evaluate_all, prepare, run_training, save_results, views
from reserve_clm.train import TrainConfig

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not DATA_DIR.exists(), reason="real data folder not present")]


@pytest.fixture(scope="module")
def prep():
    return prepare(RunConfig(data_roots=[str(DATA_DIR)], smoke_rows=150, replay_max_groups=150), log=lambda *_: None)


def test_prepare_shapes_and_texts(prep):
    L = prep.labels
    assert set(L.split) == {"train", "val", "test"} and (L.lo <= L.hi).all()
    priced = L[L.ref_source != "none"]
    assert len(priced) > 100 and priced.text.str.endswith("market price.").all()
    assert not priced.ref_price.isna().any() and set(priced.true_band) <= set(range(N_BANDS))
    assert len(prep.replay) > 0 and (prep.replay.ref_source != "none").all()
    assert {"sku", "family"} <= set(prep.counts.columns)


def test_features_use_only_past_days(prep):
    o = prep.offers
    for r in prep.labels.sample(20, random_state=0).itertuples():
        w = o[(o.sku == r.sku) & (o.wh == r.wh) & (o.day < r.day) & (o.day >= r.day - pd.Timedelta(days=30))]
        assert r.n_offers_30 == len(w)


def test_training_and_evaluation_with_fake_embeddings(prep, tmp_path):
    V = views(prep)
    rng = np.random.default_rng(0)
    E = {k: rng.standard_normal((len(v), 16)).astype(np.float32) for k, v in V.items()}
    E["replay"] = rng.standard_normal((len(prep.replay), 16)).astype(np.float32)
    E["bands"] = rng.standard_normal((N_BANDS, 16)).astype(np.float32)
    cfg = RunConfig(data_roots=[], work_dir=str(tmp_path), seeds=(1, 2),
                    train=TrainConfig(epochs=2, batch=64, width=32, depth=3, proj=8))
    runs = run_training(V, E, cfg, log=lambda *_: None)
    res = evaluate_all(prep, V, runs)
    h = res["headline_test"]
    assert {"B0 carry-forward", "B1 fixed ratio", "B2 majority band", "CLM seeds (mean)",
            "CLM ensemble (median band)", "CLM ensemble (expected ratio)"} <= set(h.method)
    assert h.interval_hit_rate.between(0, 1).all()
    assert {"reference", "method", "agreement"} <= set(res["replay"].columns)
    assert len(res["training"]) == 2
    save_results(res, str(tmp_path))
    assert (tmp_path / "results" / "summary.json").exists()
