import numpy as np
import pytest

from reserve_clm.embed import QwenEmbedder

pytestmark = pytest.mark.slow
MODEL = "Qwen/Qwen3-0.6B"


def test_embed_shapes_norms_and_cache(tmp_path):
    cache = tmp_path / "c.npz"
    e = QwenEmbedder(MODEL, max_len=64, batch_size=2, cache_path=str(cache), log=lambda *_: None)
    texts = ["short text", "a somewhat longer text about phones and prices", "short text"]
    v = e.embed(texts)
    assert v.shape == (3, e.hidden) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-2)
    np.testing.assert_array_equal(v[0], v[2])
    single = e.embed([texts[1]])                      # padding in a batch must not change the vector
    np.testing.assert_allclose(single[0], v[1], atol=2e-3)
    e.close()
    e2 = QwenEmbedder.__new__(QwenEmbedder)           # cache reload without touching the model
    e2._load_cache(str(cache))
    assert len(e2.cache) == 2


def test_embed_empty_list(tmp_path):
    cache = tmp_path / "empty.npz"
    e = QwenEmbedder(MODEL, max_len=64, batch_size=2, cache_path=str(cache), log=lambda *_: None)
    v = e.embed([])
    assert v.shape == (0, e.hidden)
    assert v.dtype == np.float32
    e.close()
