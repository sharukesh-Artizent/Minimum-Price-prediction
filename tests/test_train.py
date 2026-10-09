import numpy as np
import torch

from reserve_clm.train import (
    CLMHeadModel,
    HeadModel,
    TrainConfig,
    build_model,
    load_heads,
    predict_proba,
    predict_probs,
    save_checkpoint,
    save_heads,
    soft_infonce,
    soft_infonce_loss,
    train_heads,
)


def _data(n, rng, centers):
    y = rng.integers(0, len(centers), n)
    x = centers[y] + 0.3 * rng.standard_normal((n, centers.shape[1]))
    return x.astype(np.float32), y


def test_soft_infonce_prefers_correct_logits():
    t = torch.eye(3)
    good = soft_infonce(10 * torch.eye(3), t)
    bad = soft_infonce(-10 * torch.eye(3), t)
    assert good < bad


def test_train_heads_learns_separable_bands():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((14, 32)).astype(np.float32)
    x_tr, y_tr = _data(1400, rng, centers)
    x_va, y_va = _data(280, rng, centers)
    t_tr = np.eye(14, dtype=np.float32)[y_tr]
    cfg = TrainConfig(epochs=40, batch=128, lr=3e-3, patience=40, width=64, depth=3, proj=32)
    res = train_heads(x_tr, t_tr, centers, x_va, lambda p: float((p.argmax(1) == y_va).mean()), cfg, log=lambda *_: None)
    p = predict_proba(res.model, x_va, centers)
    assert p.shape == (280, 14) and np.allclose(p.sum(1), 1, atol=1e-5)
    assert (p.argmax(1) == y_va).mean() > 0.9 and res.best_score > 0.9 and not res.warm_start


def test_checkpoint_round_trip_warm_starts(tmp_path):
    rng = np.random.default_rng(1)
    centers = rng.standard_normal((14, 16)).astype(np.float32)
    x, y = _data(200, rng, centers)
    cfg = TrainConfig(epochs=2, batch=64, width=32, depth=3, proj=8)
    res = train_heads(x, np.eye(14, dtype=np.float32)[y], centers, x, lambda p: 0.0, cfg, log=lambda *_: None)
    path = tmp_path / "head.pt"
    save_checkpoint(res, str(path))
    model, head_cfg, warm = build_model(16, TrainConfig(width=999), init_ckpt=str(path))
    assert warm and head_cfg["width"] == 32 and head_cfg["proj"] == 8
    np.testing.assert_allclose(predict_proba(model, x, centers), predict_proba(res.model, x, centers), atol=1e-5)
    _, _, warm2 = build_model(24, cfg, init_ckpt=str(path))    # width mismatch -> fresh heads
    assert not warm2


def test_aliases_and_serialization_helpers(tmp_path):
    assert CLMHeadModel is HeadModel
    assert soft_infonce_loss is soft_infonce
    assert predict_probs is predict_proba
    assert save_heads is save_checkpoint

    rng = np.random.default_rng(42)
    centers = rng.standard_normal((5, 8)).astype(np.float32)
    x, y = _data(50, rng, centers)
    t = np.eye(5, dtype=np.float32)[y]
    cfg = TrainConfig(epochs=1, batch=32, width=16, depth=2, proj=4)
    res = train_heads(x, t, centers, x, lambda p: 0.0, cfg, log=lambda *_: None)
    path = tmp_path / "model.pt"
    save_heads(res, str(path))

    # load_heads with hidden=None (inferred from checkpoint)
    m1, c1, w1 = load_heads(str(path))
    assert w1 and c1["hidden"] == 8
    p1 = predict_probs(m1, x, centers)
    p0 = predict_proba(res.model, x, centers)
    np.testing.assert_allclose(p1, p0, atol=1e-5)


def test_early_stopping():
    rng = np.random.default_rng(7)
    centers = rng.standard_normal((3, 4)).astype(np.float32)
    x, y = _data(30, rng, centers)
    t = np.eye(3, dtype=np.float32)[y]
    # Set patience=2 with a val_score that never improves
    cfg = TrainConfig(epochs=20, batch=16, patience=2, width=8, depth=2, proj=4)
    res = train_heads(x, t, centers, x, lambda p: 0.0, cfg, log=lambda *_: None)
    # Best epoch is 0, stopped after epoch 2
    assert len(res.history) <= 4
    assert res.best_epoch == 0


def test_predict_proba_empty_and_torch_tensor():
    m, _, _ = build_model(8, TrainConfig(width=16, depth=2, proj=4))
    centers = np.ones((5, 8), dtype=np.float32)
    p_empty = predict_proba(m, np.zeros((0, 8), dtype=np.float32), centers)
    assert p_empty.shape == (0, 5)

    x_tensor = torch.randn(4, 8)
    p_tensor = predict_proba(m, x_tensor, torch.as_tensor(centers))
    assert p_tensor.shape == (4, 5)
    np.testing.assert_allclose(p_tensor.sum(axis=1), 1.0, atol=1e-5)
