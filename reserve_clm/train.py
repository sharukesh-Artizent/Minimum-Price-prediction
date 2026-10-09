"""Fine-tune CLM's projection heads on cached encoder embeddings.

Same objective as CLM's `train/finetune.py --task choice --loss infonce --targets soft`:
bidirectional InfoNCE between states and the band texts, against a soft target
distribution. The 8B encoder never changes; only the two small heads train.
"""
from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from clm.heads import make_head
except ImportError:
    def make_head(width: int, depth: int = 2, proj: int = 512, activation: str = "gelu",
                  layernorm: bool = False, residual: bool = False, hidden: int = 4096):
        """Fallback matching clm.heads.make_head architecture when clm is not installed."""
        act_map = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
        act = act_map.get(activation.lower(), nn.GELU)

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(hidden, width)
                self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
                self.norms = nn.ModuleList((nn.LayerNorm(width) if layernorm else nn.Identity())
                                           for _ in range(depth - 2))
                self.out = nn.Linear(width, proj)
                self.act = act()
                self.residual = residual

            def forward(self, x):
                x = self.act(self.inp(x))
                for lin, nrm in zip(self.hidden, self.norms):
                    h = self.act(nrm(lin(x)))
                    x = x + h if self.residual else h
                return self.out(x)

        return Head()


@dataclass
class TrainConfig:
    epochs: int = 60
    batch: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.0
    patience: int = 10
    seed: int = 1234
    width: int = 1536      # used only when there is no compatible warm-start checkpoint
    depth: int = 3
    proj: int = 512


class HeadModel(nn.Module):
    def __init__(self, sh: nn.Module, ah: nn.Module, logit_scale: nn.Parameter):
        super().__init__()
        self.sh, self.ah, self.logit_scale = sh, ah, logit_scale

    def logits(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        zq = F.normalize(self.sh(x), dim=-1)
        zc = F.normalize(self.ah(c), dim=-1)
        return self.logit_scale.exp().clamp(max=100.0) * (zq @ zc.t())


def build_model(hidden: int, cfg: TrainConfig, init_ckpt: str | None = None) -> tuple[HeadModel, dict, bool]:
    head_cfg = dict(width=cfg.width, depth=cfg.depth, proj=cfg.proj, activation="gelu",
                    layernorm=True, residual=False, hidden=hidden)
    ck = torch.load(init_ckpt, map_location="cpu", weights_only=False) if init_ckpt else None
    warm = False
    if (ck is not None
            and "state_head" in ck
            and "inp.weight" in ck["state_head"]
            and ck["state_head"]["inp.weight"].shape[1] == hidden):
        c0 = ck.get("cfg", {})
        head_cfg.update(width=c0.get("width", cfg.width),
                        depth=c0.get("depth", cfg.depth),
                        proj=c0.get("projection_dim", ck.get("projection_dim", cfg.proj)),
                        activation=c0.get("activation", "gelu"),
                        residual=c0.get("residual", False),
                        layernorm=any(k.startswith("norms.") for k in ck["state_head"]))
        warm = True
    sh, ah = make_head(**head_cfg), make_head(**head_cfg)
    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
    if warm:
        sh.load_state_dict(ck["state_head"])
        ah.load_state_dict(ck["action_head"])
        if "logit_scale" in ck:
            with torch.no_grad():
                logit_scale.copy_(torch.as_tensor(ck["logit_scale"]).float())
    return HeadModel(sh, ah, logit_scale), head_cfg, warm


def soft_infonce(lg: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """States -> bands against the soft targets, bands -> states against the states that hold them."""
    fwd = -(t * F.log_softmax(lg, 1)).sum(1).mean()
    w = t.t()
    keep = w.sum(1) > 0
    if keep.any():
        w = w[keep] / w[keep].sum(1, keepdim=True)
        bwd = -(w * F.log_softmax(lg.t()[keep], 1)).sum(1).mean()
        return (fwd + bwd) / 2
    return fwd


def _to_tensor(arr, device: torch.device | str) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(arr), dtype=torch.float32, device=device)


@torch.no_grad()
def predict_proba(model: HeadModel, x, c, batch: int = 4096) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    ct = _to_tensor(c, device)
    if len(x) == 0:
        return np.zeros((0, len(c)), dtype=np.float32)
    out = [F.softmax(model.logits(_to_tensor(x[i:i + batch], device), ct), dim=1).cpu().numpy()
           for i in range(0, len(x), batch)]
    return np.concatenate(out) if out else np.zeros((0, len(c)), dtype=np.float32)


@dataclass
class TrainResult:
    model: HeadModel
    head_cfg: dict
    warm_start: bool
    best_epoch: int
    best_score: float
    history: list = field(default_factory=list)


def train_heads(x_tr, t_tr, c, x_val, val_score, cfg: TrainConfig, init_ckpt: str | None = None,
                device: str | None = None, log=print) -> TrainResult:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    model, head_cfg, warm = build_model(int(np.asarray(x_tr).shape[1]), cfg, init_ckpt)
    model.to(device)
    x = _to_tensor(x_tr, device)
    t = _to_tensor(t_tr, device)
    ct = _to_tensor(c, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = math.ceil(len(x) / cfg.batch) * cfg.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(1, steps),
                                                pct_start=0.1, anneal_strategy="cos")
    gen = torch.Generator().manual_seed(cfg.seed)
    best = val_score(predict_proba(model, x_val, c))
    best_state, best_epoch, stale = copy.deepcopy(model.state_dict()), 0, 0
    history = [{"epoch": 0, "train_loss": None, "val_score": best}]
    log(f"[train] seed {cfg.seed} warm_start={warm} epoch 0 val {best:.4f}")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(x), generator=gen).split(cfg.batch):
            idx = idx.to(device)
            loss = soft_infonce(model.logits(x[idx], ct), t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        score = val_score(predict_proba(model, x_val, c))
        history.append({"epoch": epoch, "train_loss": total / len(x), "val_score": score})
        log(f"[train] seed {cfg.seed} epoch {epoch} loss {total / len(x):.4f} val {score:.4f}")
        if score > best + 1e-6:
            best, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg.patience:
                log(f"[train] early stop at epoch {epoch} (best {best_epoch})")
                break
    model.load_state_dict(best_state)
    return TrainResult(model, head_cfg, warm, best_epoch, float(best), history)


def save_checkpoint(res: TrainResult, path: str) -> None:
    """Same layout as CLM checkpoints, so clm.heads.HeadPair / clm.Engine can load it."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    c = res.head_cfg
    torch.save({
        "state_head": res.model.sh.state_dict(),
        "action_head": res.model.ah.state_dict(),
        "logit_scale": res.model.logit_scale.detach().cpu(),
        "cfg": {
            "width": c["width"],
            "depth": c["depth"],
            "projection_dim": c["proj"],
            "activation": c["activation"],
            "layernorm": c["layernorm"],
            "residual": c["residual"],
            "hidden_size": c["hidden"],
            "task": "reserve_band",
        },
        "epoch": res.best_epoch,
        "metrics": {"val_interval_hit_rate": res.best_score},
    }, path)


def load_checkpoint(path: str, hidden: int | None = None, cfg: TrainConfig | None = None) -> tuple[HeadModel, dict, bool]:
    """Load a checkpoint created by save_checkpoint or CLM."""
    if hidden is None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        hidden = ck["state_head"]["inp.weight"].shape[1]
    return build_model(hidden, cfg or TrainConfig(), init_ckpt=path)


# Compatibility aliases
CLMHeadModel = HeadModel
soft_infonce_loss = soft_infonce
predict_probs = predict_proba
save_heads = save_checkpoint
load_heads = load_checkpoint
