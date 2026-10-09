"""Qwen3 last-token embeddings (the reader CLM's heads sit on), cached on disk by text hash.

Recipe matches CLM's train/embed_utils.py: tokens without special tokens, keep the last
max_len - 1, last-token hidden state, L2-normalised. Batches are length-sorted and right
padded; causal attention means real tokens never see the padding.
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np


def _key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


class QwenEmbedder:
    def __init__(self, model_name: str, max_len: int = 2048, batch_size: int = 16,
                 cache_path: str | None = None, log=print):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.log = torch, log
        self.max_len, self.batch_size, self.cache_path = max_len, batch_size, cache_path
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, device_map="auto").eval()
        else:
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).eval()
        self.hidden = int(self.model.config.hidden_size)
        self.device = self.model.get_input_embeddings().weight.device
        pad = self.tok.pad_token_id
        self.pad_id = pad if pad is not None else (self.tok.eos_token_id or 0)
        self._load_cache(cache_path)

    def _load_cache(self, path: str | None) -> None:
        self.cache = {}
        if not path:
            return
        if not os.path.exists(path) and not path.endswith(".npz") and os.path.exists(path + ".npz"):
            path = path + ".npz"
        if os.path.exists(path):
            z = np.load(path)
            self.cache = dict(zip(z["keys"].tolist(), z["vecs"]))
            if hasattr(z, "files") and "vecs" in z.files and len(z["vecs"]) > 0 and not hasattr(self, "hidden"):
                self.hidden = int(z["vecs"].shape[1])

    def _save_cache(self) -> None:
        if not self.cache_path or not self.cache:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        keys = list(self.cache)
        np.savez(self.cache_path, keys=np.array(keys), vecs=np.stack([self.cache[k] for k in keys]))

    def _ids(self, text: str) -> list[int]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = self.tok(" ", add_special_tokens=False)["input_ids"]
        limit = max(1, self.max_len - 1)
        return ids[-limit:]

    def embed(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        keys = [_key(t) for t in texts]
        todo = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in self.cache))
        if todo:
            if self.model is None:
                raise RuntimeError("Model has been closed and cannot embed uncached texts.")
            ids = [self._ids(t) for t in todo]
            order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
            started, done = time.time(), 0
            with torch.no_grad():
                for b, s in enumerate(range(0, len(order), self.batch_size)):
                    chunk = order[s:s + self.batch_size]
                    width = max(len(ids[i]) for i in chunk)
                    x = torch.full((len(chunk), width), self.pad_id, dtype=torch.long)
                    m = torch.zeros((len(chunk), width), dtype=torch.long)
                    for r, i in enumerate(chunk):
                        x[r, :len(ids[i])] = torch.tensor(ids[i])
                        m[r, :len(ids[i])] = 1
                    h = self.model(input_ids=x.to(self.device), attention_mask=m.to(self.device)).last_hidden_state
                    last = torch.tensor([len(ids[i]) - 1 for i in chunk], device=h.device)
                    v = h[torch.arange(len(chunk), device=h.device), last].float()
                    v = torch.nn.functional.normalize(v, dim=-1).cpu().numpy().astype(np.float16)
                    for i, vec in zip(chunk, v):
                        self.cache[_key(todo[i])] = vec
                    done += len(chunk)
                    if b % 50 == 0 or done == len(todo):
                        rate = done / max(1e-9, time.time() - started)
                        self.log(f"[embed] {done}/{len(todo)} new texts ({rate:.1f}/s)")
            self._save_cache()
        if not keys:
            return np.zeros((0, getattr(self, "hidden", 0)), dtype=np.float32)
        return np.stack([self.cache[k] for k in keys]).astype(np.float32)

    def close(self) -> None:
        """Release the 8B model (GPU memory) once all texts are embedded."""
        self.model = None
        if hasattr(self, "torch") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
