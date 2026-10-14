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
    """The model is loaded only when a text is not in the cache yet, so a re-run whose texts are
    all cached never touches the GPU (useful on persistent storage)."""

    def __init__(self, model_name: str, max_len: int = 2048, batch_size: int = 16,
                 cache_path: str | None = None, device: str | None = None, log=print):
        """device: e.g. "cuda:1" to pin the model to one GPU; default = first GPU, else CPU."""
        self.model_name, self.log, self.device_name = model_name, log, device
        self.max_len, self.batch_size, self.cache_path = max_len, batch_size, cache_path
        self.model = self.tok = None
        self._hidden = None
        self._load_cache(cache_path)

    @property
    def hidden(self) -> int:
        if self._hidden is None:
            from transformers import AutoConfig
            self._hidden = int(AutoConfig.from_pretrained(self.model_name).hidden_size)
        return self._hidden

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.log(f"[embed] loading {self.model_name}")
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        device = torch.device(self.device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        else:
            dtype = torch.float32
        # Whole model on one device: no device_map, so `accelerate` is not required.
        self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=dtype).to(device).eval()
        self._hidden = int(self.model.config.hidden_size)
        self.device = self.model.get_input_embeddings().weight.device
        pad = self.tok.pad_token_id
        self.pad_id = pad if pad is not None else (self.tok.eos_token_id or 0)

    def _load_cache(self, path: str | None) -> None:
        self.cache = {}
        if not path:
            return
        if not os.path.exists(path) and not path.endswith(".npz") and os.path.exists(path + ".npz"):
            path = path + ".npz"
        if os.path.exists(path):
            with np.load(path) as z:
                self.cache = dict(zip(z["keys"].tolist(), z["vecs"]))
            if self.cache:
                self._hidden = int(next(iter(self.cache.values())).shape[0])

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
        keys = [_key(t) for t in texts]
        todo = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in self.cache))
        self.log(f"[embed] {len(set(keys)) - len(todo):,} texts from cache, {len(todo):,} to embed")
        if todo:
            import torch
            if self.model is None:
                self._load_model()
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
            return np.zeros((0, self.hidden), dtype=np.float32)
        return np.stack([self.cache[k] for k in keys]).astype(np.float32)

    def close(self) -> None:
        """Release the 8B model (GPU memory) once all texts are embedded."""
        if self.model is None:
            return
        self.model = None
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
