"""End-to-end steps the notebook calls: prepare -> embed -> train -> evaluate."""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .bands import BAND_KEYS, BAND_MIDS, band_of, soft_targets
from .config import VAL_END
from .data import find_data_files, load_offers, load_products, load_sales
from .evaluate import band_ece, calibration_table, decode_bands, interval_metrics
from .features import build_features
from .labels import assign_split, floor_intervals
from .report import (acceptance_table, breakdowns, method_ratios, prediction_examples, replay_table,
                     score_table)
from .text import band_descriptions, state_text
from .train import TrainConfig, predict_proba, save_checkpoint, train_heads

KEYS3 = ["sku", "wh", "day"]


@dataclass
class RunConfig:
    data_roots: list
    work_dir: str = "work"
    embed_model: str = "Qwen/Qwen3-8B"
    max_len: int = 2048
    embed_batch: int = 16
    seeds: tuple = (1234, 1, 2)
    train: TrainConfig = field(default_factory=TrainConfig)
    replay_max_groups: int = 20000
    init_ckpt: str | None = None
    smoke_rows: int | None = None        # subsample labels per split (quick runs only)
    device: str | None = None            # e.g. "cuda:1"; used for embedding and training (default: first GPU)


@dataclass
class Prepared:
    labels: pd.DataFrame
    replay: pd.DataFrame
    offers: pd.DataFrame
    counts: pd.DataFrame


def prepare(cfg: RunConfig, log=print) -> Prepared:
    files = find_data_files(cfg.data_roots)
    log(f"[data] {files}")
    offers = load_offers(files["offers"])
    sales = load_sales(files["sales"])
    products = load_products(files["products"])
    floors = floor_intervals(offers)
    labels = floors.assign(split=assign_split(floors.day))
    test_groups = offers.loc[offers.day >= VAL_END, KEYS3].drop_duplicates()
    replay = test_groups.sample(n=min(cfg.replay_max_groups, len(test_groups)), random_state=0)
    if cfg.smoke_rows:
        labels = pd.concat([g.sample(min(len(g), cfg.smoke_rows), random_state=0)
                            for _, g in labels.groupby("split")])
        replay = replay.head(cfg.smoke_rows)
    log(f"[data] offers {len(offers):,}  sales lines {len(sales):,}  floor days {len(floors):,}  "
        f"labels used {len(labels):,}  replay groups {len(replay):,}")
    targets = pd.concat([labels[KEYS3], replay[KEYS3]]).drop_duplicates()
    feats = build_features(offers, sales, products, floors, targets)
    labels = labels.merge(feats, on=KEYS3, how="left")
    for c in ("lo", "hi", "mid"):
        labels[f"{c}_ratio"] = labels[c] / labels.ref_price
    labels["true_band"] = band_of(labels.mid_ratio.fillna(1.0))
    priced = (labels.ref_source != "none").to_numpy()
    labels["text"] = ""
    labels.loc[priced, "text"] = [state_text(r) for r in labels[priced].itertuples(index=False)]
    replay = replay.merge(feats, on=KEYS3, how="left")
    replay = replay[replay.ref_source != "none"].reset_index(drop=True)
    replay["text"] = [state_text(r) for r in replay.itertuples(index=False)]
    counts = labels.pivot_table(index="split", columns="ref_source", values="lo", aggfunc="size", fill_value=0)
    for c in ("sku", "family", "none"):
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[["sku", "family", "none"]].reindex(["train", "val", "test"]).fillna(0).astype(int)
    return Prepared(labels.reset_index(drop=True), replay, offers, counts)


def views(prep: Prepared) -> dict[str, pd.DataFrame]:
    L = prep.labels
    main, fam = L.ref_source == "sku", L.ref_source == "family"
    out = {"train": L[main & (L.split == "train")], "val": L[main & (L.split == "val")],
           "test": L[main & (L.split == "test")], "test_fallback": L[fam & (L.split == "test")]}
    return {k: v.reset_index(drop=True) for k, v in out.items()}


def embed_all(prep: Prepared, cfg: RunConfig, embedder=None, log=print):
    from .embed import QwenEmbedder
    V = views(prep)
    band_texts = [band_descriptions()[k] for k in BAND_KEYS]
    texts = list(dict.fromkeys(band_texts + [t for v in V.values() for t in v.text] + list(prep.replay.text)))
    created = embedder is None
    if created:
        slug = cfg.embed_model.replace("/", "_")
        embedder = QwenEmbedder(cfg.embed_model, cfg.max_len, cfg.embed_batch, log=log, device=cfg.device,
                                cache_path=os.path.join(cfg.work_dir, f"embed_cache_{slug}_{cfg.max_len}.npz"))
    log(f"[embed] {len(texts):,} unique texts")
    vecs = embedder.embed(texts)
    index = {t: i for i, t in enumerate(texts)}

    def get(ts):
        return vecs[[index[t] for t in ts]] if len(ts) else np.zeros((0, vecs.shape[1]), dtype=np.float32)

    emb = {k: get(list(v.text)) for k, v in V.items()}
    emb["replay"] = get(list(prep.replay.text))
    emb["bands"] = get(band_texts)
    if created:
        embedder.close()
    return V, emb


def run_training(V: dict, E: dict, cfg: RunConfig, log=print) -> list[dict]:
    tr, val = V["train"], V["val"]
    t_tr = soft_targets(tr.lo_ratio, tr.hi_ratio)

    def val_score(p):
        price = val.ref_price.to_numpy() * BAND_MIDS[decode_bands(p)]
        return interval_metrics(price, val.lo, val.hi, val.mid)["interval_hit_rate"]

    runs = []
    for seed in cfg.seeds:
        res = train_heads(E["train"], t_tr, E["bands"], E["val"], val_score,
                          dataclasses.replace(cfg.train, seed=seed), init_ckpt=cfg.init_ckpt,
                          device=cfg.device, log=log)
        save_checkpoint(res, os.path.join(cfg.work_dir, f"head_seed{seed}.pt"))
        probs = {k: predict_proba(res.model, E[k], E["bands"]) for k in ("val", "test", "test_fallback", "replay")}
        runs.append({"seed": seed, "result": res, "probs": probs})
    return runs


def evaluate_all(prep: Prepared, V: dict, runs: list[dict]) -> dict[str, pd.DataFrame]:
    train_df = V["train"]
    training = pd.DataFrame([{"seed": r["seed"], "warm_start": r["result"].warm_start,
                              "best_epoch": r["result"].best_epoch,
                              "val_interval_hit_rate": r["result"].best_score} for r in runs])
    test, fb = V["test"], V["test_fallback"]
    ratios, ens = method_ratios(test, train_df, runs, "test")
    fb_ratios, _ = method_ratios(fb, train_df, runs, "test_fallback")
    offers_test = prep.offers[prep.offers.day >= VAL_END]

    def hit(df, r):
        p = df.ref_price.to_numpy() * r
        return ((p >= df.lo.to_numpy()) & (p <= df.hi.to_numpy())).astype(float)

    ens_ratio = ratios["CLM ensemble (median band)"]
    return {
        "training": training,
        "headline_test": score_table(test, ratios),
        "headline_fallback": score_table(fb, fb_ratios),
        "acceptance_test": acceptance_table(test, offers_test, ratios),
        "acceptance_fallback": acceptance_table(fb, offers_test, fb_ratios),
        "replay": replay_table(prep.replay, offers_test, train_df, runs),
        "calibration": calibration_table(ens.max(1) if len(ens) else [], hit(test, ens_ratio)),
        "band_ece": pd.DataFrame([{"split": "test", "band_ece": band_ece(ens, test.true_band.to_numpy())}]),
        "breakdowns": breakdowns(test, {"CLM ensemble hit": hit(test, ens_ratio),
                                        "B0 carry-forward hit": hit(test, ratios["B0 carry-forward"])}),
        "examples": prediction_examples(test, ens),
    }


def save_results(results: dict[str, pd.DataFrame], work_dir: str) -> None:
    out = os.path.join(work_dir, "results")
    os.makedirs(out, exist_ok=True)
    for name, t in results.items():
        t.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    head = results["headline_test"].set_index("method").interval_hit_rate.to_dict()
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"test_interval_hit_rate": head}, f, indent=1)
