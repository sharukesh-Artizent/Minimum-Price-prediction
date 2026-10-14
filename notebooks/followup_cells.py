# Follow-up experiments for reserve_price_clm.ipynb.
# Paste each "# %%" block into a new cell at the end of the notebook, after section 10.
# Reuses prep, V, E, runs, cfg from the session. Nothing is re-embedded.
# If the kernel restarted: run notebook cells 1-6 again (not 7 "Embed"), then start here;
# cell 1 rebuilds E from the embedding cache and retrains `runs` if needed (a few minutes).

# %% Cell 1 - imports, and rebuild E / runs from cache if the kernel restarted
import hashlib, math, os
import numpy as np, pandas as pd, torch
from IPython.display import display
from scipy.stats import binomtest
from reserve_clm.bands import BAND_KEYS, BAND_LABELS, BAND_MIDS, N_BANDS, band_of, soft_targets
from reserve_clm.baselines import carry_forward_ratio, fixed_ratio, majority_band
from reserve_clm.evaluate import band_metrics, decode_bands, interval_metrics
from reserve_clm.text import band_descriptions
from reserve_clm.train import TrainConfig, build_model, predict_proba, soft_infonce, train_heads

if "E" not in globals():
    cache = os.path.join(cfg.work_dir, f"embed_cache_{cfg.embed_model.replace('/', '_')}_{cfg.max_len}.npz")
    with np.load(cache) as z:
        _vec = dict(zip(z["keys"].tolist(), z["vecs"]))
    _get = lambda ts: np.stack([_vec[hashlib.sha1(t.encode()).hexdigest()] for t in ts]).astype(np.float32)
    E = {k: _get(list(v.text)) for k, v in V.items()}
    E["replay"] = _get(list(prep.replay.text))
    E["bands"] = _get([band_descriptions()[k] for k in BAND_KEYS])
    print("E rebuilt from", cache)
if "runs" not in globals():
    from reserve_clm.pipeline import run_training
    runs = run_training(V, E, cfg)

TR, VA, TE, FB = V["train"], V["val"], V["test"], V["test_fallback"]
FALLBACK = float(TR.mid_ratio.median())
MAJ = majority_band(TR.true_band.to_numpy())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(cfg.work_dir, "results_followup")
os.makedirs(OUT, exist_ok=True)


def hits(df, ratio):
    p = df.ref_price.to_numpy() * ratio
    return (p >= df.lo.to_numpy()) & (p <= df.hi.to_numpy())


def scores(df, ratio):
    m = interval_metrics(df.ref_price.to_numpy() * ratio, df.lo, df.hi, df.mid)
    m.update(band_metrics(band_of(ratio), df.true_band.to_numpy()))
    return {k: m[k] for k in ("n", "interval_hit_rate", "too_high_rate", "too_low_rate", "mape_vs_mid",
                              "bias_vs_mid", "band_exact", "band_within_1", "band_mae")}


def baseline_ratios(df):
    return {"B0 carry-forward": carry_forward_ratio(df, FALLBACK),
            "B1 fixed ratio": fixed_ratio(len(df), FALLBACK),
            "B2 majority band": fixed_ratio(len(df), BAND_MIDS[MAJ])}


def ens(probs_list):
    return np.mean(probs_list, axis=0)


CLM_ORIG = {k: ens([r["probs"][k] for r in runs]) for k in ("val", "test", "test_fallback")}
print({k: len(v) for k, v in V.items()}, "| device:", DEVICE)

# %% Cell 2 - is CLM's lead real? Paired McNemar test + per-row predictions saved to CSV
R = {"CLM ensemble": BAND_MIDS[decode_bands(CLM_ORIG["test"])], **baseline_ratios(TE)}
H = {k: hits(TE, r) for k, r in R.items()}


def mcnemar(a, b):
    """a, b: per-row hit arrays. Exact McNemar on the discordant pairs."""
    a_only, b_only = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return a_only, b_only, p


def paired_bootstrap(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = a.astype(float) - b.astype(float)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return np.percentile(boots, [2.5, 97.5])


rows = []
for k in ("B0 carry-forward", "B1 fixed ratio", "B2 majority band"):
    a_only, b_only, p = mcnemar(H["CLM ensemble"], H[k])
    lo, hi = paired_bootstrap(H["CLM ensemble"], H[k])
    rows.append({"CLM vs": k, "CLM hit": H["CLM ensemble"].mean(), "baseline hit": H[k].mean(),
                 "gain (pts)": 100 * (H["CLM ensemble"].mean() - H[k].mean()),
                 "95% CI of gain (pts)": f"{100 * lo:+.1f} to {100 * hi:+.1f}",
                 "CLM right / baseline wrong": a_only, "baseline right / CLM wrong": b_only, "McNemar p": p})
sig = pd.DataFrame(rows)
display(sig.round(4))

per_row = TE[["sku", "wh", "day", "ref_price", "lo", "hi", "mid", "true_band"]].copy()
per_row["clm_confidence"] = CLM_ORIG["test"].max(1)
per_row["clm_top_band"] = [BAND_LABELS[b] for b in CLM_ORIG["test"].argmax(1)]
for k, r in R.items():
    per_row[f"reserve {k}"] = (TE.ref_price.to_numpy() * r).round(2)
    per_row[f"hit {k}"] = H[k]
per_row.to_csv(os.path.join(OUT, "test_per_row_predictions.csv"), index=False)
sig.to_csv(os.path.join(OUT, "significance.csv"), index=False)
print("saved per-row predictions and significance to", OUT)

# %% Cell 3 - more stable training: (a) select epoch by validation log-loss, (b) weight averaging
T_TR = soft_targets(TR.lo_ratio, TR.hi_ratio)
T_VA = soft_targets(VA.lo_ratio, VA.hi_ratio)


def neg_logloss(p):
    """Higher is better: minus cross-entropy of predictions against the validation soft targets."""
    return float((T_VA * np.log(np.clip(p, 1e-9, 1))).sum(1).mean())


def train_weight_avg(seed, epochs=25, avg_last=10):
    """Fixed epoch budget, average the weights of the last `avg_last` epochs (no noisy early stopping)."""
    tc = TrainConfig(**{**cfg.train.__dict__, "seed": seed})
    torch.manual_seed(seed)
    model, _, _ = build_model(E["train"].shape[1], tc, cfg.init_ckpt)
    model.to(DEVICE)
    x = torch.as_tensor(E["train"], dtype=torch.float32, device=DEVICE)
    t = torch.as_tensor(T_TR, dtype=torch.float32, device=DEVICE)
    c = torch.as_tensor(E["bands"], dtype=torch.float32, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc.lr, total_steps=math.ceil(len(x) / tc.batch) * epochs,
                                                pct_start=0.1, anneal_strategy="cos")
    gen = torch.Generator().manual_seed(seed)
    avg, n = None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(len(x), generator=gen).split(tc.batch):
            idx = idx.to(DEVICE)
            loss = soft_infonce(model.logits(x[idx], c), t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        if ep > epochs - avg_last:
            sd = {k: v.detach().float().clone() for k, v in model.state_dict().items()}
            avg = sd if avg is None else {k: avg[k] + sd[k] for k in avg}
            n += 1
    model.load_state_dict({k: v / n for k, v in avg.items()})
    model.eval()
    return model


VARIANTS = {"CLM original (val hit-rate)": CLM_ORIG}
probs_ll, probs_avg = [], []
for seed in cfg.seeds:
    tc = TrainConfig(**{**cfg.train.__dict__, "seed": seed, "patience": 15, "epochs": 80})
    res = train_heads(E["train"], T_TR, E["bands"], E["val"], neg_logloss, tc, init_ckpt=cfg.init_ckpt,
                      log=lambda *_: None)
    print(f"seed {seed}: log-loss selection best epoch {res.best_epoch}")
    probs_ll.append({k: predict_proba(res.model, E[k], E["bands"]) for k in ("val", "test", "test_fallback")})
    m = train_weight_avg(seed)
    probs_avg.append({k: predict_proba(m, E[k], E["bands"]) for k in ("val", "test", "test_fallback")})
VARIANTS["CLM log-loss selection"] = {k: ens([p[k] for p in probs_ll]) for k in ("val", "test", "test_fallback")}
VARIANTS["CLM weight averaging"] = {k: ens([p[k] for p in probs_avg]) for k in ("val", "test", "test_fallback")}

for name, p in VARIANTS.items():
    vh = hits(VA, BAND_MIDS[decode_bands(p["val"])]).mean()
    th = hits(TE, BAND_MIDS[decode_bands(p["test"])]).mean()
    print(f"{name:32s} val hit {vh:.4f} | test hit {th:.4f}")

# %% Cell 4 - routing: use CLM when it is confident, otherwise a rule (threshold chosen on validation)
def best_route(p_val, rule_val_ratio, p_test, rule_test_ratio):
    conf_v, conf_t = p_val.max(1), p_test.max(1)
    clm_v, clm_t = BAND_MIDS[decode_bands(p_val)], BAND_MIDS[decode_bands(p_test)]
    grid = np.quantile(conf_v, np.linspace(0, 1, 21))
    val_hit = [hits(VA, np.where(conf_v >= g, clm_v, rule_val_ratio)).mean() for g in grid]
    g = grid[int(np.argmax(val_hit))]
    return g, np.where(conf_t >= g, clm_t, rule_test_ratio), float((conf_t >= g).mean())


best_name = max(VARIANTS, key=lambda k: hits(VA, BAND_MIDS[decode_bands(VARIANTS[k]["val"])]).mean())
ROUTES = {}
for rule in ("B1 fixed ratio", "B2 majority band"):
    g, ratio, share = best_route(VARIANTS[best_name]["val"], baseline_ratios(VA)[rule],
                                 VARIANTS[best_name]["test"], baseline_ratios(TE)[rule])
    ROUTES[f"Route: {best_name} if conf>={g:.3f} else {rule}"] = ratio
    print(f"{rule}: threshold {g:.3f}, CLM used on {share:.0%} of test rows, test hit {hits(TE, ratio).mean():.4f}")

# %% Cell 5 - tabular benchmark on the SAME features: LightGBM (does the 8B model earn its keep?)
import lightgbm as lgb

NUM = ["ref_price", "ref_n", "ref_trend", "n_offers_7", "n_offers_30", "n_bidders_30", "units_req_30",
       "bid_med_ratio_30", "bid_top_ratio_30", "accept_share_30", "units_sold_7", "units_sold_30",
       "days_since_last_sale", "sale_price_ratio_30", "last_lo_ratio", "last_hi_ratio", "days_since_last_floor"]
CAT = ["manufacturer", "model", "capacity", "grade", "carrier", "lock_status", "wh"]
_all = pd.concat([TR, VA, TE, FB])
_cats = {c: pd.Index(_all[c].astype(str).unique()) for c in CAT}


def tab(df):
    X = df[NUM].astype(float).copy()
    X["weekday"] = df.day.dt.weekday
    for c in CAT:
        X[c] = pd.Categorical(df[c].astype(str), categories=_cats[c])
    return X


Xtr, Xva, Xte, Xfb = tab(TR), tab(VA), tab(TE), tab(FB)
params = dict(learning_rate=0.03, num_leaves=31, min_child_samples=30, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)
cb = [lgb.early_stopping(100, verbose=False)]

clf = lgb.train({**params, "objective": "multiclass", "num_class": N_BANDS},
                lgb.Dataset(Xtr, TR.true_band.astype(int)), 3000,
                valid_sets=[lgb.Dataset(Xva, VA.true_band.astype(int))], callbacks=cb)
reg = lgb.train({**params, "objective": "regression_l1"},
                lgb.Dataset(Xtr, TR.mid_ratio), 3000, valid_sets=[lgb.Dataset(Xva, VA.mid_ratio)], callbacks=cb)
LGB = {"LightGBM band classifier": {k: BAND_MIDS[decode_bands(clf.predict(X))] for k, X in (("test", Xte), ("test_fallback", Xfb))},
       "LightGBM ratio regression": {k: reg.predict(X) for k, X in (("test", Xte), ("test_fallback", Xfb))}}
print("trees: classifier", clf.best_iteration, "| regression", reg.best_iteration)
imp = pd.Series(reg.feature_importance("gain"), index=Xtr.columns).sort_values(ascending=False)
display((imp / imp.sum()).head(12).round(3).rename("share of gain (regression)"))

# %% Cell 6 - final comparison on the same test rows, plus significance vs the original CLM
def table(df, split):
    rows = {k: scores(df, r) for k, r in baseline_ratios(df).items()}
    for name, p in VARIANTS.items():
        rows[f"{name} (median band)"] = scores(df, BAND_MIDS[decode_bands(p[split])])
        rows[f"{name} (expected ratio)"] = scores(df, p[split] @ BAND_MIDS)
    for name, r in LGB.items():
        rows[name] = scores(df, r[split])
    if split == "test":
        for name, r in ROUTES.items():
            rows[name] = scores(df, r)
    return pd.DataFrame(rows).T.sort_values("interval_hit_rate", ascending=False)


main, fallback = table(TE, "test"), table(FB, "test_fallback")
print("MAIN TEST SET (SKU-level market price) - sorted by interval hit rate")
display(main.round(4))
print("FALLBACK TEST SET (family-level market price)")
display(fallback.round(4))

ref = hits(TE, BAND_MIDS[decode_bands(CLM_ORIG["test"])])
cmp_rows = []
for name in main.index[:6]:
    if name.startswith("Route"):
        h = hits(TE, ROUTES[name])
    elif name in LGB:
        h = hits(TE, LGB[name]["test"])
    elif name.startswith("B"):
        h = hits(TE, baseline_ratios(TE)[name])
    else:
        base, how = name.rsplit(" (", 1)
        p = VARIANTS[base]["test"]
        h = hits(TE, BAND_MIDS[decode_bands(p)] if how.startswith("median") else p @ BAND_MIDS)
    a_only, b_only, p_val = mcnemar(h, ref)
    cmp_rows.append({"method": name, "hit": h.mean(), "gain vs original CLM (pts)": 100 * (h.mean() - ref.mean()),
                     "McNemar p": p_val})
display(pd.DataFrame(cmp_rows).round(4))
main.to_csv(os.path.join(OUT, "comparison_main.csv"))
fallback.to_csv(os.path.join(OUT, "comparison_fallback.csv"))
print("saved to", OUT)
