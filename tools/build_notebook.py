import pathlib
import sys

# Ensure Windows Store Python packages are on sys.path if missing in subprocess
_win_local = pathlib.Path.home() / "AppData/Local/Packages"
for _p in _win_local.glob("PythonSoftwareFoundation.Python.3.13_*/LocalCache/local-packages/Python313/site-packages"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULES = ["__init__", "config", "bands", "data", "labels", "features", "text", "baselines",
           "evaluate", "report", "embed", "train", "pipeline"]

INTRO = """# Reserve price prediction with CLM-8B fine-tuning (POC)

Predicts a **daily reserve (minimum) price band per SKU × warehouse** by fine-tuning
[CLM-8B](https://github.com/Contrastive-LM/CLM) (frozen Qwen3-8B reader + small trained heads),
and scores it against Assurant's own historical floors.

**What the label is.** On a day where the same SKU × warehouse had both an accepted and a
non-accepted (countered / rejected) offer, Assurant's floor lies between the highest bid it turned
down and the lowest bid it accepted. The model learns to place today's reserve in that interval,
so a perfect score means *matches Assurant's current floor*, not *optimal price*.

**Inputs (all from days before the target day):** product attributes, a 30-day market reference
price, bids (counts, buyers, units, median/top bid vs market, accepted share), sales velocity,
Assurant's previous floor, weekday. Inventory and Buyer Profile files are not used (no date
overlap / label leakage).

**Candidates:** 14 reserve bands (multiples of the market price) described in English.
**Training:** CLM's soft-target InfoNCE on the heads, warm-started from the released CLM head,
3 seeds, early stopping on February validation. **Test:** March–April 2026.

**Runtime:** a GPU with ≥ 24 GB (L4 / A10G / A100) or 2 × 16 GB GPUs. Qwen3-8B needs ~16 GB.
Embedding ~35K texts takes roughly 10–25 minutes depending on the GPU; training the heads takes minutes.

**Data:** `Offers`, `Product Information` and `Sales Order&Line items` (`.csv` or `.xlsx`; CSV is used
when both exist and loads much faster) anywhere under `DATA_DIR` = `/dist_home/suryansh/sharukesh/Work`.
**Outputs** (embedding cache, trained heads, result tables) go to `DATA_DIR/reserve_clm_run`, so they
survive restarts: re-running the notebook reuses the cached embeddings instead of embedding again.
"""

SETUP = '''#@title 1. Environment setup
import os, subprocess, sys
CLM_COMMIT = "d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6"
if os.environ.get("RESERVE_SKIP_INSTALL") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.51", "accelerate",
                    "pandas", "pyarrow", "openpyxl", "huggingface_hub"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    f"contrastive-lm @ git+https://github.com/Contrastive-LM/CLM@{CLM_COMMIT}"], check=True)
import torch
print("torch", torch.__version__, "| GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or "none")
'''

DATA = '''#@title 2. Data and output folders (persistent server storage)
DATA_DIR = os.environ.get("RESERVE_DATA_DIR", "/dist_home/suryansh/sharukesh/Work")
WORK = os.environ.get("RESERVE_WORK_DIR", os.path.join(DATA_DIR, "reserve_clm_run"))
assert os.path.isdir(DATA_DIR), f"data folder not found: {DATA_DIR}"
DATA_ROOTS = [DATA_DIR]
os.makedirs("reserve_clm", exist_ok=True)
os.makedirs(WORK, exist_ok=True)
print("data from", DATA_DIR, "| outputs ->", WORK)
'''

CONFIG = '''#@title 4. Run configuration
import pandas as pd
from IPython.display import display
from reserve_clm.pipeline import RunConfig, prepare, views, embed_all, run_training, evaluate_all, save_results
from reserve_clm.train import TrainConfig
from reserve_clm.bands import BAND_KEYS, BAND_LABELS
from reserve_clm.text import band_descriptions
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.width", 200)

GPU = 1                                           # which GPU to use (0, 1, ...); falls back to CPU if there is none
DEVICE = f"cuda:{GPU}" if torch.cuda.device_count() > GPU else ("cuda" if torch.cuda.is_available() else "cpu")
SMOKE = os.environ.get("RESERVE_SMOKE") == "1"   # tiny CPU run used to test the notebook itself
if SMOKE:
    cfg = RunConfig(data_roots=DATA_ROOTS, work_dir=WORK, embed_model="Qwen/Qwen3-0.6B", max_len=512,
                    embed_batch=8, seeds=(1234,), train=TrainConfig(epochs=3, patience=3),
                    replay_max_groups=40, smoke_rows=40, init_ckpt=None, device=DEVICE)
else:
    from clm.heads import download
    cfg = RunConfig(data_roots=DATA_ROOTS, work_dir=WORK, init_ckpt=download(), device=DEVICE)  # released CLM head
print(cfg)
if DEVICE.startswith("cuda"):
    free, total = torch.cuda.mem_get_info(torch.device(DEVICE))
    print(f"{DEVICE}: {torch.cuda.get_device_name(torch.device(DEVICE))}, {free / 2**30:.1f} GB free of {total / 2**30:.1f} GB "
          "(Qwen3-8B needs about 17 GB)")
'''

PREP = '''#@title 5. Load data, build labels and features
prep = prepare(cfg)
print("Labelled SKU x warehouse x days by split and reference source")
print("(sku = main set, family = fallback set tested separately, none = cannot be priced)")
display(prep.counts)
V = views(prep)
print({k: len(v) for k, v in V.items()}, "| replay groups:", len(prep.replay))
'''

SHOW = '''#@title 6. What CLM reads: one state and the 14 band candidates
print(V["test"].text.iloc[0])
display(pd.DataFrame({"band": BAND_LABELS, "description": [band_descriptions()[k] for k in BAND_KEYS]}))
'''

EMBED = '''#@title 7. Embed every text once with Qwen3-8B (cached to WORK)
V, E = embed_all(prep, cfg)
import gc; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
print({k: v.shape for k, v in E.items()})
'''

TRAIN = '''#@title 8. Fine-tune the CLM heads (one run per seed)
runs = run_training(V, E, cfg)
'''

EVAL = '''#@title 9. Evaluate on March-April 2026 and save results
results = evaluate_all(prep, V, runs)
save_results(results, WORK)
print("Training"); display(results["training"])
cols = ["method", "n", "interval_hit_rate", "too_high_rate", "too_low_rate", "mape_vs_mid", "bias_vs_mid",
        "band_exact", "band_within_1", "band_within_3", "band_within_5", "band_mae"]
print("MAIN TEST SET (SKU-level market price)")
display(results["headline_test"].reindex(columns=cols + ["hit_rate_std"]).round(4))
print("FALLBACK TEST SET (family-level market price)")
display(results["headline_fallback"].reindex(columns=cols).round(4))
'''

EVAL2 = '''#@title 10. Daily acceptance, replay, calibration, breakdowns, examples
acc_cols = ["method", "n_days", "n_offers", "actual_accept_rate", "pred_accept_rate", "agreement",
            "accept_precision", "accept_recall", "day_exact_match", "day_count_mae",
            "accepted_per_day_actual", "accepted_per_day_pred", "lost_sales_rate", "underpricing_rate"]
print("DAILY ACCEPTANCE on labelled test days: reserve = base price for the day, accept every offer with bid >= reserve")
print("Main test set (SKU-level market price)")
display(results["acceptance_test"].reindex(columns=acc_cols).round(4))
print("Fallback test set (family-level market price)")
display(results["acceptance_fallback"].reindex(columns=acc_cols).round(4))
print("Replay on ALL sampled test-period days (including days where every offer was accepted or none was)")
display(results["replay"].reindex(columns=["reference"] + acc_cols).round(4))
print("Calibration of the CLM ensemble (test, main set)")
display(results["calibration"].round(4)); display(results["band_ece"].round(4))
print("Interval hit rate by segment (test, main set)")
display(results["breakdowns"].round(4))
print("Example predictions")
display(results["examples"])
print("All tables saved to", os.path.join(WORK, "results"))
'''

READ = """## How to read the results

- **interval_hit_rate** (main metric): share of test days where the predicted reserve lies inside
  Assurant's floor interval [highest bid turned down, lowest bid accepted].
- **too_high / too_low**: reserve above the lowest accepted bid (would have lost that sale) /
  below the highest bid Assurant turned down (would have accepted a bid Assurant did not).
- **mape / bias vs mid**: price error against the interval midpoint; positive bias = reserve too high.
- **band_exact / within_1 / within_3 / within_5 / mae**: share of days whose predicted band is 0, ≤1, ≤3, ≤5
  bands away from the band containing the interval midpoint (each band = 2.5% of market price, so
  ±3 bands ≈ ±7.5% and ±5 bands ≈ ±12.5%).
- **Daily acceptance**: the predicted reserve is that day's base price; every offer that day with bid ≥ reserve is
  accepted. *agreement* = share of offers where we match Assurant's decision; *accept_precision* = of the offers
  we accept, share Assurant accepted; *accept_recall* = of the offers Assurant accepted, share we accept;
  *day_exact_match* = share of days where every offer's decision matches; *day_count_mae* = average gap between
  how many offers we accept and how many Assurant accepted per day.
  *lost_sales_rate* = we reject, Assurant accepted; *underpricing_rate* = we accept, Assurant did not.
- The first daily-acceptance tables use the labelled test days (they have both accepted and turned-down offers);
  the replay table uses a sample of all test-period days.
- **B0 carry-forward** (previous floor) is the strongest rule baseline; CLM must beat it to be useful.
- Fallback rows use a family-level market price and are reported separately; the model never trains on them.
"""


def build() -> pathlib.Path:
    cells = [new_markdown_cell(INTRO), new_code_cell(SETUP), new_code_cell(DATA),
             new_markdown_cell("## 3. Project code\nThe tested modules, written to `reserve_clm/` (collapsed).")]
    for m in MODULES:
        src = (ROOT / "reserve_clm" / f"{m}.py").read_text(encoding="utf-8")
        cell = new_code_cell(f"%%writefile reserve_clm/{m}.py\n{src}")
        cell.metadata["jupyter"] = {"source_hidden": True}
        cell.metadata["cellView"] = "form"
        cells.append(cell)
    cells += [new_code_cell(CONFIG), new_code_cell(PREP), new_code_cell(SHOW), new_code_cell(EMBED),
              new_code_cell(TRAIN), new_code_cell(EVAL), new_code_cell(EVAL2), new_markdown_cell(READ)]
    for idx, cell in enumerate(cells):
        cell.id = f"cell_{idx:03d}"
    nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                            "language": "python"},
                                             "accelerator": "GPU"})
    out = ROOT / "notebooks" / "reserve_price_clm.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    print(build())
