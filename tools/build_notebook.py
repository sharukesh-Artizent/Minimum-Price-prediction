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

**Runtime:** a GPU with ≥ 24 GB (L4 / A10G / A100) or 2 × T4 (Kaggle). Qwen3-8B needs ~16 GB.
Embedding ~35K texts takes roughly 10–25 minutes depending on the GPU; training the heads takes minutes.

**Data:** put `Offers.csv`, `Product Information.csv` and `Sales Order&Line items.csv` anywhere
under `DATA_DIR` (set below) — e.g. upload the `Data Files` folder as a Kaggle dataset or to Google Drive.
"""

SETUP = '''#@title 1. Environment setup
import os, subprocess, sys
CLM_COMMIT = "d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6"
if os.environ.get("RESERVE_SKIP_INSTALL") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.51", "accelerate",
                    "pandas", "pyarrow", "huggingface_hub"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    f"contrastive-lm @ git+https://github.com/Contrastive-LM/CLM@{CLM_COMMIT}"], check=True)
import torch
print("torch", torch.__version__, "| GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or "none")
'''

DATA = '''#@title 2. Where is the data?  (edit DATA_DIR if auto-detection fails)
DATA_DIR = os.environ.get("RESERVE_DATA_DIR", "")
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB and not DATA_DIR:
    from google.colab import drive
    drive.mount("/content/drive")
DATA_ROOTS = [DATA_DIR, "/kaggle/input", "/content/data", "/content/drive/MyDrive", "."]
WORK = os.environ.get("RESERVE_WORK_DIR") or ("/kaggle/working/reserve_clm_run" if os.path.isdir("/kaggle/working")
                                              else "reserve_clm_run")
os.makedirs("reserve_clm", exist_ok=True)
os.makedirs(WORK, exist_ok=True)
print("searching", [r for r in DATA_ROOTS if r], "| outputs ->", WORK)
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

SMOKE = os.environ.get("RESERVE_SMOKE") == "1"   # tiny CPU run used to test the notebook itself
if SMOKE:
    cfg = RunConfig(data_roots=DATA_ROOTS, work_dir=WORK, embed_model="Qwen/Qwen3-0.6B", max_len=512,
                    embed_batch=8, seeds=(1234,), train=TrainConfig(epochs=3, patience=3),
                    replay_max_groups=40, smoke_rows=40, init_ckpt=None)
else:
    from clm.heads import download
    cfg = RunConfig(data_roots=DATA_ROOTS, work_dir=WORK, init_ckpt=download())   # released CLM head
print(cfg)
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
        "band_exact", "band_within_1", "band_mae"]
print("MAIN TEST SET (SKU-level market price)")
display(results["headline_test"].reindex(columns=cols + ["hit_rate_std"]).round(4))
print("FALLBACK TEST SET (family-level market price)")
display(results["headline_fallback"].reindex(columns=cols).round(4))
'''

EVAL2 = '''#@title 10. Offer-level replay, calibration, breakdowns, examples
print("Replay: accept if bid >= reserve, compared with Assurant's actual OFFER_STATUS")
display(results["replay"].round(4))
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
- **band_exact / within_1 / mae**: band accuracy against the band containing the interval midpoint.
- **Replay**: every test-period offer on sampled SKU × warehouse × days, accepted if bid ≥ reserve.
  *lost_sales_rate* = we reject, Assurant accepted; *underpricing_rate* = we accept, Assurant did not.
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
    nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                            "language": "python"},
                                             "accelerator": "GPU"})
    out = ROOT / "notebooks" / "reserve_price_clm.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    print(build())
