import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT.parents[1] / "Data"          # repo-root Data/ (ROOT = poc/CML)
