"""Execute the generated notebook end to end in smoke mode (CPU, Qwen3-0.6B, subsampled data)."""
import asyncio
import os
import pathlib
import sys
import time

# Ensure Windows Store Python packages are on sys.path and PYTHONPATH for kernel subprocess
_win_local = pathlib.Path.home() / "AppData/Local/Packages"
_extra = [str(p) for p in _win_local.glob("PythonSoftwareFoundation.Python.3.13_*/LocalCache/local-packages/Python313/site-packages")]
for _p in _extra:
    if _p not in sys.path:
        sys.path.append(_p)
if _extra:
    _cur_pypath = os.environ.get("PYTHONPATH", "")
    _paths = _extra + ([_cur_pypath] if _cur_pypath else [])
    os.environ["PYTHONPATH"] = os.pathsep.join(_paths)

import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parents[1]

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

work = ROOT / "work" / "smoke"
work.mkdir(parents=True, exist_ok=True)
os.environ.update(RESERVE_SMOKE="1", RESERVE_SKIP_INSTALL="1", RESERVE_DATA_DIR=str(ROOT.parents[1] / "Data"),
                  RESERVE_WORK_DIR=str(work / "run"))
nb = nbformat.read(ROOT / "notebooks" / "reserve_price_clm.ipynb", as_version=4)
started = time.time()
try:
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(work)}}).execute()
finally:
    nbformat.write(nb, work / "executed.ipynb")
print(f"smoke run OK in {time.time() - started:.0f}s -> {work / 'executed.ipynb'}")
