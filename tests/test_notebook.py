import ast
import subprocess
import sys

import nbformat

from conftest import ROOT

MODULES = ["__init__", "config", "bands", "data", "labels", "features", "text", "baselines",
           "evaluate", "report", "embed", "train", "pipeline"]


def test_build_notebook_inlines_current_modules():
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_notebook.py")], check=True)
    nb = nbformat.read(ROOT / "notebooks" / "reserve_price_clm.ipynb", as_version=4)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    for m in MODULES:
        cell = next(s for s in code if s.startswith(f"%%writefile reserve_clm/{m}.py\n"))
        assert cell.split("\n", 1)[1] == (ROOT / "reserve_clm" / f"{m}.py").read_text()
    for s in code:                                     # every non-magic cell must parse
        if not s.startswith("%%"):
            ast.parse(s)
    assert any("evaluate_all(" in s for s in code)
    assert not any("zero-shot" in s.lower() or "zero_shot" in s.lower() for s in code)
