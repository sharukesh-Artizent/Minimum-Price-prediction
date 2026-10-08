import pandas as pd
import pytest

from factories import synthetic_world
from reserve.config import Params
from reserve.pipeline import prepare_from_frames, run_window
from reserve.report import build_report, criteria, md_table


def test_md_table():
    s = md_table(pd.DataFrame({"a": [1.5], "b": ["x"]}))
    assert s.splitlines()[0] == "| a | b |"
    assert "1.500" in s


@pytest.fixture(scope="module")
def window():
    o, s, pr = synthetic_world(1)
    D = prepare_from_frames(o, s, pr, Params())
    return run_window(D, "2026-03-01", "2026-04-09", Params(), extras=True)


def test_criteria_rows(window):
    c = criteria(window)
    assert list(c.id) == ["F1", "F2", "F3", "S1", "S2", "S3", "S4"]
    assert c["pass"].dtype == bool


def test_build_report_writes_files(window, tmp_path):
    path = build_report(window, tmp_path, "t")
    text = path.read_text(encoding="utf-8")
    for section in ["## Success criteria", "## Policy comparison", "## Pareto segments",
                    "## Deviation from current floor", "## Frontier", "## Sensitivity",
                    "## Sample reserves"]:
        assert section in text, section
    assert (tmp_path / "frontier_t.png").exists()
    assert (tmp_path / "counter_curve_t.png").exists()
