import json

from reserve.cli_phase1 import main


def test_phase1_cli_writes_outputs(data_dir, tmp_path):
    out = tmp_path / "out"
    metrics = main(["--data-dir", str(data_dir), "--out-dir", str(out),
                    "--train-end", "2025-10-23", "--train-days", "5", "--test-days", "3"])
    assert (out / "phase1_floors.csv").exists()
    saved = json.loads((out / "phase1_metrics.json").read_text())
    assert saved["graph"]["nodes"] == 1
    assert metrics["admm_fused"]["overall"]["n"] == 1
