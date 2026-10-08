from reserve.cli_tune_phase1 import main


def test_tuning_cli_one_row_per_combination(data_dir, tmp_path):
    table = main(["--data-dir", str(data_dir), "--out-dir", str(tmp_path / "out"),
                  "--train-end", "2025-10-23", "--train-days", "5", "--test-days", "3",
                  "--lams", "0", "1", "--rhos", "1", "--max-iter", "50"])
    assert len(table) == 2
    assert {"lam", "rho", "converged", "val_n"} <= set(table.columns)
    assert (tmp_path / "out" / "phase1_tuning.csv").exists()
