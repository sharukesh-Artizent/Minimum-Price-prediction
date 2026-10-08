from reserve import config


def test_raw_files_exist():
    assert config.OFFERS_CSV.exists(), config.OFFERS_CSV
    assert config.SALES_CSV.exists(), config.SALES_CSV
    assert config.PRODUCT_CSV.exists(), config.PRODUCT_CSV


def test_levels_coarse_to_fine():
    assert config.LEVELS[0] == ["category"]
    assert config.LEVELS[-1] == ["sku", "wh"]


def test_params_grid_defaults():
    p = config.Params()
    assert (p.grid_lo, p.grid_hi, p.grid_step) == (0.70, 1.15, 0.01)
    assert p.trust_region == 0.15
    assert p.holding_cost_pct_per_day == 0.001
