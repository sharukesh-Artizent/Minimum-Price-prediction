"""B0: Assurant's implicit floor (as a ratio to m-hat) learned from interval-censored decisions."""
import numpy as np
import pandas as pd
import xgboost as xgb

from reserve.frame import CAT_COLS, apply_categories, fit_categories

FLOOR_NUM = ["age_days", "tau", "nb", "ub", "dow", "px_n_L3", "px_n_L5", "units_sold_w"]
XGB_PARAMS = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "aft_loss_distribution": "normal",
    # Scale controls the assumed standard deviation in log(floor ratio).
    # Default 0.2 under-penalized intervals and led to biased medians on censored bounds;
    # 0.01 provides tight interval adherence, recovering true floors within tolerance
    # and enabling B0 policy recovery (>80% decision agreement) in the backtest pipeline.
    "aft_loss_distribution_scale": 0.01,
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 20,
}


class FloorModel:
    def __init__(self, seed: int = 0, scale: float | None = None):
        self.seed = seed
        self.scale = scale
        self.cats = None
        self.booster = None

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        return apply_categories(df[FLOOR_NUM + CAT_COLS], self.cats)

    def _dmatrix(self, df: pd.DataFrame) -> xgb.DMatrix:
        d = xgb.DMatrix(self._X(df), enable_categorical=True)
        d.set_float_info("label_lower_bound", (df.lower / df.m).fillna(0.0).to_numpy(float))
        d.set_float_info("label_upper_bound", (df.upper / df.m).fillna(np.inf).to_numpy(float))
        return d

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> "FloorModel":
        train, valid = train[train.consistent], valid[valid.consistent]
        self.cats = fit_categories(train)
        params = {**XGB_PARAMS, "seed": self.seed}
        if self.scale is not None:
            params["aft_loss_distribution_scale"] = self.scale
        self.booster = xgb.train(
            params,
            self._dmatrix(train),
            num_boost_round=2000,
            evals=[(self._dmatrix(valid), "valid")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        return self

    def predict_ratio(self, df: pd.DataFrame) -> np.ndarray:
        d = xgb.DMatrix(self._X(df), enable_categorical=True)
        return self.booster.predict(d, iteration_range=(0, self.booster.best_iteration + 1))
