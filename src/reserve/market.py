"""M1: market value m-hat = shrunk trailing log price + optional LightGBM residual."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from reserve.frame import CAT_COLS, apply_categories, fit_categories

FEATURES_NUM = ["log_m_naive", "px_mean_L1", "px_mean_L3", "px_mean_L5", "px_n_L3", "px_n_L5",
                "units_sold_w", "nb", "ub", "tau", "age_days", "dow"]
LGB_PARAMS = dict(objective="l1", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)


def realized_targets(sales: pd.DataFrame) -> pd.DataFrame:
    g = (sales.assign(pu=sales.unit_price * sales.units)
         .groupby(["sku", "wh", "day"], as_index=False)
         .agg(pu=("pu", "sum"), units=("units", "sum")))
    g["y"] = np.log(g.pu / g.units)
    return g[["sku", "wh", "day", "y"]]


def mape(y_log, yhat_log) -> float:
    y, yh = np.exp(np.asarray(y_log, float)), np.exp(np.asarray(yhat_log, float))
    return float(np.mean(np.abs(yh - y) / y))


class MarketModel:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.booster = None
        self.cats = None
        self.use_lgb = False

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        return apply_categories(df[FEATURES_NUM + CAT_COLS], self.cats)

    def _predict_lgb(self, df: pd.DataFrame) -> np.ndarray:
        resid = self.booster.predict(self._X(df), num_iteration=self.booster.best_iteration)
        return df.log_m_naive.to_numpy(float) + resid

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> "MarketModel":
        self.cats = fit_categories(train)
        dtr = lgb.Dataset(self._X(train), train.y - train.log_m_naive)
        dva = lgb.Dataset(self._X(valid), valid.y - valid.log_m_naive, reference=dtr)
        self.booster = lgb.train({**LGB_PARAMS, "seed": self.seed}, dtr, num_boost_round=2000,
                                 valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        self.use_lgb = mape(valid.y, self._predict_lgb(valid)) < mape(valid.y, valid.log_m_naive)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._predict_lgb(df) if self.use_lgb else df.log_m_naive.to_numpy(float)
