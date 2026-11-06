"""Three small stages: robust price model, tree correction, and lane adjustment."""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from .features import TREE_FEATURES

# Two plausible ways to extrapolate the remaining time pattern.
TREND_OPTIONS = {
    "quadratic": ["months_elapsed", "months_squared"],
    "seasonal": ["months_elapsed", "annual_sin", "annual_cos"],
}
BASE_COLUMNS = [
    "log_distance", "inverse_distance", "log_weight", "market_index",
    "quarter_end_ramp", "quote_signal"
]
TREE_SETTINGS = dict(
    objective="l2", n_estimators=400, learning_rate=0.05, num_leaves=15,
    min_child_samples=40, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1, verbose=-1, n_jobs=1
)


def linear_features(data, trend_columns):
    """Numeric design matrix for the log-rate regression."""
    equipment = np.column_stack([
        (data["equipment_code"] == code).to_numpy() for code in [1, 2]
    ])
    weekdays = np.column_stack([
        (data["weekday"] == day).to_numpy() for day in range(1, 7)
    ])
    return np.column_stack([
        np.ones(len(data)), data[BASE_COLUMNS + trend_columns].to_numpy(),
        equipment, weekdays
    ]).astype(float)


class FreightModel:
    def __init__(self, stage=3):
        # stage=1 regression only; stage=2 adds LightGBM; stage=3 adds lane effects.
        self.stage = stage
        self.fitted = {}
        self.outlier_count = {}

    def fit(self, train):
        log_rate = np.log(train["posted_rate"].to_numpy())
        for trend_name, trend_cols in TREND_OPTIONS.items():
            X = linear_features(train, trend_cols)

            # Fit regression repeatedly, excluding extreme training-only log errors.
            coefficients = np.linalg.lstsq(X, log_rate, rcond=None)[0]
            for _ in range(5):
                residual = log_rate - X @ coefficients
                ordinary = np.abs(residual) < 0.25
                coefficients = np.linalg.lstsq(X[ordinary], log_rate[ordinary], rcond=None)[0]

            residual = log_rate - X @ coefficients
            ordinary = np.abs(residual) < 0.25
            self.outlier_count[trend_name] = int((~ordinary).sum())
            clean = train.loc[ordinary]
            clean_residual = residual[ordinary]

            trees = []
            if self.stage >= 2:
                # Three small trees with different seeds stabilize corrections.
                for seed in [0, 1, 2]:
                    tree = LGBMRegressor(**TREE_SETTINGS, random_state=seed)
                    tree.fit(clean[TREE_FEATURES], clean_residual)
                    trees.append(tree)

            lanes = None
            if self.stage >= 3:
                lanes = self._lane_adjustments(clean, clean_residual)
            self.fitted[trend_name] = (trend_cols, coefficients, trees, lanes)
        return self

    def _lane_adjustments(self, train, residual):
        """Calculate small route corrections using tree out-of-fold predictions."""
        split = np.random.RandomState(0).randint(0, 5, len(train))
        oof = np.empty(len(train))
        for fold in range(5):
            tree = LGBMRegressor(**TREE_SETTINGS, random_state=0)
            tree.fit(train.loc[split != fold, TREE_FEATURES], residual[split != fold])
            oof[split == fold] = tree.predict(train.loc[split == fold, TREE_FEATURES])

        route = pd.DataFrame({
            "pickup": train["pickup"].to_numpy(),
            "delivery": train["delivery"].to_numpy(),
            "error": residual - oof,
        }).groupby(["pickup", "delivery"])["error"].agg(["sum", "count"])
        # Shrink noisy lanes toward zero; unseen lanes receive zero at prediction.
        return route["sum"] / (route["count"] + 3)

    def predict_log(self, data, stage=None):
        if stage is None:
            stage = self.stage
        predictions = []
        for trend_cols, coefficients, trees, lanes in self.fitted.values():
            value = linear_features(data, trend_cols) @ coefficients
            if trees and stage >= 2:
                value = value + np.mean([t.predict(data[TREE_FEATURES]) for t in trees], axis=0)
            if lanes is not None and stage >= 3:
                pairs = pd.MultiIndex.from_arrays([data["pickup"], data["delivery"]])
                value = value + lanes.reindex(pairs).fillna(0).to_numpy()
            predictions.append(value)
        return np.mean(predictions, axis=0)

    def predict(self, data, stage=None):
        return np.exp(self.predict_log(data, stage=stage))
