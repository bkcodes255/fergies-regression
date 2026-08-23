"""Model Lab: side-effect-free feature-subset evaluation for the dashboard's interactive
tinkering tab (dashboard/app.py). Lets you toggle which of the ~45 trained-model input columns
go into Linear Regression/Random Forest/XGBoost and see the real effect on held-out accuracy.

Deliberately NOT train.py's train_target(): that function joblib.dump()s to a fixed path
(models/{target}_{model_type}.joblib) and inserts a model_versions row on every call. Calling
it from an interactive "toggle a checkbox, click Run" loop would silently overwrite the real
deployed model artifact with an experimental feature-subset version while model_versions still
described it as trained on the full feature set - a real correctness bug for live serving
(predict_live.py loads whichever artifact model_versions currently ranks best). This module
fits models in memory only and returns everything the caller needs; the dashboard is
responsible for persisting a ledger row itself (model_versions, is_experiment=true,
artifact_path=NULL - see dashboard/app.py's Model Lab tab).

Same train/test split and hyperparameters as the deployed model (imported from train.py, not
duplicated) - only the feature list varies, so results are directly comparable to the real
deployed numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.models.train import RF_PARAMS, TEST_SEASON, TRAIN_SEASONS, XGB_PARAMS, _metrics

MODEL_TYPES = ("baseline", "linear_regression", "random_forest", "xgboost")
N_BOOTSTRAP = 1000
OVERFIT_GAP_THRESHOLD = 0.15  # train_r2 - test_r2 above this reads as "possible overfit" - a
# heuristic, not a hard rule, same spirit as every other judgment call this project's docs
# already caveat (see README's Phase 6.6 "modest, not a breakthrough" framing).


def _bootstrap_r2_ci(y_true: np.ndarray, y_pred: np.ndarray, n: int = N_BOOTSTRAP,
                      rng: np.random.Generator | None = None) -> tuple[float, float]:
    """95% CI on test R² via resampling the (y_true, y_pred) pairs with replacement - answers
    'is this R² meaningfully different from another one, or noise on one held-out season.' No
    refit needed: just resampling already-computed predictions."""
    rng = rng or np.random.default_rng(42)
    idx = np.arange(len(y_true))
    scores = np.empty(n)
    for i in range(n):
        sample = rng.choice(idx, size=len(idx), replace=True)
        yt, yp = y_true[sample], y_pred[sample]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        scores[i] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    scores = scores[~np.isnan(scores)]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def paired_bootstrap_p_value(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray,
                              n: int = N_BOOTSTRAP) -> float:
    """Paired bootstrap significance test: is model B's R² really different from model A's on
    this same test set, or within noise? Resamples the same indices for both predictions each
    draw (paired, not independent), and the p-value is the two-sided fraction of draws where
    the sign of (R²_b - R²_a) disagrees with the sign observed on the real (unresampled) data -
    the standard nonparametric approach when there's no clean parametric alternative for a
    prediction-based metric like R²."""
    rng = np.random.default_rng(42)
    idx = np.arange(len(y_true))
    observed_diff = None
    diffs = np.empty(n)
    for i in range(n):
        sample = rng.choice(idx, size=len(idx), replace=True)
        yt = y_true[sample]
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        if ss_tot == 0:
            diffs[i] = np.nan
            continue
        r2_a = 1 - np.sum((yt - y_pred_a[sample]) ** 2) / ss_tot
        r2_b = 1 - np.sum((yt - y_pred_b[sample]) ** 2) / ss_tot
        diffs[i] = r2_b - r2_a
    diffs = diffs[~np.isnan(diffs)]
    ss_tot_obs = np.sum((y_true - y_true.mean()) ** 2)
    observed_diff = (
        (1 - np.sum((y_true - y_pred_b) ** 2) / ss_tot_obs)
        - (1 - np.sum((y_true - y_pred_a) ** 2) / ss_tot_obs)
    )
    if observed_diff >= 0:
        return float(np.mean(diffs <= 0)) * 2
    return float(np.mean(diffs >= 0)) * 2


def _ols_diagnostics(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Per-coefficient p-values/CIs + overall F-test p-value via statsmodels - sklearn's
    LinearRegression doesn't expose significance at all. The one model type here where a
    classical p-value is actually a well-defined concept."""
    X_with_const = sm.add_constant(X_train.astype(float), has_constant="add")
    ols = sm.OLS(y_train.to_numpy(dtype=float), X_with_const.to_numpy(dtype=float)).fit()
    coef_names = ["const"] + list(X_train.columns)
    return {
        "f_pvalue": float(ols.f_pvalue),
        "r_squared": float(ols.rsquared),
        "coefficients": {
            name: {"coef": float(c), "p_value": float(p), "ci_low": float(lo), "ci_high": float(hi)}
            for name, c, p, (lo, hi) in zip(
                coef_names, ols.params, ols.pvalues, ols.conf_int()
            )
        },
    }


def _permutation_diagnostics(model, X_test: pd.DataFrame, y_test: pd.Series, n_repeats: int = 10) -> dict:
    """Tree ensembles have no p-values in the classical sense - permutation importance (shuffle
    one feature on the test set, measure the score drop, repeat) is the honest substitute: a
    mean +/- std per feature, not a significance claim."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=42, scoring="r2", n_jobs=-1
    )
    return {
        col: {"importance_mean": float(m), "importance_std": float(s)}
        for col, m, s in zip(X_test.columns, result.importances_mean, result.importances_std)
    }


def evaluate_feature_subset(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_total_points",
    baseline_col: str = "season_points_baseline",
    sample_frac: float | None = None,
    compute_extended: bool = False,
) -> dict[str, dict]:
    """Fits baseline/linear_regression/random_forest/xgboost on TRAIN_SEASONS, scores on
    TEST_SEASON - the exact split train.py uses, so results are directly comparable to the
    deployed model's numbers. Purely in-memory: no joblib.dump, no DB writes (the caller decides
    whether/how to persist a ledger row).

    Returns {
        "_y_test": np.ndarray - the true held-out target, same order/length as every
            model_type's "y_test_pred" below (both TRAIN_SEASONS/TEST_SEASON split, no
            shuffling) - callers use this plus two "y_test_pred" arrays for
            paired_bootstrap_p_value() without re-deriving the split themselves.
        model_type: {
            "test_metrics": {...}, "train_metrics": {...} (None for baseline - nothing is "fit"),
            "overfit_gap": train_r2 - test_r2 (None for baseline),
            "r2_ci": (low, high) - bootstrap 95% CI on test R²,
            "model": fitted estimator or None,
            "y_test_pred": np.ndarray,
            "extended": {...} or None - OLS coefficients (linear_regression) / permutation
                importance (random_forest, xgboost), only when compute_extended=True.
        } for each of MODEL_TYPES
    }
    """
    train_df = df[df["season"].isin(TRAIN_SEASONS) & (df["gw_number"] > 1)].dropna(subset=[target_col])
    test_df = df[(df["season"] == TEST_SEASON) & (df["gw_number"] > 1)].dropna(subset=[target_col])
    if sample_frac is not None:
        train_df = train_df.sample(frac=sample_frac, random_state=42)

    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df[target_col]
    X_test = test_df[feature_cols].fillna(0.0)
    y_test = test_df[target_col].to_numpy()

    results: dict[str, dict] = {"_y_test": y_test}

    baseline_pred = test_df[baseline_col].fillna(train_df[baseline_col].mean()).to_numpy()
    results["baseline"] = {
        "test_metrics": _metrics(y_test, baseline_pred),
        "train_metrics": None,
        "overfit_gap": None,
        "r2_ci": _bootstrap_r2_ci(y_test, baseline_pred),
        "model": None,
        "y_test_pred": baseline_pred,
        "extended": None,
    }

    def _fit_and_score(model, model_type: str) -> None:
        model.fit(X_train, y_train)
        pred_test = model.predict(X_test)
        pred_train = model.predict(X_train)
        test_metrics = _metrics(y_test, pred_test)
        train_metrics = _metrics(y_train, pred_train)
        extended = None
        if compute_extended:
            if model_type == "linear_regression":
                extended = _ols_diagnostics(X_train, y_train)
            else:
                extended = _permutation_diagnostics(model, X_test, pd.Series(y_test), n_repeats=10)
        results[model_type] = {
            "test_metrics": test_metrics,
            "train_metrics": train_metrics,
            "overfit_gap": round(train_metrics["r2"] - test_metrics["r2"], 4),
            "r2_ci": _bootstrap_r2_ci(y_test, pred_test),
            "model": model,
            "y_test_pred": pred_test,
            "extended": extended,
        }

    _fit_and_score(LinearRegression(), "linear_regression")
    _fit_and_score(RandomForestRegressor(**RF_PARAMS), "random_forest")
    _fit_and_score(XGBRegressor(**XGB_PARAMS), "xgboost")

    return results
