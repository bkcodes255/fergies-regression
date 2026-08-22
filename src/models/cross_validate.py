"""Phase 6: walk-forward cross-validation of the prediction model across seasons.

src/models/train.py evaluates model selection with a SINGLE train/holdout split (train on
2022-23+2023-24+2024-25, test on 2025-26) - one R2/RMSE reading per model type per target. This
module checks whether that reading is stable rather than a lucky/unlucky split, by repeating the
same expanding-window backtest at every season boundary we have data for, using the identical
feature set and hyperparameters train.py trains the deployed model with.

Purely an evaluation exercise - does NOT write to model_versions or save joblib artifacts (that
stays train.py's job: deploying the model trained on ALL prior seasons, tested against the true
holdout season). A fold trained on much less data could land a lucky low RMSE on an easier test
season; we don't want that outranking the real deployed model in predict_live.get_best_model's
selection, which just picks the lowest recorded RMSE.

Folds (expanding window, always respecting season order - a fold never tests on a season that's
also in its own training data):
  fold 1: train=[2022-23]                    test=2023-24
  fold 2: train=[2022-23, 2023-24]            test=2024-25
  fold 3: train=[2022-23, 2023-24, 2024-25]   test=2025-26   (== train.py's own split)

Run directly:
    python -m src.models.cross_validate
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.features.historical_features import FEATURE_COLS, build_training_frame
from src.models.train import RF_PARAMS, XGB_PARAMS, direct_points_baseline

SEASON_ORDER = ["2022-23", "2023-24", "2024-25", "2025-26"]
FOLDS = [(SEASON_ORDER[:i], SEASON_ORDER[i]) for i in range(1, len(SEASON_ORDER))]

TARGETS = [
    ("target_minutes", "minutes", "season_minutes_avg"),
    ("target_points_per90", "points_per_90", "season_points_per90_avg"),
    ("target_total_points", "total_points_direct", "season_points_baseline"),
]


def _position_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("pos_")]


def _metrics(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }


def _fit_and_eval(train_df, test_df, feature_cols, target_col, baseline_col) -> dict[str, dict]:
    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df[target_col]
    X_test = test_df[feature_cols].fillna(0.0)
    y_test = test_df[target_col]

    results = {}
    baseline_pred = test_df[baseline_col].fillna(train_df[baseline_col].mean())
    results["baseline"] = _metrics(y_test, baseline_pred)
    results["linear_regression"] = _metrics(y_test, LinearRegression().fit(X_train, y_train).predict(X_test))
    results["random_forest"] = _metrics(
        y_test, RandomForestRegressor(**RF_PARAMS).fit(X_train, y_train).predict(X_test)
    )
    results["xgboost"] = _metrics(y_test, XGBRegressor(**XGB_PARAMS).fit(X_train, y_train).predict(X_test))
    return results


def run_cross_validation() -> pd.DataFrame:
    print("Loading and engineering historical features...")
    df = build_training_frame()
    df["season_points_baseline"] = direct_points_baseline(df)
    feature_cols = FEATURE_COLS + _position_cols(df)

    rows = []
    for train_seasons, test_season in FOLDS:
        print(f"\nFold: train={train_seasons} test={test_season}")
        for target_col, target_name, baseline_col in TARGETS:
            fold_df = df.dropna(subset=["target_points_per90"]) if target_name == "points_per_90" else df

            train_df = fold_df[fold_df["season"].isin(train_seasons) & (fold_df["gw_number"] > 1)].dropna(
                subset=[target_col]
            )
            test_df = fold_df[(fold_df["season"] == test_season) & (fold_df["gw_number"] > 1)].dropna(
                subset=[target_col]
            )
            if train_df.empty or test_df.empty:
                continue

            results = _fit_and_eval(train_df, test_df, feature_cols, target_col, baseline_col)
            for model_type, metrics in results.items():
                rows.append({
                    "train_seasons": ",".join(train_seasons), "test_season": test_season,
                    "target": target_name, "model_type": model_type,
                    "n_train": len(train_df), "n_test": len(test_df), **metrics,
                })
            print(f"  {target_name:18s} " + "  ".join(f"{mt}: R2={m['r2']:.3f}" for mt, m in results.items()))

    return pd.DataFrame(rows)


def summarize(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std R2/RMSE per (target, model_type) across folds - the headline "is this
    stable, or a lucky split" answer."""
    summary = cv_results.groupby(["target", "model_type"], as_index=False).agg(
        folds=("test_season", "count"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
    )
    return summary.sort_values(["target", "rmse_mean"]).reset_index(drop=True)


def run() -> None:
    cv_results = run_cross_validation()
    summary = summarize(cv_results)
    print("\n=== Cross-validated summary (mean +/- std across folds) ===")
    for target in summary["target"].unique():
        print(f"\n--- {target} ---")
        for _, row in summary[summary["target"] == target].iterrows():
            std_r2 = 0.0 if pd.isna(row["r2_std"]) else row["r2_std"]
            std_rmse = 0.0 if pd.isna(row["rmse_std"]) else row["rmse_std"]
            print(
                f"  {row['model_type']:20s} R2={row['r2_mean']:.3f}+/-{std_r2:.3f}  "
                f"RMSE={row['rmse_mean']:.3f}+/-{std_rmse:.3f}  (folds={int(row['folds'])})"
            )


if __name__ == "__main__":
    run()
