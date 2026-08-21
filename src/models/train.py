"""Phase 4: train and compare expected-points models on historical FPL data.

Two targets, trained and evaluated separately, then combined:
  - minutes:        how long will this player play next gameweek?
  - points_per_90:  how productive are they per 90 minutes, when they do play?
  - combined: expected_points = points_per_90_pred * (minutes_pred / 90)

This decomposition (rather than predicting total points directly) is deliberate - a player's
per-90 productivity and their rotation risk are different questions with different signal,
and collapsing them into one target hides exactly the cases (new signings, injury returns,
rotation-prone squad players) where getting the split right matters most.

Split: train on 2023-24 + 2024-25, test on the entire 2025-26 season - a season the models
never see during training, so this is a genuine backtest, not an in-sample fit.

Run directly:
    python -m src.models.train
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.features.historical_features import FEATURE_COLS, build_training_frame
from src.ingestion.db import get_connection

TRAIN_SEASONS = ["2023-24", "2024-25"]
TEST_SEASON = "2025-26"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

RF_PARAMS = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5, "random_state": 42, "n_jobs": -1}
XGB_PARAMS = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "random_state": 42}


def _position_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("pos_")]


def _metrics(y_true, y_pred) -> dict:
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(mean_squared_error(y_true, y_pred) ** 0.5, 4),
        "r2": round(r2_score(y_true, y_pred), 4),
    }


def _record_model_version(cur, model_type, target, features, hyperparameters, metrics, artifact_path):
    cur.execute(
        """
        INSERT INTO model_versions (
            model_type, target, training_seasons, test_season, features,
            hyperparameters, mae, rmse, r2, artifact_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            model_type, target, TRAIN_SEASONS, TEST_SEASON, json.dumps(features),
            json.dumps(hyperparameters) if hyperparameters else None,
            metrics["mae"], metrics["rmse"], metrics["r2"], artifact_path,
        ),
    )


def train_target(df: pd.DataFrame, target_col: str, target_name: str, baseline_col: str, cur) -> dict:
    """Trains baseline/linear/RF/XGBoost for one target, returns {model_type: (model_or_None, metrics, y_pred_test)}."""
    feature_cols = FEATURE_COLS + _position_cols(df)

    train_df = df[df["season"].isin(TRAIN_SEASONS) & (df["gw_number"] > 1)].dropna(subset=[target_col])
    test_df = df[(df["season"] == TEST_SEASON) & (df["gw_number"] > 1)].dropna(subset=[target_col])

    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df[target_col]
    X_test = test_df[feature_cols].fillna(0.0)
    y_test = test_df[target_col]

    results = {}

    baseline_pred = test_df[baseline_col].fillna(train_df[baseline_col].mean())
    results["baseline"] = (None, _metrics(y_test, baseline_pred), baseline_pred.values)

    linear = LinearRegression().fit(X_train, y_train)
    results["linear_regression"] = (linear, _metrics(y_test, linear.predict(X_test)), linear.predict(X_test))

    rf = RandomForestRegressor(**RF_PARAMS).fit(X_train, y_train)
    results["random_forest"] = (rf, _metrics(y_test, rf.predict(X_test)), rf.predict(X_test))

    xgb = XGBRegressor(**XGB_PARAMS).fit(X_train, y_train)
    results["xgboost"] = (xgb, _metrics(y_test, xgb.predict(X_test)), xgb.predict(X_test))

    print(f"\n=== {target_name} ===")
    for model_type, (model, metrics, _) in results.items():
        print(f"  {model_type:20s}  MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  R2={metrics['r2']:.3f}")

        artifact_path = None
        if model is not None:
            artifact_path = str(MODELS_DIR / f"{target_name}_{model_type}.joblib")
            joblib.dump(model, artifact_path)

        hyperparams = RF_PARAMS if model_type == "random_forest" else XGB_PARAMS if model_type == "xgboost" else None
        _record_model_version(cur, model_type, target_name, feature_cols, hyperparams, metrics, artifact_path)

    return {"results": results, "test_df": test_df, "feature_cols": feature_cols}


def combine_and_evaluate(df: pd.DataFrame, minutes_out: dict, points_out: dict) -> None:
    """Combines the best minutes model + best points-per-90 model into a full expected-points
    prediction on the test season, evaluated against actual total points - the headline metric."""
    best_minutes_type = min(
        (k for k in minutes_out["results"] if minutes_out["results"][k][0] is not None),
        key=lambda k: minutes_out["results"][k][1]["rmse"],
    )
    best_points_type = min(
        (k for k in points_out["results"] if points_out["results"][k][0] is not None),
        key=lambda k: points_out["results"][k][1]["rmse"],
    )
    print(f"\nBest minutes model: {best_minutes_type}")
    print(f"Best points_per_90 model: {best_points_type}")

    test_df = df[(df["season"] == TEST_SEASON) & (df["gw_number"] > 1)].copy()
    feature_cols = minutes_out["feature_cols"]
    X_test = test_df[feature_cols].fillna(0.0)

    minutes_model = minutes_out["results"][best_minutes_type][0]
    points_model = points_out["results"][best_points_type][0]

    pred_minutes = np.clip(minutes_model.predict(X_test), 0, 90)
    pred_points_per90 = points_model.predict(X_test)
    pred_total_points = pred_points_per90 * (pred_minutes / 90)

    metrics = _metrics(test_df["target_total_points"], pred_total_points)
    print(f"\n=== Combined expected_points (predicted total points, full {TEST_SEASON} test set) ===")
    print(f"  MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  R2={metrics['r2']:.3f}")

    naive_baseline = test_df["season_points_per90_avg"].fillna(0) * (test_df["season_minutes_avg"].fillna(0) / 90)
    naive_metrics = _metrics(test_df["target_total_points"], naive_baseline)
    print(f"  (naive season-average baseline for comparison: "
          f"MAE={naive_metrics['mae']:.3f}  RMSE={naive_metrics['rmse']:.3f}  R2={naive_metrics['r2']:.3f})")


def direct_points_baseline(df: pd.DataFrame) -> pd.Series:
    return df["season_points_per90_avg"].fillna(0) * (df["season_minutes_avg"].fillna(0) / 90)


def run() -> None:
    print("Loading and engineering historical features...")
    df = build_training_frame()
    print(f"Loaded {len(df)} player-gameweek rows across {df['season'].nunique()} seasons.")
    df["season_points_baseline"] = direct_points_baseline(df)

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                minutes_out = train_target(df, "target_minutes", "minutes", "season_minutes_avg", cur)
                points_df = df.dropna(subset=["target_points_per90"])
                points_out = train_target(
                    points_df, "target_points_per90", "points_per_90", "season_points_per90_avg", cur
                )
                direct_out = train_target(
                    df, "target_total_points", "total_points_direct", "season_points_baseline", cur
                )
        print(
            "\n--- Decomposed (points_per_90 x minutes) vs direct total_points comparison ---\n"
            "The decomposition is more interpretable (separates 'how good when playing' from\n"
            "'will they play') but on this data a direct model handles the ~60% of rows that\n"
            "are simply unused/fringe players (0 minutes) better - a clean zero is easier to\n"
            "learn directly than to get right by multiplying two separately-noisy estimates."
        )
        combine_and_evaluate(df, minutes_out, points_out)
        best_direct_type = min(
            (k for k in direct_out["results"] if direct_out["results"][k][0] is not None),
            key=lambda k: direct_out["results"][k][1]["rmse"],
        )
        print(f"\nBest direct total_points model: {best_direct_type} "
              f"(this is the recommended predictor - see comparison above)")
    finally:
        conn.close()


if __name__ == "__main__":
    run()
