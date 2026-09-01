"""Diagnostic pass on the deployed total_points_direct model: which features actually drive its
predictions, and where does it miss - not just the headline R2/MAE (already in notebooks 06 and
08), but *what kind* of rows the error concentrates in.

Reuses the exact same held-out 2025-26 test set and model artifact predict_live.py serves live
predictions from - this describes the model that's actually deployed, not a fresh fit.

Run directly:
    python -m src.models.error_analysis
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from config.settings import MODELS_DIR
from src.features.historical_features import build_training_frame
from src.ingestion.db import get_connection
from src.models.predict_live import get_best_model

BUCKET_EDGES = [-1, 0, 2, 5, 10, 100]
BUCKET_LABELS = ["blank (0)", "1-2", "3-5", "6-10", "11+ (haul)"]


def load_deployed_model():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            model_id, model_type, features_json, artifact_path = get_best_model(cur)
    finally:
        conn.close()
    feature_cols = json.loads(features_json) if isinstance(features_json, str) else features_json
    model = joblib.load(MODELS_DIR / artifact_path)
    return model, feature_cols, model_id, model_type, artifact_path


def score_test_set(model, feature_cols: list[str]) -> pd.DataFrame:
    """Returns the 2025-26 held-out test rows (gw_number>1) with `pred` and `resid` columns
    added - the same rows notebook 06/08 evaluate R2/MAE on, just kept row-level here."""
    df = build_training_frame()
    test_df = df[(df["season"] == "2025-26") & (df["gw_number"] > 1)].dropna(
        subset=["target_total_points"]
    ).copy()
    X_test = test_df[feature_cols].fillna(0.0)
    test_df["pred"] = model.predict(X_test)
    test_df["resid"] = test_df["target_total_points"] - test_df["pred"]
    return test_df


def feature_importance(model, feature_cols: list[str]) -> pd.Series:
    return pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)


def residuals_by_position(test_df: pd.DataFrame) -> pd.DataFrame:
    return test_df.groupby("position").apply(
        lambda g: pd.Series({
            "n": len(g),
            "mean_actual": g["target_total_points"].mean(),
            "mean_pred": g["pred"].mean(),
            "bias": (g["pred"] - g["target_total_points"]).mean(),
            "mae": g["resid"].abs().mean(),
        }),
        include_groups=False,
    )


def residuals_by_bucket(test_df: pd.DataFrame) -> pd.DataFrame:
    """The key diagnostic: does the model's average prediction track the average actual result
    within bands of actual outcome, or does it compress toward the middle regardless of what
    actually happened (classic MSE-trained-regressor behavior on a skewed target)?"""
    bucketed = test_df.copy()
    bucketed["bucket"] = pd.cut(bucketed["target_total_points"], bins=BUCKET_EDGES, labels=BUCKET_LABELS)
    return bucketed.groupby("bucket", observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "mean_actual": g["target_total_points"].mean(),
            "mean_pred": g["pred"].mean(),
            "mae": g["resid"].abs().mean(),
        }),
        include_groups=False,
    )


def worst_misses(test_df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    ranked = test_df.reindex(test_df["resid"].abs().sort_values(ascending=False).index)
    return ranked.head(top_n)[["name", "position", "GW", "target_total_points", "pred", "resid"]]


def run() -> None:
    model, feature_cols, model_id, model_type, artifact_path = load_deployed_model()
    print(f"Analyzing model_id={model_id} ({model_type}) from {artifact_path}")

    test_df = score_test_set(model, feature_cols)
    r2 = r2_score(test_df["target_total_points"], test_df["pred"])
    mae = mean_absolute_error(test_df["target_total_points"], test_df["pred"])
    print(f"Held-out 2025-26: R2={r2:.3f}  MAE={mae:.3f}  n={len(test_df)}")

    print("\n=== Top 10 feature importances ===")
    print(feature_importance(model, feature_cols).head(10))

    print("\n=== Residuals by position ===")
    print(residuals_by_position(test_df).round(3))

    print("\n=== Residuals by actual-points bucket ===")
    print(residuals_by_bucket(test_df).round(3))

    print("\n=== Worst 15 misses ===")
    print(worst_misses(test_df).to_string(index=False))


if __name__ == "__main__":
    run()
