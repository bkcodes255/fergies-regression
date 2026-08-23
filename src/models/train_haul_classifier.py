"""Phase 6.5: haul-probability classifiers - the direct fix for the "haul-blindness" finding
in notebooks/09_error_analysis.ipynb.

The deployed total_points_direct regressor (R2=0.320) is trained under squared-error loss on a
right-skewed target: for actual 11+ point hauls, its mean prediction is only ~3 - about a
quarter of what happened. That's not a bug or a missing feature, it's what MSE minimization
DOES on a skewed target - hedging low costs less in squared-error terms than confidently
guessing big and being wrong. Adding threat_roll/creativity_roll (see historical_features.py)
left the regressor's R2 completely unchanged (0.320 -> 0.320), confirming this: more/better
features don't fix an objective-function problem.

The fix here is to stop asking the regressor to solve a problem it structurally can't, and
instead train a classifier for a DIFFERENT question it can actually answer well: not "how many
points will they score" but "how likely is a big score." Two thresholds:
  - haul_6plus:  P(actual points >= 6)  - a good, captain-worthy return
  - haul_10plus: P(actual points >= 10) - a genuine haul, the specific blind spot found

This becomes the "ceiling probability" signal from the plan's own Risk Model section
(floor/median/ceiling, safe pick vs differential pick) - exactly what captaincy and
differential-transfer decisions need and the point estimate alone can't provide.

Same train/test split, same feature set, same reused build_training_frame() as train.py - only
the target and model type change.

Run directly:
    python -m src.models.train_haul_classifier
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from src.features.historical_features import FEATURE_COLS, build_training_frame
from src.ingestion.db import get_connection
from src.models.train import MODELS_DIR, TEST_SEASON, TRAIN_SEASONS

RF_PARAMS = {
    "n_estimators": 300, "max_depth": 6, "min_samples_leaf": 10,
    "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
}
XGB_PARAMS = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "random_state": 42}
THRESHOLDS = {"haul_6plus": 6, "haul_10plus": 10}


def _position_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("pos_")]


def _metrics(y_true, p_pred) -> dict:
    return {
        "roc_auc": round(roc_auc_score(y_true, p_pred), 4),
        "brier_score": round(brier_score_loss(y_true, p_pred), 4),
    }


def _record_model_version(cur, model_type, target, features, hyperparameters, metrics, artifact_path):
    cur.execute(
        """
        INSERT INTO model_versions (
            model_type, target, training_seasons, test_season, features,
            hyperparameters, roc_auc, brier_score, artifact_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            model_type, target, TRAIN_SEASONS, TEST_SEASON, json.dumps(features),
            json.dumps(hyperparameters) if hyperparameters else None,
            metrics["roc_auc"], metrics["brier_score"], artifact_path,
        ),
    )


def train_threshold(df: pd.DataFrame, threshold: int, target_name: str, cur) -> dict:
    feature_cols = FEATURE_COLS + _position_cols(df)
    labeled = df.copy()
    labeled["label"] = (labeled["target_total_points"] >= threshold).astype(int)

    train_df = labeled[labeled["season"].isin(TRAIN_SEASONS) & (labeled["gw_number"] > 1)]
    test_df = labeled[(labeled["season"] == TEST_SEASON) & (labeled["gw_number"] > 1)]

    X_train, y_train = train_df[feature_cols].fillna(0.0), train_df["label"]
    X_test, y_test = test_df[feature_cols].fillna(0.0), test_df["label"]

    base_rate_train = y_train.mean()
    print(f"\n=== {target_name} (base rate: train={base_rate_train:.3f}, test={y_test.mean():.3f}) ===")

    results = {}

    baseline_pred = pd.Series(base_rate_train, index=X_test.index)
    results["baseline"] = (None, _metrics(y_test, baseline_pred), baseline_pred)

    logit = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train, y_train)
    results["logistic_regression"] = (logit, _metrics(y_test, logit.predict_proba(X_test)[:, 1]), None)

    rf = RandomForestClassifier(**RF_PARAMS).fit(X_train, y_train)
    results["random_forest"] = (rf, _metrics(y_test, rf.predict_proba(X_test)[:, 1]), None)

    xgb = XGBClassifier(**XGB_PARAMS, eval_metric="logloss").fit(X_train, y_train)
    results["xgboost"] = (xgb, _metrics(y_test, xgb.predict_proba(X_test)[:, 1]), None)

    for model_type, (model, metrics, _) in results.items():
        print(f"  {model_type:20s}  ROC-AUC={metrics['roc_auc']:.3f}  Brier={metrics['brier_score']:.3f}")
        artifact_path = None
        if model is not None:
            artifact_path = str(MODELS_DIR / f"{target_name}_{model_type}.joblib")
            joblib.dump(model, artifact_path)
        hyperparams = RF_PARAMS if model_type == "random_forest" else XGB_PARAMS if model_type == "xgboost" else None
        _record_model_version(cur, model_type, target_name, feature_cols, hyperparams, metrics, artifact_path)

    return results


def run() -> None:
    print("Loading and engineering historical features...")
    df = build_training_frame()
    print(f"Loaded {len(df)} player-gameweek rows.")

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                for target_name, threshold in THRESHOLDS.items():
                    train_threshold(df, threshold, target_name, cur)
    finally:
        conn.close()

    print(
        "\nFor comparison, the deployed total_points_direct regressor's R2=0.320 doesn't "
        "translate to a classification metric directly - ROC-AUC substantially above 0.5 here "
        "means the classifier discriminates hauls from non-hauls well even though the "
        "regressor can't put a believable NUMBER on how big. That's the whole point: different "
        "questions, different objective, and this one the model can actually answer."
    )


if __name__ == "__main__":
    run()
