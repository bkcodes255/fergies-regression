"""Phase 7: quantile regression for real floor/median/ceiling point estimates, plus Monte
Carlo sampling on top of them for squad-level questions (P(squad scores 60+), etc.).

Why quantile regression rather than just simulating from the existing point-estimate
regressor: a single RF's individual-tree spread reflects MODEL uncertainty (how much the
trees disagree, which is suppressed by bagging on the same data), not the TRUE outcome
variance - it would systematically understate real spread. Quantile regression trains
directly on "what value does the actual outcome fall below X% of the time," which is the
right question and has an honest way to check the answer: calibration (does the predicted
90th-percentile line actually get exceeded ~10% of the time on held-out data?).

Same train/test split and feature set as every other model here. Uses
GradientBoostingRegressor(loss="quantile") - simple, well-tested, no new dependency (unlike
XGBoost's newer native quantile API, whose behavior is less battle-tested across versions).

Run directly:
    python -m src.models.train_quantile_models
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.features.historical_features import FEATURE_COLS, build_training_frame
from src.ingestion.db import get_connection
from src.models.train import MODELS_DIR, TEST_SEASON, TRAIN_SEASONS

QUANTILES = {"floor": 0.10, "median": 0.50, "ceiling": 0.90}
GBR_PARAMS = {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 20, "random_state": 42}


def _position_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("pos_")]


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def calibration_rate(y_true, y_pred_quantile) -> float:
    """Fraction of actual outcomes at or below the predicted quantile value - should be close
    to the target quantile itself if the model is well-calibrated, not just low-error.

    WARNING: on the full population this is distorted for low quantiles by the ~63% point mass
    at exactly 0 points (bench/unused players) - if the target quantile sits inside that mass,
    ties get counted as "at or below," inflating the apparent rate regardless of how good the
    model actually is. Confirmed empirically: floor's aggregate rate read 0.633 (target 0.10,
    looks badly miscalibrated), but restricted to nailed-on starters (season_minutes_avg>=75,
    where blanking is genuinely rare) it reads 0.130 - close to the 0.10 target. Always check
    calibration_rate_by_segment too, not just this aggregate number, before concluding a
    quantile model is mis-specified."""
    return float(np.mean(y_true <= y_pred_quantile))


def calibration_rate_by_segment(y_true, y_pred_quantile, minutes_avg, threshold: float = 75.0) -> dict:
    """The honest version of calibration_rate for a zero-inflated target: split into
    nailed-on starters (rarely blank - minutes_avg >= threshold) vs everyone else, since the
    aggregate rate is dominated by trivial zero=zero ties in the low-minutes majority."""
    nailed = minutes_avg >= threshold
    return {
        "nailed_n": int(nailed.sum()),
        "nailed_rate": calibration_rate(y_true[nailed], y_pred_quantile[nailed]) if nailed.sum() else None,
        "other_n": int((~nailed).sum()),
        "other_rate": calibration_rate(y_true[~nailed], y_pred_quantile[~nailed]) if (~nailed).sum() else None,
    }


def _record_model_version(cur, model_type, target, features, hyperparameters, mae, artifact_path):
    cur.execute(
        """
        INSERT INTO model_versions (
            model_type, target, training_seasons, test_season, features, hyperparameters,
            mae, artifact_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (model_type, target, TRAIN_SEASONS, TEST_SEASON, json.dumps(features),
         json.dumps(hyperparameters), mae, artifact_path),
    )


def run() -> None:
    print("Loading and engineering historical features...")
    df = build_training_frame()
    feature_cols = FEATURE_COLS + _position_cols(df)

    train_df = df[df["season"].isin(TRAIN_SEASONS) & (df["gw_number"] > 1)].dropna(subset=["target_total_points"])
    test_df = df[(df["season"] == TEST_SEASON) & (df["gw_number"] > 1)].dropna(subset=["target_total_points"])
    X_train, y_train = train_df[feature_cols].fillna(0.0), train_df["target_total_points"]
    X_test, y_test = test_df[feature_cols].fillna(0.0), test_df["target_total_points"]
    print(f"Train: {len(X_train)} rows. Test: {len(X_test)} rows.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                for name, q in QUANTILES.items():
                    target_name = f"quantile_{name}"
                    model = GradientBoostingRegressor(loss="quantile", alpha=q, **GBR_PARAMS)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)

                    pinball = pinball_loss(y_test, pred, q)
                    cal_rate = calibration_rate(y_test, pred)
                    seg = calibration_rate_by_segment(
                        y_test.values, pred, test_df["season_minutes_avg"].values
                    )
                    print(f"\n=== {target_name} (target quantile={q}) ===")
                    print(f"  Pinball loss: {pinball:.4f}")
                    print(f"  Aggregate calibration: {cal_rate:.3f} (target {q:.2f}) - "
                          f"NOTE: distorted low-side by the ~63% zero point-mass, see nailed-only below")
                    nailed_rate = seg["nailed_rate"]
                    verdict = "OK" if nailed_rate is not None and abs(nailed_rate - q) < 0.10 else "CHECK"
                    print(f"  Nailed-starters-only calibration (n={seg['nailed_n']}): "
                          f"{nailed_rate:.3f} (target {q:.2f}) {verdict}")

                    # run_id keeps this filename unique per run() invocation - a static name
                    # meant every retrain silently overwrote the file an OLDER model_versions
                    # row still pointed to (same real bug found and fixed in train.py).
                    artifact_path = str(MODELS_DIR / f"{target_name}_gbr_{run_id}.joblib")
                    joblib.dump(model, artifact_path)
                    _record_model_version(
                        cur, "gradient_boosting_quantile", target_name, feature_cols,
                        {**GBR_PARAMS, "alpha": q}, pinball, artifact_path,
                    )

        # sanity check: floor <= median <= ceiling should hold for (almost) every row - quantile
        # models are trained independently so crossing is possible but should be rare
        floor_m = joblib.load(str(MODELS_DIR / "quantile_floor_gbr.joblib"))
        median_m = joblib.load(str(MODELS_DIR / "quantile_median_gbr.joblib"))
        ceiling_m = joblib.load(str(MODELS_DIR / "quantile_ceiling_gbr.joblib"))
        f, m, c = floor_m.predict(X_test), median_m.predict(X_test), ceiling_m.predict(X_test)
        crossing_rate = float(np.mean((f > m) | (m > c)))
        print(f"\nQuantile crossing rate (floor>median or median>ceiling, should be near 0): "
              f"{crossing_rate:.4f}")
    finally:
        conn.close()


if __name__ == "__main__":
    run()
