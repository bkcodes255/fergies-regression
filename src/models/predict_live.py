"""Applies the best trained model to our own live 2026/27 data to predict the next
gameweek's points for every current player, and stores the result in `predictions`.

Run directly:
    python -m src.models.predict_live
"""
from __future__ import annotations

import json

import joblib
import psycopg2.extras

from config import settings
from src.features.live_features import build_live_feature_frame
from src.ingestion.db import get_connection

TARGET = "total_points_direct"
HAUL_TARGETS = ["haul_6plus", "haul_10plus"]  # from src.models.train_haul_classifier


def get_best_model(cur):
    cur.execute(
        """
        SELECT model_id, model_type, features, artifact_path
        FROM model_versions
        WHERE target = %s AND artifact_path IS NOT NULL
        ORDER BY rmse ASC, trained_at DESC
        LIMIT 1
        """,
        (TARGET,),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"No trained model found for target={TARGET}. Run `python -m src.models.train` first.")
    return row


def get_best_classifier(cur, target: str):
    """Lowest Brier score wins, not highest ROC-AUC - calibration is what makes a displayed
    probability trustworthy as a probability, not just a good ranking signal. See
    train_haul_classifier.py's module docstring for why (class-balanced RF/logistic separate
    classes well but distort the actual probability; XGBoost trained on the natural imbalance
    stays close to true base rates)."""
    cur.execute(
        """
        SELECT model_id, model_type, features, artifact_path
        FROM model_versions
        WHERE target = %s AND artifact_path IS NOT NULL
        ORDER BY brier_score ASC, trained_at DESC
        LIMIT 1
        """,
        (target,),
    )
    return cur.fetchone()


def _load_and_predict(conn, season, row):
    model_id, model_type, features_json, artifact_path = row
    feature_cols = json.loads(features_json) if isinstance(features_json, str) else features_json
    model = joblib.load(artifact_path)
    meta, X = build_live_feature_frame(conn, season, feature_cols)
    return model_id, model_type, meta, X, model


def run() -> None:
    season = settings.SEASON
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            points_row = get_best_model(cur)
            haul_rows = {t: get_best_classifier(cur, t) for t in HAUL_TARGETS}

        points_model_id, points_model_type, meta, X, points_model = _load_and_predict(conn, season, points_row)
        print(f"Points model: model_id={points_model_id} ({points_model_type}), "
              f"{X.shape[1]} features, from {points_row[3]}")
        next_gw = int(meta["predicting_gw"].max())
        print(f"Predicting GW{next_gw} for {len(meta)} players...")
        predicted_points = points_model.predict(X)

        haul_probs = {}  # target -> {player_code: probability}
        for target, row in haul_rows.items():
            if row is None:
                print(f"No trained classifier found for {target} - run `python -m src.models.train_haul_classifier`."
                      f" Skipping (column will be NULL).")
                continue
            model_id, model_type, h_meta, h_X, h_model = _load_and_predict(conn, season, row)
            print(f"{target} model: model_id={model_id} ({model_type}), calibrated on Brier score")
            probs = h_model.predict_proba(h_X)[:, 1]
            haul_probs[target] = dict(zip(h_meta["player_code"], probs))

        rows = []
        for player_code, pred in zip(meta["player_code"], predicted_points):
            p6 = haul_probs.get("haul_6plus", {}).get(player_code)
            p10 = haul_probs.get("haul_10plus", {}).get(player_code)
            rows.append((
                season, next_gw, int(player_code), points_model_id, round(float(pred), 3),
                round(float(p6), 4) if p6 is not None else None,
                round(float(p10), 4) if p10 is not None else None,
            ))

        with conn:
            with conn.cursor() as cur:
                # predictions is "current best predictions", not a historical log - old
                # model_versions rows already carry full provenance/metrics, so superseded
                # prediction rows for this gameweek are just clutter, not lost history. Without
                # this, every retrain leaves stale rows behind under their old model_id, and
                # since the table's PK includes model_id, ON CONFLICT never touches them - they
                # silently fan out any join keyed on (season, event_id, player_code) alone, and
                # (Postgres sorts NULL first in DESC by default) can make a stale NULL-haul row
                # from before the classifiers existed outrank real predictions in a naive query.
                cur.execute(
                    "DELETE FROM predictions WHERE season = %s AND event_id = %s",
                    (season, next_gw),
                )
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO predictions (
                        season, event_id, player_code, model_id, predicted_points,
                        p_return_6plus, p_haul_10plus
                    )
                    VALUES %s
                    ON CONFLICT (season, event_id, player_code, model_id) DO UPDATE SET
                        predicted_points = EXCLUDED.predicted_points,
                        p_return_6plus = EXCLUDED.p_return_6plus,
                        p_haul_10plus = EXCLUDED.p_haul_10plus,
                        predicted_at = now()
                    """,
                    rows,
                )
        print(f"Done. Stored {len(rows)} predictions for GW{next_gw}.")
    finally:
        conn.close()


if __name__ == "__main__":
    run()
