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


def run() -> None:
    season = settings.SEASON
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            model_id, model_type, features_json, artifact_path = get_best_model(cur)
        feature_cols = json.loads(features_json) if isinstance(features_json, str) else features_json
        print(f"Using model_id={model_id} ({model_type}), {len(feature_cols)} features, from {artifact_path}")

        model = joblib.load(artifact_path)
        meta, X = build_live_feature_frame(conn, season, feature_cols)
        next_gw = int(meta["predicting_gw"].max())
        print(f"Predicting GW{next_gw} for {len(meta)} players...")

        predictions = model.predict(X)

        rows = [
            (season, next_gw, int(player_code), model_id, round(float(pred), 3))
            for player_code, pred in zip(meta["player_code"], predictions)
        ]

        with conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO predictions (season, event_id, player_code, model_id, predicted_points)
                    VALUES %s
                    ON CONFLICT (season, event_id, player_code, model_id) DO UPDATE SET
                        predicted_points = EXCLUDED.predicted_points,
                        predicted_at = now()
                    """,
                    rows,
                )
        print(f"Done. Stored {len(rows)} predictions for GW{next_gw}.")
    finally:
        conn.close()


if __name__ == "__main__":
    run()
