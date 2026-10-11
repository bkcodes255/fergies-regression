"""Writes the current gameweek's recommendation into transfer_plans, lineup_plans and
player_horizons (sql/plan_snapshots.sql) for the read-only Supabase Edge Function to serve.

Runs as the last step of ingest_and_predict.yml, right after the squad sync and the prediction
refresh, so the stored plans always match the newest inputs. Overwrites the rows for the
upcoming gameweek on every run; earlier gameweeks' rows are kept as the record of what was
planned before each deadline.

Run directly (needs DATABASE_URL, FPL_SEASON, FPL_ENTRY_ID):
    python -m src.recommendations.plan_snapshots
"""
from __future__ import annotations

import json
import sys

import pandas as pd
from psycopg2.extras import execute_values

from config import settings
from src.ingestion.db import get_connection
from src.recommendations.gameweek_plan import compute_recommendation, load_next_deadline
from src.recommendations.plan_payloads import lineup_plan_payload, transfer_plan_payload

TABLES = ("transfer_plans", "lineup_plans", "player_horizons")


def tables_exist(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT " + ", ".join(f"to_regclass('public.{t}')" for t in TABLES))
        return all(cur.fetchone())


def write_snapshots(conn, gw: dict, rec: dict) -> dict:
    """Upserts both transfer plans, both post-transfer lineups, and every player's horizon
    points for gw. Commits. Returns row counts."""
    season, event_id = gw["season"], gw["event_id"]
    rankings = rec["rankings"]
    prediction_gw = rankings["predicted_gw"].dropna().max() if "predicted_gw" in rankings else None
    prediction_gw = None if prediction_gw is None or pd.isna(prediction_gw) else int(prediction_gw)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MAX(event_id) FROM squad_picks WHERE entry_id = %s AND season = %s",
            (settings.ENTRY_ID, season),
        )
        squad_gw = cur.fetchone()[0]
        for horizon in ("1gw", "5gw"):
            for table, payload in (
                ("transfer_plans", transfer_plan_payload(rec, horizon)),
                ("lineup_plans", lineup_plan_payload(rec, horizon, event_id)),
            ):
                cur.execute(
                    f"""
                    INSERT INTO {table} (season, event_id, horizon, plan_json, prediction_gameweek, squad_sync_gameweek)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (season, event_id, horizon) DO UPDATE SET
                        plan_json = EXCLUDED.plan_json,
                        prediction_gameweek = EXCLUDED.prediction_gameweek,
                        squad_sync_gameweek = EXCLUDED.squad_sync_gameweek,
                        computed_at = now()
                    """,
                    (season, event_id, horizon, json.dumps(payload), prediction_gw, squad_gw),
                )
        cur.execute("DELETE FROM player_horizons WHERE season = %s AND event_id = %s", (season, event_id))
        rows = [
            (season, event_id, int(r["player_code"]), float(r["predicted_points"]),
             float(r["horizon_points"]), int(r["horizon_fixtures"]))
            for _, r in rankings.iterrows()
        ]
        execute_values(
            cur,
            "INSERT INTO player_horizons (season, event_id, player_code, predicted_points, horizon_points, horizon_fixtures) VALUES %s",
            rows,
        )
    conn.commit()
    return {"plans": 2, "lineups": 2, "player_horizons": len(rows)}


def run() -> int:
    conn = get_connection()
    try:
        if not tables_exist(conn):
            # A setup step, not a failure: don't turn the whole ingest run red before the
            # migration has been applied.
            print("Snapshot tables don't exist yet - run sql/plan_snapshots.sql first. Skipping.")
            return 0
        gw = load_next_deadline(conn)
        if gw is None:
            print("No upcoming deadline - nothing to snapshot.")
            return 0
        rec = compute_recommendation(conn, gw)
        if rec is None:
            print("No FPL_ENTRY_ID configured or no squad ingested - nothing to snapshot.")
            return 0
        counts = write_snapshots(conn, gw, rec)
        print(f"Snapshot for {gw['name']}: {counts}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(run())
