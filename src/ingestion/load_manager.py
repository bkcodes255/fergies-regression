"""Pulls one manager's entry info, full season gameweek history, and current gameweek squad
picks, and upserts them.

Requires FPL_ENTRY_ID in .env. Run directly:
    python -m src.ingestion.load_manager

The season history pull (entry/{id}/history/) is what keeps past gameweeks' points/rank
correct — get_entry_picks only ever reflects the currently-active gameweek, so without this,
manager_gameweeks rows for already-finished gameweeks go stale (provisional bonus points,
auto-subs, and rank churn all land after a GW's initial pull) and never get corrected.
"""
from __future__ import annotations

from datetime import datetime, timezone

import psycopg2.extras

from config import settings
from src.ingestion.db import get_connection
from src.ingestion.fpl_client import FPLClient


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _backfill_gameweek_history(cur, client: FPLClient, entry_id: int, season: str, pulled_at: datetime) -> int:
    """Upserts every elapsed gameweek's finalized points/rank from entry/{id}/history/ — the
    only endpoint that reflects post-deadline corrections (bonus points, auto-subs) for
    gameweeks that are no longer the "current" one. Leaves active_chip untouched (this
    endpoint doesn't carry it; the picks payload does, in run() below)."""
    history = client.get_entry_history(entry_id)
    rows = [
        (
            entry_id, season, gw["event"], gw["points"], gw["total_points"], gw.get("overall_rank"),
            gw["bank"], gw["value"], gw["event_transfers"], gw["event_transfers_cost"],
            gw["points_on_bench"], gw.get("rank"), gw.get("percentile_rank"),
            float(gw["overall_rank_percentage"]) if gw.get("overall_rank_percentage") is not None else None,
            pulled_at,
        )
        for gw in history["current"]
    ]
    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO manager_gameweeks (
            entry_id, season, event_id, points, total_points, overall_rank,
            bank, team_value, event_transfers, event_transfers_cost, points_on_bench,
            event_rank, percentile_rank, rank_percentage, pulled_at
        ) VALUES %s
        ON CONFLICT (entry_id, season, event_id) DO UPDATE SET
            points = EXCLUDED.points, total_points = EXCLUDED.total_points,
            overall_rank = EXCLUDED.overall_rank, bank = EXCLUDED.bank,
            team_value = EXCLUDED.team_value, event_transfers = EXCLUDED.event_transfers,
            event_transfers_cost = EXCLUDED.event_transfers_cost,
            points_on_bench = EXCLUDED.points_on_bench,
            event_rank = EXCLUDED.event_rank, percentile_rank = EXCLUDED.percentile_rank,
            rank_percentage = EXCLUDED.rank_percentage,
            active_chip = COALESCE(manager_gameweeks.active_chip, EXCLUDED.active_chip),
            pulled_at = EXCLUDED.pulled_at
        """,
        rows,
    )
    return len(rows)


def run() -> None:
    if not settings.ENTRY_ID:
        raise RuntimeError("FPL_ENTRY_ID is not set in .env")

    season = settings.SEASON
    entry_id = settings.ENTRY_ID
    client = FPLClient()
    conn = get_connection()
    try:
        entry = client.get_entry(entry_id)
        current_event = entry["current_event"]
        picks_payload = client.get_entry_picks(entry_id, current_event)
        pulled_at = _now()

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO managers (entry_id, player_first_name, player_last_name, favourite_team)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (entry_id) DO UPDATE SET
                        player_first_name = EXCLUDED.player_first_name,
                        player_last_name = EXCLUDED.player_last_name,
                        favourite_team = EXCLUDED.favourite_team
                    """,
                    (entry_id, entry.get("player_first_name"), entry.get("player_last_name"),
                     entry.get("favourite_team")),
                )

                gw_count = _backfill_gameweek_history(cur, client, entry_id, season, pulled_at)
                print(f"Backfilled {gw_count} gameweek(s) of history for entry {entry_id}.")

                eh = picks_payload["entry_history"]
                cur.execute(
                    """
                    INSERT INTO manager_gameweeks (
                        entry_id, season, event_id, points, total_points, overall_rank,
                        bank, team_value, event_transfers, event_transfers_cost,
                        points_on_bench, active_chip, pulled_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (entry_id, season, event_id) DO UPDATE SET
                        points = EXCLUDED.points, total_points = EXCLUDED.total_points,
                        overall_rank = EXCLUDED.overall_rank, bank = EXCLUDED.bank,
                        team_value = EXCLUDED.team_value, event_transfers = EXCLUDED.event_transfers,
                        event_transfers_cost = EXCLUDED.event_transfers_cost,
                        points_on_bench = EXCLUDED.points_on_bench, active_chip = EXCLUDED.active_chip,
                        pulled_at = EXCLUDED.pulled_at
                    """,
                    (entry_id, season, current_event, eh["points"], eh["total_points"],
                     eh.get("overall_rank"), eh["bank"], eh["value"], eh["event_transfers"],
                     eh["event_transfers_cost"], eh["points_on_bench"],
                     picks_payload.get("active_chip"), pulled_at),
                )

                # element -> player_code, scoped to this season (season-local id, per the
                # gotcha documented in data/data_dictionary.md)
                cur.execute(
                    "SELECT fpl_id, player_code FROM player_seasons WHERE season = %s", (season,)
                )
                element_to_code = dict(cur.fetchall())

                rows = []
                for pick in picks_payload["picks"]:
                    player_code = element_to_code.get(pick["element"])
                    if player_code is None:
                        continue  # shouldn't happen if bootstrap-static was ingested for this season
                    rows.append((
                        entry_id, season, current_event, player_code, pick["position"],
                        pick["multiplier"], pick["is_captain"], pick["is_vice_captain"], pulled_at,
                    ))
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO squad_picks (
                        entry_id, season, event_id, player_code, squad_position,
                        multiplier, is_captain, is_vice_captain, pulled_at
                    ) VALUES %s
                    ON CONFLICT (entry_id, season, event_id, player_code) DO UPDATE SET
                        squad_position = EXCLUDED.squad_position, multiplier = EXCLUDED.multiplier,
                        is_captain = EXCLUDED.is_captain, is_vice_captain = EXCLUDED.is_vice_captain,
                        pulled_at = EXCLUDED.pulled_at
                    """,
                    rows,
                )

        print(f"Ingested squad for entry {entry_id}, GW{current_event}: {len(rows)} picks.")
    finally:
        conn.close()


if __name__ == "__main__":
    run()
