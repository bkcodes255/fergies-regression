"""Phase 1 ingestion: pull bootstrap-static + fixtures + every finished/current gameweek's
live stats, and upsert into the normalized schema (sql/schema.sql) plus raw_snapshots.

Run directly for a full manual pull:

    python -m src.ingestion.load

Safe to re-run any time — every table is upserted (INSERT ... ON CONFLICT DO UPDATE) on its
primary key, so re-pulling today's data just refreshes it rather than duplicating rows.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from typing import Any

import psycopg2.extras

from config import settings
from src.ingestion.db import get_connection
from src.ingestion.fpl_client import FPLClient

LIVE_PULL_DELAY_SECONDS = 1  # be polite to a free public API across a multi-event backfill


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_float(value: Any) -> float | None:
    """FPL returns several numeric fields (influence, xG, selected_by_percent, ...) as
    strings (e.g. "0.00"). Cast explicitly rather than relying on implicit casting through
    psycopg2's parameter binding."""
    return None if value is None else float(value)


def _upsert(cur, sql: str, rows: list[tuple]) -> None:
    if rows:
        psycopg2.extras.execute_values(cur, sql, rows)


def insert_raw_snapshot(cur, endpoint: str, season: str, event_id: int | None, payload: Any) -> None:
    cur.execute(
        """
        INSERT INTO raw_snapshots (endpoint, season, event_id, pulled_at, payload)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (endpoint, season, event_id, _now(), json.dumps(payload)),
    )


def upsert_teams(cur, bootstrap: dict) -> dict[int, int]:
    """Returns team_id -> team_code lookup for use by the rest of this run."""
    rows = [(t["code"], t["name"], t["short_name"]) for t in bootstrap["teams"]]
    _upsert(
        cur,
        """
        INSERT INTO teams (team_code, name, short_name) VALUES %s
        ON CONFLICT (team_code) DO UPDATE SET name = EXCLUDED.name, short_name = EXCLUDED.short_name
        """,
        rows,
    )
    return {t["id"]: t["code"] for t in bootstrap["teams"]}


def upsert_team_season_stats(cur, bootstrap: dict, season: str, pulled_at: datetime) -> None:
    rows = [
        (
            t["code"], season, t["id"], t.get("strength"),
            t.get("strength_overall_home"), t.get("strength_overall_away"),
            t.get("strength_attack_home"), t.get("strength_attack_away"),
            t.get("strength_defence_home"), t.get("strength_defence_away"),
            pulled_at,
        )
        for t in bootstrap["teams"]
    ]
    _upsert(
        cur,
        """
        INSERT INTO team_season_stats (
            team_code, season, fpl_id, strength,
            strength_overall_home, strength_overall_away,
            strength_attack_home, strength_attack_away,
            strength_defence_home, strength_defence_away, pulled_at
        ) VALUES %s
        ON CONFLICT (team_code, season) DO UPDATE SET
            fpl_id = EXCLUDED.fpl_id, strength = EXCLUDED.strength,
            strength_overall_home = EXCLUDED.strength_overall_home,
            strength_overall_away = EXCLUDED.strength_overall_away,
            strength_attack_home = EXCLUDED.strength_attack_home,
            strength_attack_away = EXCLUDED.strength_attack_away,
            strength_defence_home = EXCLUDED.strength_defence_home,
            strength_defence_away = EXCLUDED.strength_defence_away,
            pulled_at = EXCLUDED.pulled_at
        """,
        rows,
    )


def upsert_players(cur, bootstrap: dict) -> dict[int, int]:
    """Returns element_id -> player_code lookup for use by the rest of this run."""
    rows = [
        (el["code"], el["first_name"], el["second_name"], el["web_name"], el.get("birth_date"))
        for el in bootstrap["elements"]
    ]
    _upsert(
        cur,
        """
        INSERT INTO players (player_code, first_name, second_name, web_name, birth_date) VALUES %s
        ON CONFLICT (player_code) DO UPDATE SET
            first_name = EXCLUDED.first_name, second_name = EXCLUDED.second_name,
            web_name = EXCLUDED.web_name, birth_date = EXCLUDED.birth_date
        """,
        rows,
    )
    return {el["id"]: el["code"] for el in bootstrap["elements"]}


def upsert_player_seasons(
    cur, bootstrap: dict, season: str, team_id_to_code: dict[int, int], pulled_at: datetime
) -> None:
    rows = [
        (el["code"], season, el["id"], team_id_to_code[el["team"]], el["element_type"], pulled_at)
        for el in bootstrap["elements"]
    ]
    _upsert(
        cur,
        """
        INSERT INTO player_seasons (player_code, season, fpl_id, team_code, element_type, pulled_at)
        VALUES %s
        ON CONFLICT (player_code, season) DO UPDATE SET
            fpl_id = EXCLUDED.fpl_id, team_code = EXCLUDED.team_code,
            element_type = EXCLUDED.element_type, pulled_at = EXCLUDED.pulled_at
        """,
        rows,
    )


def upsert_gameweeks(
    cur, bootstrap: dict, season: str, player_id_to_code: dict[int, int], pulled_at: datetime
) -> None:
    def _code(el_id: int | None) -> int | None:
        return player_id_to_code.get(el_id) if el_id else None

    rows = [
        (
            season, ev["id"], ev["name"], ev["deadline_time"],
            ev.get("finished", False), ev.get("data_checked", False), ev.get("is_current", False),
            ev.get("average_entry_score"), ev.get("highest_score"),
            _code(ev.get("most_selected")), _code(ev.get("most_captained")),
            _code(ev.get("most_vice_captained")), _code(ev.get("top_element")),
            json.dumps(ev.get("chip_plays")), pulled_at,
        )
        for ev in bootstrap["events"]
    ]
    _upsert(
        cur,
        """
        INSERT INTO gameweeks (
            season, event_id, name, deadline_time, finished, data_checked, is_current,
            average_entry_score, highest_score, most_selected, most_captained,
            most_vice_captained, top_element, chip_plays, pulled_at
        ) VALUES %s
        ON CONFLICT (season, event_id) DO UPDATE SET
            name = EXCLUDED.name, deadline_time = EXCLUDED.deadline_time,
            finished = EXCLUDED.finished, data_checked = EXCLUDED.data_checked,
            is_current = EXCLUDED.is_current,
            average_entry_score = EXCLUDED.average_entry_score, highest_score = EXCLUDED.highest_score,
            most_selected = EXCLUDED.most_selected, most_captained = EXCLUDED.most_captained,
            most_vice_captained = EXCLUDED.most_vice_captained, top_element = EXCLUDED.top_element,
            chip_plays = EXCLUDED.chip_plays, pulled_at = EXCLUDED.pulled_at
        """,
        rows,
    )


def upsert_fixtures(
    cur, fixtures: list, season: str, team_id_to_code: dict[int, int], pulled_at: datetime
) -> None:
    rows = [
        (
            season, fx["id"], fx.get("event"),
            team_id_to_code[fx["team_h"]], team_id_to_code[fx["team_a"]],
            fx.get("team_h_score"), fx.get("team_a_score"), fx.get("kickoff_time"),
            fx.get("finished", False), fx.get("finished_provisional", False),
            fx.get("team_h_difficulty"), fx.get("team_a_difficulty"),
            pulled_at,
        )
        for fx in fixtures
    ]
    _upsert(
        cur,
        """
        INSERT INTO fixtures (
            season, fpl_fixture_id, event_id, team_h_code, team_a_code,
            team_h_score, team_a_score, kickoff_time, finished, finished_provisional,
            team_h_difficulty, team_a_difficulty, pulled_at
        ) VALUES %s
        ON CONFLICT (season, fpl_fixture_id) DO UPDATE SET
            event_id = EXCLUDED.event_id,
            team_h_score = EXCLUDED.team_h_score, team_a_score = EXCLUDED.team_a_score,
            kickoff_time = EXCLUDED.kickoff_time, finished = EXCLUDED.finished,
            finished_provisional = EXCLUDED.finished_provisional,
            team_h_difficulty = EXCLUDED.team_h_difficulty, team_a_difficulty = EXCLUDED.team_a_difficulty,
            pulled_at = EXCLUDED.pulled_at
        """,
        rows,
    )


def upsert_price_snapshots(cur, bootstrap: dict, season: str, pulled_at: datetime) -> None:
    today = date.today()
    rows = [
        (
            season, el["code"], today, el["now_cost"], el.get("cost_change_event"),
            el.get("cost_change_start"), _to_float(el.get("selected_by_percent")),
            el.get("transfers_in_event"), el.get("transfers_out_event"),
            el.get("status"), el.get("chance_of_playing_next_round"), el.get("news"),
            pulled_at,
        )
        for el in bootstrap["elements"]
    ]
    _upsert(
        cur,
        """
        INSERT INTO player_price_snapshots (
            season, player_code, snapshot_date, now_cost, cost_change_event, cost_change_start,
            selected_by_percent, transfers_in_event, transfers_out_event, status,
            chance_of_playing_next_round, news, pulled_at
        ) VALUES %s
        ON CONFLICT (season, player_code, snapshot_date) DO UPDATE SET
            now_cost = EXCLUDED.now_cost, cost_change_event = EXCLUDED.cost_change_event,
            cost_change_start = EXCLUDED.cost_change_start,
            selected_by_percent = EXCLUDED.selected_by_percent,
            transfers_in_event = EXCLUDED.transfers_in_event,
            transfers_out_event = EXCLUDED.transfers_out_event, status = EXCLUDED.status,
            chance_of_playing_next_round = EXCLUDED.chance_of_playing_next_round,
            news = EXCLUDED.news, pulled_at = EXCLUDED.pulled_at
        """,
        rows,
    )


def upsert_gameweek_stats(
    cur,
    live_payload: dict,
    event_id: int,
    season: str,
    player_id_to_code: dict[int, int],
    player_id_to_team_code: dict[int, int],
    pulled_at: datetime,
) -> None:
    rows = []
    for el in live_payload["elements"]:
        s = el["stats"]
        player_id = el["id"]
        if player_id not in player_id_to_code or player_id not in player_id_to_team_code:
            continue  # player not in the current bootstrap pull (e.g. dropped from the game)
        rows.append(
            (
                season, event_id, player_id_to_code[player_id], player_id_to_team_code[player_id],
                s["minutes"], s.get("starts", 0), s["goals_scored"], s["assists"], s["clean_sheets"],
                s["goals_conceded"], s["own_goals"], s["penalties_saved"], s["penalties_missed"],
                s["yellow_cards"], s["red_cards"], s["saves"], s["bonus"], s["bps"],
                _to_float(s.get("influence")), _to_float(s.get("creativity")),
                _to_float(s.get("threat")), _to_float(s.get("ict_index")),
                s.get("clearances_blocks_interceptions", 0), s.get("tackles", 0), s.get("recoveries", 0),
                s.get("defensive_contribution", 0),
                _to_float(s.get("expected_goals")), _to_float(s.get("expected_assists")),
                _to_float(s.get("expected_goal_involvements")), _to_float(s.get("expected_goals_conceded")),
                s["total_points"], s.get("in_dreamteam", False),
                json.dumps(el.get("explain")), pulled_at,
            )
        )
    _upsert(
        cur,
        """
        INSERT INTO player_gameweek_stats (
            season, event_id, player_code, team_code, minutes, starts, goals_scored, assists,
            clean_sheets, goals_conceded, own_goals, penalties_saved, penalties_missed,
            yellow_cards, red_cards, saves, bonus, bps, influence, creativity, threat, ict_index,
            clearances_blocks_interceptions, tackles, recoveries, defensive_contribution,
            expected_goals, expected_assists, expected_goal_involvements, expected_goals_conceded,
            total_points, in_dreamteam, explain, pulled_at
        ) VALUES %s
        ON CONFLICT (season, event_id, player_code) DO UPDATE SET
            team_code = EXCLUDED.team_code, minutes = EXCLUDED.minutes, starts = EXCLUDED.starts,
            goals_scored = EXCLUDED.goals_scored, assists = EXCLUDED.assists,
            clean_sheets = EXCLUDED.clean_sheets, goals_conceded = EXCLUDED.goals_conceded,
            own_goals = EXCLUDED.own_goals, penalties_saved = EXCLUDED.penalties_saved,
            penalties_missed = EXCLUDED.penalties_missed, yellow_cards = EXCLUDED.yellow_cards,
            red_cards = EXCLUDED.red_cards, saves = EXCLUDED.saves, bonus = EXCLUDED.bonus,
            bps = EXCLUDED.bps, influence = EXCLUDED.influence, creativity = EXCLUDED.creativity,
            threat = EXCLUDED.threat, ict_index = EXCLUDED.ict_index,
            clearances_blocks_interceptions = EXCLUDED.clearances_blocks_interceptions,
            tackles = EXCLUDED.tackles, recoveries = EXCLUDED.recoveries,
            defensive_contribution = EXCLUDED.defensive_contribution,
            expected_goals = EXCLUDED.expected_goals, expected_assists = EXCLUDED.expected_assists,
            expected_goal_involvements = EXCLUDED.expected_goal_involvements,
            expected_goals_conceded = EXCLUDED.expected_goals_conceded,
            total_points = EXCLUDED.total_points, in_dreamteam = EXCLUDED.in_dreamteam,
            explain = EXCLUDED.explain, pulled_at = EXCLUDED.pulled_at
        """,
        rows,
    )


def run(season: str | None = None) -> None:
    season = season or settings.SEASON
    client = FPLClient()
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                print("Pulling bootstrap-static...")
                bootstrap = client.get_bootstrap_static()
                pulled_at = _now()
                insert_raw_snapshot(cur, "bootstrap-static", season, None, bootstrap)

                team_id_to_code = upsert_teams(cur, bootstrap)
                upsert_team_season_stats(cur, bootstrap, season, pulled_at)
                player_id_to_code = upsert_players(cur, bootstrap)
                upsert_player_seasons(cur, bootstrap, season, team_id_to_code, pulled_at)
                upsert_gameweeks(cur, bootstrap, season, player_id_to_code, pulled_at)
                upsert_price_snapshots(cur, bootstrap, season, pulled_at)

                player_id_to_team_code = {
                    el["id"]: team_id_to_code[el["team"]] for el in bootstrap["elements"]
                }

                print("Pulling fixtures...")
                fixtures = client.get_fixtures()
                insert_raw_snapshot(cur, "fixtures", season, None, fixtures)
                upsert_fixtures(cur, fixtures, season, team_id_to_code, pulled_at)

                events_to_pull = [
                    ev["id"] for ev in bootstrap["events"]
                    if ev.get("finished") or ev.get("is_current")
                ]

            for event_id in events_to_pull:
                print(f"Pulling live stats for GW{event_id}...")
                live_payload = client.get_event_live(event_id)
                live_pulled_at = _now()
                with conn.cursor() as cur:
                    insert_raw_snapshot(cur, f"event/{event_id}/live", season, event_id, live_payload)
                    upsert_gameweek_stats(
                        cur, live_payload, event_id, season,
                        player_id_to_code, player_id_to_team_code, live_pulled_at,
                    )
                conn.commit()
                time.sleep(LIVE_PULL_DELAY_SECONDS)

        print(f"Done. Ingested {len(events_to_pull)} gameweek(s) for season {season}.")
    finally:
        conn.close()


if __name__ == "__main__":
    run()
