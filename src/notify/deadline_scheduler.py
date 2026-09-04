"""Deadline reminder scheduler - the notifications-only piece of the semi-autopilot plan
(src/fpl_write and this module are the two verified primitives; this is the first thing built
on top of them). Meant to run on a schedule via GitHub Actions (see
.github/workflows/deadline_reminders.yml), not on Brian's own machine, so reminders keep firing
even when his PC is off. DATABASE_URL therefore points at Supabase in that context, not local
Postgres - this module doesn't care which, it just uses config.settings like everything else.

Does NOT submit any transfer/lineup/captain change - see the hard rule in project memory
(src/fpl_write/client.py's docstring) about never firing a confirmed transfer outside a real,
deliberate, user-approved submission. This only tells Brian what the engine currently
recommends; he still acts on it himself (or a later negotiation-loop piece will, once the
transfer write path is trusted - not yet).

Idempotent via the reminder_log table (sql/reminder_log.sql): safe to run on any cron cadence
finer than the gap between tiers, since each (season, event_id, tier) only ever fires once.

Run directly (needs DATABASE_URL/FPL_ENTRY_ID/FPL_SEASON/TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID -
via .env locally, or real env vars in CI):
    python -m src.notify.deadline_scheduler
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone

import pandas as pd

# Windows' console defaults to cp1252, which can't encode the emoji in reminder messages -
# without this, a local `python -m src.notify.deadline_scheduler` run crashes on the first
# print() of a built message before it ever reaches Telegram. GitHub Actions' Ubuntu runners
# default to UTF-8 already, so this is a no-op there.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from config import settings
from src.ingestion.db import get_connection as _get_connection
from src.notify.telegram_bot import send_message
from src.recommendations.squad_optimizer import best_starting_xi
from src.recommendations.transfers import compute_free_transfers, suggest_transfer_plan

# (tier name, hours before deadline the window opens). Checked in order; a tier fires once
# `now` has crossed into its window and no reminder_log row exists yet for it. Not confirmed
# in fine detail with Brian - defaults from the original autopilot plan discussion, adjust here
# if he wants different offsets.
TIERS = [("T-24h", 24.0), ("T-3h", 3.0), ("T-30m", 0.5)]


def get_connection():
    # src.ingestion.db.get_connection() is the shared, retry-wrapped helper (covers a real
    # Supavisor-side cold-pool auth bug - see its docstring). autocommit=True here is
    # scheduler-specific: log_sent() below INSERTs with no explicit commit() call, relying on it.
    conn = _get_connection()
    conn.autocommit = True
    return conn


def load_next_deadline(conn) -> dict | None:
    """The soonest not-yet-finished gameweek. None if every ingested gameweek is finished
    (nothing upcoming to remind about) or none have been ingested at all."""
    df = pd.read_sql_query(
        """
        SELECT season, event_id, name, deadline_time
        FROM gameweeks
        WHERE season = %(season)s AND NOT finished AND deadline_time > now()
        ORDER BY deadline_time ASC
        LIMIT 1
        """,
        conn, params={"season": settings.SEASON},
    )
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "season": row["season"], "event_id": int(row["event_id"]),
        "name": row["name"], "deadline_time": row["deadline_time"],
    }


def already_sent(conn, season: str, event_id: int, tier: str) -> bool:
    df = pd.read_sql_query(
        "SELECT 1 FROM reminder_log WHERE season = %(s)s AND event_id = %(e)s AND tier = %(t)s",
        conn, params={"s": season, "e": event_id, "t": tier},
    )
    return not df.empty


def log_sent(conn, season: str, event_id: int, tier: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO reminder_log (season, event_id, tier) VALUES (%s, %s, %s) "
            "ON CONFLICT (season, event_id, tier) DO NOTHING",
            (season, event_id, tier),
        )


def load_rankings(conn, season: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT
            st.player_code, p.web_name, pos.element_type, pos.fpl_id, t.short_name AS team,
            st.now_cost, st.status,
            pr.predicted_points
        FROM v_player_season_totals st
        JOIN players p ON p.player_code = st.player_code
        JOIN player_seasons pos ON pos.player_code = st.player_code AND pos.season = st.season
        JOIN teams t ON t.team_code = st.team_code
        LEFT JOIN predictions pr ON pr.player_code = st.player_code AND pr.season = st.season
            AND pr.event_id = (SELECT MAX(event_id) FROM predictions WHERE season = st.season)
        WHERE st.season = %(season)s
        """,
        conn, params={"season": season},
    )
    position_names = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    df["position"] = df["element_type"].map(position_names)
    df["price"] = df["now_cost"] / 10
    df["predicted_points"] = df["predicted_points"].fillna(0.0)
    return df


def load_squad(conn, season: str, entry_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    manager_gw = pd.read_sql_query(
        "SELECT * FROM manager_gameweeks WHERE entry_id = %(e)s AND season = %(s)s ORDER BY event_id DESC",
        conn, params={"e": entry_id, "s": season},
    )
    squad = pd.read_sql_query(
        """
        SELECT
            sp.player_code, sp.multiplier, p.web_name, pos.element_type, pos.fpl_id,
            t.short_name AS team, lp.now_cost, pr.predicted_points
        FROM squad_picks sp
        JOIN players p ON p.player_code = sp.player_code
        JOIN player_seasons pos ON pos.player_code = sp.player_code AND pos.season = sp.season
        JOIN teams t ON t.team_code = pos.team_code
        JOIN LATERAL (
            SELECT now_cost FROM player_price_snapshots pps
            WHERE pps.player_code = sp.player_code AND pps.season = sp.season
            ORDER BY snapshot_date DESC LIMIT 1
        ) lp ON true
        LEFT JOIN predictions pr ON pr.player_code = sp.player_code AND pr.season = sp.season
            AND pr.event_id = (SELECT MAX(event_id) FROM predictions WHERE season = sp.season)
        WHERE sp.entry_id = %(e)s AND sp.season = %(s)s
            AND sp.event_id = (SELECT MAX(event_id) FROM squad_picks WHERE entry_id = %(e)s AND season = %(s)s)
        """,
        conn, params={"e": entry_id, "s": season},
    )
    position_names = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    squad["position"] = squad["element_type"].map(position_names)
    squad["price"] = squad["now_cost"] / 10
    squad["predicted_points"] = squad["predicted_points"].fillna(0.0)
    return squad, manager_gw


def build_message(conn, tier: str, gw: dict, hours_left: float) -> str:
    lines = [f"⏰ {tier} - {gw['name']} deadline in ~{hours_left:.1f}h"]

    if not settings.ENTRY_ID:
        lines.append("\n(No FPL_ENTRY_ID configured - deadline-only reminder.)")
        return "\n".join(lines)

    squad, manager_gw = load_squad(conn, gw["season"], settings.ENTRY_ID)
    if squad.empty:
        lines.append("\n(No squad data ingested yet for this entry.)")
        return "\n".join(lines)

    rankings = load_rankings(conn, gw["season"])
    bank = (manager_gw.iloc[0]["bank"] / 10) if not manager_gw.empty else 0.0
    free_transfers = compute_free_transfers(manager_gw)

    plan, _, remaining_bank = suggest_transfer_plan(squad, rankings, bank, free_transfers)
    if plan.empty:
        lines.append("\nNo recommended transfers this week.")
    else:
        lines.append(f"\nSuggested transfers ({free_transfers} free):")
        for _, row in plan.iterrows():
            hit_note = "" if row["free_transfer_used"] else f" (-{4} hit)"
            lines.append(f"  {row['sell']} → {row['buy']} (net {row['net']:+.1f}){hit_note}")

    starting_xi, formation = best_starting_xi(squad)
    captain = starting_xi.sort_values("predicted_points", ascending=False).iloc[0]
    lines.append(f"\nSuggested captain: {captain['web_name']} ({captain['predicted_points']:.1f} pred pts)")
    lines.append(f"Formation: {formation[0]}-{formation[1]}-{formation[2]}")
    lines.append(
        "\n⚠️ This is a recommendation only - nothing has been submitted to your FPL "
        "team automatically."
    )
    return "\n".join(lines)


def run() -> None:
    conn = get_connection()
    gw = load_next_deadline(conn)
    if gw is None:
        print("No upcoming deadline found - nothing to do.")
        return

    now = datetime.now(timezone.utc)
    hours_left = (gw["deadline_time"] - now).total_seconds() / 3600
    print(f"Next deadline: {gw['name']} in {hours_left:.2f}h")

    for tier, window_hours in TIERS:
        if not (0 <= hours_left <= window_hours):
            continue
        if already_sent(conn, gw["season"], gw["event_id"], tier):
            print(f"{tier} already sent for {gw['name']} - skipping.")
            continue
        message = build_message(conn, tier, gw, hours_left)
        print(f"Sending {tier} reminder:\n{message}")
        asyncio.run(send_message(message))
        log_sent(conn, gw["season"], gw["event_id"], tier)
        print(f"{tier} reminder sent and logged.")


if __name__ == "__main__":
    run()
