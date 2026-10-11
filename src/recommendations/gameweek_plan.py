"""The current gameweek's recommendation: squad, rankings, transfer plans (next gameweek and
5-gameweek horizon), best XI and captain - one definition shared by everything that needs it.

Moved here unchanged from src/notify/deadline_scheduler.py so other readers (the web site's
data export, and the read-only tools server planned next) can use exactly the plan the Telegram
approval flow proposes and auto-submits, without importing the Telegram bot.
deadline_scheduler.py imports it from here. Takes any connection pandas can read from: a
psycopg2 connection (the scheduler) or a SQLAlchemy connection (the web export).
"""
from __future__ import annotations

import pandas as pd

from config import settings
from src.recommendations.horizon import DEFAULT_HORIZON, compute_horizon_points, load_fixture_window
from src.recommendations.squad_optimizer import best_starting_xi
from src.recommendations.transfers import compute_free_transfers, suggest_transfer_plan


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


def load_rankings(conn, season: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT
            st.player_code, p.web_name, pos.element_type, t.short_name AS team,
            st.now_cost, st.status,
            pr.predicted_points, pr.event_id AS predicted_gw
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


def with_horizon(conn, rankings: pd.DataFrame, season: str) -> pd.DataFrame:
    """Adds horizon_points/horizon_fixtures (src.recommendations.horizon) to rankings that
    already has predicted_points + predicted_gw - same helper dashboard/app.py's own
    with_horizon applies for the "Next 5 GWs" toggle there, kept in sync so the Telegram
    approval flow's 5-GW option matches the dashboard's own 5-GW option exactly, not a
    second, subtly different implementation. Falls back to a flat 1-gameweek horizon when
    there's no next-GW prediction yet to anchor the window on (e.g. before the first
    predict_live run) - matches dashboard/app.py's own fallback."""
    predicted_gw = rankings["predicted_gw"].dropna().max() if not rankings.empty else None
    if predicted_gw is None or pd.isna(predicted_gw):
        result = rankings.copy()
        result["horizon_points"] = result["predicted_points"]
        result["horizon_fixtures"] = 1
        return result
    fixtures = load_fixture_window(conn, season, int(predicted_gw), DEFAULT_HORIZON)
    return compute_horizon_points(rankings, fixtures, DEFAULT_HORIZON)


def load_squad(conn, season: str, entry_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    manager_gw = pd.read_sql_query(
        "SELECT * FROM manager_gameweeks WHERE entry_id = %(e)s AND season = %(s)s ORDER BY event_id DESC",
        conn, params={"e": entry_id, "s": season},
    )
    squad = pd.read_sql_query(
        """
        SELECT
            sp.player_code, sp.multiplier, p.web_name, pos.element_type,
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


def compute_recommendation(conn, gw: dict) -> dict | None:
    """Core recommendation computation, shared by build_message (plain text, every tier) and
    the APPROVAL_TIER path (structured plans for approval_bot's buttons) - kept as one function
    so the two never quietly drift apart on what "the current recommendation" actually is.

    Builds the transfer plan two ways - value_col="predicted_points" (next GW only) and
    value_col="horizon_points" (fixture-weighted over the next DEFAULT_HORIZON gameweeks,
    same horizon dashboard/app.py's own toggle uses) - so the Telegram approval message can
    offer both and let the horizon choice be made at approval time (see
    approval_bot.build_transfer_plan/send_approval_requests), not decided unilaterally here."""
    if not settings.ENTRY_ID:
        return None
    squad, manager_gw = load_squad(conn, gw["season"], settings.ENTRY_ID)
    if squad.empty:
        return None

    rankings = load_rankings(conn, gw["season"])
    rankings = with_horizon(conn, rankings, gw["season"])
    # squad needs horizon_points too - suggest_transfer_plan compares current[value_col] against
    # cand[value_col] for every squad player, and rankings is the only frame with horizon_points
    # computed above. Same merge dashboard/app.py's "This Week" tab needs for the identical
    # reason (a KeyError here previously broke every "Next 5 GWs" plan on the dashboard).
    squad = squad.merge(rankings[["player_code", "horizon_points", "horizon_fixtures"]], on="player_code", how="left")

    bank = (manager_gw.iloc[0]["bank"] / 10) if not manager_gw.empty else 0.0
    free_transfers = compute_free_transfers(manager_gw)
    plan_1gw, squad_after_1gw, bank_after_1gw = suggest_transfer_plan(
        squad, rankings, bank, free_transfers, value_col="predicted_points"
    )
    plan_5gw, squad_after_5gw, bank_after_5gw = suggest_transfer_plan(
        squad, rankings, bank, free_transfers, value_col="horizon_points"
    )
    starting_xi, formation = best_starting_xi(squad)
    bench = squad[~squad["player_code"].isin(starting_xi["player_code"])].sort_values(
        "predicted_points", ascending=False
    )
    captain = starting_xi.sort_values("predicted_points", ascending=False).iloc[0]
    vice = starting_xi.sort_values("predicted_points", ascending=False).iloc[1]
    return {
        # "plan" stays the next-GW-only plan - build_message's plain-text T-24h/T-30m reminders
        # keep showing just that one, unchanged from before; plan_1gw/plan_5gw are the pair
        # approval_bot needs for the two-button choice at APPROVAL_TIER.
        "plan": plan_1gw, "plan_1gw": plan_1gw, "plan_5gw": plan_5gw, "free_transfers": free_transfers,
        "starting_xi": starting_xi, "bench": bench, "formation": formation,
        "captain": captain, "vice": vice,
        # The squad and bank each plan would leave behind - what the tools server builds the
        # post-transfer lineup from. starting_xi/bench/captain above stay pre-transfer, exactly
        # as the Telegram approval flow has always proposed them.
        "squad": squad, "bank": bank,
        "squad_after_1gw": squad_after_1gw, "bank_after_1gw": bank_after_1gw,
        "squad_after_5gw": squad_after_5gw, "bank_after_5gw": bank_after_5gw,
        "rankings": rankings,
    }
