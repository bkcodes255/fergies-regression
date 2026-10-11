"""The dashboard's SQL loaders, shared by the Streamlit app (dashboard/app.py wraps each in
st.cache_data) and the web site's data export (src/web_data/this_week.py), so both read the
same rows the same way. Each takes any connection or engine pandas can read from.
"""
from __future__ import annotations

import pandas as pd


def load_player_rankings(conn, season: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT
            st.player_code, p.web_name, pos.element_type, t.short_name AS team,
            st.now_cost, st.total_points, st.games_played, st.points_per_90,
            st.points_per_million, st.selected_by_percent, st.status,
            pr.predicted_points, pr.event_id AS predicted_gw,
            pr.p_return_6plus, pr.p_haul_10plus,
            pr.floor_points, pr.median_points, pr.ceiling_points
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
    df["predicted_value"] = df["predicted_points"] / df["price"]
    return df


def load_squad(conn, season: str, entry_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (squad_df, manager_gw_df). manager_gw_df has ALL ingested gameweeks for this
    entry (needed for compute_free_transfers, which needs the whole event_transfers history,
    not just the latest row) - iloc[0] is still the most recent gameweek, ordered descending."""
    manager_gw = pd.read_sql_query(
        """
        SELECT * FROM manager_gameweeks
        WHERE entry_id = %(entry_id)s AND season = %(season)s
        ORDER BY event_id DESC
        """,
        conn, params={"entry_id": entry_id, "season": season},
    )
    squad = pd.read_sql_query(
        """
        SELECT
            sp.player_code, sp.squad_position, sp.multiplier, sp.is_captain, sp.is_vice_captain,
            p.web_name, pos.element_type, t.short_name AS team, lp.now_cost,
            pr.predicted_points, pr.p_haul_10plus,
            pr.floor_points, pr.median_points, pr.ceiling_points
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
        WHERE sp.entry_id = %(entry_id)s AND sp.season = %(season)s
            AND sp.event_id = (
                SELECT MAX(event_id) FROM squad_picks WHERE entry_id = %(entry_id)s AND season = %(season)s
            )
        ORDER BY sp.squad_position
        """,
        conn, params={"entry_id": entry_id, "season": season},
    )
    position_names = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    squad["position"] = squad["element_type"].map(position_names)
    squad["price"] = squad["now_cost"] / 10
    return squad, manager_gw


def load_last_ingested_gw(conn, season: str) -> int | None:
    row = pd.read_sql_query(
        "SELECT MAX(event_id) AS gw FROM player_gameweek_stats WHERE season = %(season)s",
        conn, params={"season": season},
    )
    return None if row["gw"].isna().all() else int(row["gw"].iloc[0])


def load_next_deadline(conn, season: str) -> dict | None:
    df = pd.read_sql_query(
        """
        SELECT event_id, name, deadline_time FROM gameweeks
        WHERE season = %(season)s AND NOT finished AND deadline_time > now()
        ORDER BY deadline_time ASC LIMIT 1
        """,
        conn, params={"season": season},
    )
    if df.empty:
        return None
    row = df.iloc[0]
    return {"event_id": int(row["event_id"]), "name": row["name"], "deadline_time": row["deadline_time"]}


def load_fixture_grid(conn, season: str, start_gw: int, horizon: int) -> pd.DataFrame:
    """Every team's fixture(s) across [start_gw, start_gw + horizon) - not just the immediate
    next one (v_fixture_difficulty already carries one row per upcoming fixture; the Fixtures
    tab used to throw all but the earliest away via DISTINCT ON, which is why it could only
    ever show one gameweek). A team with two fixtures in one event_id (a double gameweek)
    naturally produces two rows here; a team with none (a blank) produces zero - the caller
    handles both."""
    return pd.read_sql_query(
        """
        SELECT th.short_name AS team, ta.short_name AS opponent, fd.is_home,
               fd.event_id, fd.fixture_difficulty_score
        FROM v_fixture_difficulty fd
        JOIN teams th ON th.team_code = fd.team_code
        JOIN teams ta ON ta.team_code = fd.opponent_code
        WHERE fd.season = %(season)s AND fd.event_id >= %(start_gw)s AND fd.event_id < %(start_gw)s + %(horizon)s
        ORDER BY th.short_name, fd.event_id
        """,
        conn, params={"season": season, "start_gw": start_gw, "horizon": horizon},
    )
