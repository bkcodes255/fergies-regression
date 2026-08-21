"""Fergie's Regression - Phase 3 MVP dashboard.

Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import psycopg2
import streamlit as st

from config import settings

st.set_page_config(page_title="Fergie's Regression", layout="wide")


@st.cache_resource
def get_connection():
    return psycopg2.connect(settings.DATABASE_URL)


@st.cache_data(ttl=300)
def load_player_rankings(season: str) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT
            p.web_name, pos.element_type, t.short_name AS team,
            st.now_cost, st.total_points, st.games_played, st.points_per_90,
            st.points_per_million, st.selected_by_percent, st.status,
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
    df["predicted_value"] = df["predicted_points"] / df["price"]
    return df


@st.cache_data(ttl=300)
def load_fixture_difficulty(season: str) -> pd.DataFrame:
    conn = get_connection()
    return pd.read_sql_query(
        """
        SELECT DISTINCT ON (fd.team_code)
               th.short_name AS team, ta.short_name AS opponent, fd.is_home,
               fd.fpl_fdr, fd.fixture_difficulty_score, fd.kickoff_time
        FROM v_fixture_difficulty fd
        JOIN teams th ON th.team_code = fd.team_code
        JOIN teams ta ON ta.team_code = fd.opponent_code
        WHERE fd.season = %(season)s
        ORDER BY fd.team_code, fd.kickoff_time
        """,
        conn, params={"season": season},
    ).sort_values("fixture_difficulty_score")


@st.cache_data(ttl=300)
def load_last_ingested_gw(season: str) -> int | None:
    conn = get_connection()
    row = pd.read_sql_query(
        "SELECT MAX(event_id) AS gw FROM player_gameweek_stats WHERE season = %(season)s",
        conn, params={"season": season},
    )
    return None if row["gw"].isna().all() else int(row["gw"].iloc[0])


season = settings.SEASON
st.title("Fergie's Regression")
st.caption("An FPL Analytics Report")

last_gw = load_last_ingested_gw(season)
rankings = load_player_rankings(season)
predicted_gw = rankings["predicted_gw"].dropna().max() if not rankings.empty else None

col1, col2, col3 = st.columns(3)
col1.metric("Season", season)
col2.metric("Last ingested gameweek", last_gw if last_gw is not None else "—")
col3.metric("Predictions for", f"GW{int(predicted_gw)}" if predicted_gw and pd.notna(predicted_gw) else "—")

if last_gw is not None and last_gw <= 2:
    st.info(
        f"Early season ({last_gw} gameweek(s) ingested) — rolling form and fixture difficulty "
        "are still thin-sample and will sharpen as more gameweeks are ingested. This is "
        "expected behavior, not a bug."
    )

tab_rankings, tab_fixtures, tab_targets = st.tabs(["Player Rankings", "Fixture Planner", "Transfer Targets"])

with tab_rankings:
    st.subheader("Player rankings")
    positions = ["All"] + sorted(rankings["position"].dropna().unique().tolist())
    pos_filter = st.selectbox("Position", positions, key="rankings_pos")
    min_minutes = st.slider("Minimum minutes played (season)", 0, 90, 0, step=15, key="rankings_min")

    view = rankings.copy()
    if pos_filter != "All":
        view = view[view["position"] == pos_filter]
    view = view[view["total_points"].notna()]

    st.dataframe(
        view.sort_values("predicted_points", ascending=False, na_position="last")[
            ["web_name", "position", "team", "price", "predicted_points",
             "total_points", "points_per_90", "points_per_million", "selected_by_percent", "status"]
        ].rename(columns={
            "web_name": "Player", "position": "Pos", "team": "Team", "price": "Price (£m)",
            "predicted_points": "Predicted (next GW)", "total_points": "Total pts",
            "points_per_90": "Pts/90", "points_per_million": "Pts/£m",
            "selected_by_percent": "Owned %", "status": "Status",
        }),
        use_container_width=True, height=500,
    )

with tab_fixtures:
    st.subheader("Each team's next fixture, easiest to hardest")
    st.caption("Lower fixture difficulty score = easier fixture. v1 formula — see data/data_dictionary.md.")
    fixtures = load_fixture_difficulty(season)
    st.dataframe(
        fixtures.rename(columns={
            "team": "Team", "opponent": "Opponent", "is_home": "Home", "fpl_fdr": "FPL FDR",
            "fixture_difficulty_score": "Difficulty Score", "kickoff_time": "Kickoff",
        }),
        use_container_width=True, height=500,
    )

with tab_targets:
    st.subheader("Transfer targets: predicted points per £m")
    min_predicted = st.slider("Minimum predicted points", 0.0, 10.0, 1.0, step=0.5, key="targets_min")
    max_ownership = st.slider("Maximum ownership % (differentials)", 0.0, 100.0, 100.0, step=5.0, key="targets_own")

    targets = rankings[
        (rankings["predicted_points"] >= min_predicted) &
        (rankings["selected_by_percent"] <= max_ownership) &
        (rankings["status"] == "a")
    ].sort_values("predicted_value", ascending=False)

    st.dataframe(
        targets[["web_name", "position", "team", "price", "predicted_points",
                  "predicted_value", "selected_by_percent"]].rename(columns={
            "web_name": "Player", "position": "Pos", "team": "Team", "price": "Price (£m)",
            "predicted_points": "Predicted (next GW)", "predicted_value": "Predicted pts/£m",
            "selected_by_percent": "Owned %",
        }),
        use_container_width=True, height=500,
    )

st.divider()
st.caption(
    "Squad view and transfer-in/out comparisons against an actual squad need your FPL entry "
    "ID hooked up — not built yet. Predictions come from a Random Forest model trained on 3 "
    "historical seasons; see notebooks/06_model_comparison.ipynb for backtested accuracy."
)
