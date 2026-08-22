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
from src.recommendations.squad_optimizer import best_starting_xi
from src.recommendations.transfers import suggest_transfers

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
            st.player_code, p.web_name, pos.element_type, t.short_name AS team,
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
def load_squad(season: str, entry_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (squad_df, manager_gw_df) for the entry's most recent ingested gameweek."""
    conn = get_connection()
    manager_gw = pd.read_sql_query(
        """
        SELECT * FROM manager_gameweeks
        WHERE entry_id = %(entry_id)s AND season = %(season)s
        ORDER BY event_id DESC LIMIT 1
        """,
        conn, params={"entry_id": entry_id, "season": season},
    )
    squad = pd.read_sql_query(
        """
        SELECT
            sp.player_code, sp.squad_position, sp.multiplier, sp.is_captain, sp.is_vice_captain,
            p.web_name, pos.element_type, t.short_name AS team, lp.now_cost,
            pr.predicted_points
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

tab_names = ["Player Rankings", "Fixture Planner", "Transfer Targets"]
if settings.ENTRY_ID:
    tab_names.insert(0, "My Squad")
tabs = st.tabs(tab_names)
tab_lookup = dict(zip(tab_names, tabs))
tab_rankings = tab_lookup["Player Rankings"]
tab_fixtures = tab_lookup["Fixture Planner"]
tab_targets = tab_lookup["Transfer Targets"]

if settings.ENTRY_ID:
    with tab_lookup["My Squad"]:
        squad, manager_gw = load_squad(season, settings.ENTRY_ID)
        if squad.empty:
            st.warning("No squad data ingested yet. Run `python -m src.ingestion.load_manager` first.")
        else:
            gw_row = manager_gw.iloc[0] if not manager_gw.empty else None
            if gw_row is not None:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("GW points", int(gw_row["points"]))
                c2.metric("Overall rank", f"{int(gw_row['overall_rank']):,}" if pd.notna(gw_row["overall_rank"]) else "—")
                c3.metric("Team value", f"£{gw_row['team_value']/10:.1f}m")
                c4.metric("Bank", f"£{gw_row['bank']/10:.1f}m")

            starting = squad[squad["multiplier"] > 0].copy()
            bench = squad[squad["multiplier"] == 0].copy()

            projected_total = (starting["predicted_points"] * starting["multiplier"]).sum()
            st.metric("Projected next-GW points (starting XI, with captain)", round(projected_total, 1))

            best_captain_idx = starting["predicted_points"].idxmax() if not starting["predicted_points"].isna().all() else None
            current_captain = starting[starting["is_captain"]]
            if best_captain_idx is not None and not current_captain.empty:
                best_captain_name = starting.loc[best_captain_idx, "web_name"]
                current_captain_name = current_captain.iloc[0]["web_name"]
                if best_captain_name != current_captain_name:
                    st.warning(
                        f"Model suggests **{best_captain_name}** as captain "
                        f"(predicted {starting.loc[best_captain_idx, 'predicted_points']:.2f}) over your "
                        f"current pick **{current_captain_name}** "
                        f"(predicted {current_captain.iloc[0]['predicted_points']:.2f})."
                    )
                else:
                    st.success(f"Your captain **{current_captain_name}** is also the model's top pick.")

            def _format_squad(df: pd.DataFrame) -> pd.DataFrame:
                labeled = df.copy()
                labeled["role"] = labeled.apply(
                    lambda r: "C" if r["is_captain"] else ("VC" if r["is_vice_captain"] else ""), axis=1
                )
                return labeled[["web_name", "position", "team", "price", "predicted_points", "role"]].rename(
                    columns={"web_name": "Player", "position": "Pos", "team": "Team",
                              "price": "Price (£m)", "predicted_points": "Predicted (next GW)", "role": ""}
                )

            st.markdown("**Starting XI**")
            st.dataframe(_format_squad(starting), use_container_width=True, hide_index=True)
            st.markdown("**Bench**")
            st.dataframe(_format_squad(bench), use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Recommended starting XI (auto-substitution)")
            st.caption(
                "Best valid formation from your actual 15, maximizing predicted points. "
                "Brute-forces all 8 legal FPL formations (GKP fixed at 1; DEF 3-5, MID 2-5, "
                "FWD 1-3, 11 total) — within a fixed formation the top-N players per position "
                "is always optimal, so this is exact, not a heuristic."
            )
            optimal_xi, formation = best_starting_xi(squad)
            optimal_total = optimal_xi["predicted_points"].sum()
            actual_total = starting["predicted_points"].sum()  # unweighted, for apples-to-apples vs optimal_total
            if formation is None:
                st.warning("Could not find a valid formation from this squad.")
            else:
                d, m, f = formation
                st.metric(
                    f"Optimal formation: 1-{d}-{m}-{f}",
                    round(optimal_total, 2),
                    delta=round(optimal_total - actual_total, 2) if abs(optimal_total - actual_total) > 0.01 else None,
                    help="Delta vs. your actual starting XI's (unweighted) predicted total.",
                )
                st.dataframe(
                    optimal_xi[["web_name", "position", "predicted_points"]].rename(columns={
                        "web_name": "Player", "position": "Pos", "predicted_points": "Predicted (next GW)",
                    }),
                    use_container_width=True, hide_index=True,
                )

            st.divider()
            st.subheader("Transfer suggestions")
            st.caption(
                "Same position, affordable with your current price + bank, respects the max-3-per-team "
                "rule. Each row is an independent 1-for-1 comparison — not a coordinated multi-transfer "
                "plan, so don't read two rows recommending the same buy as 'buy them twice'. Doesn't "
                "account for a transfer hit (-4) if you're out of free transfers."
            )
            bank = float(gw_row["bank"]) / 10 if gw_row is not None else 0.0
            suggestions = suggest_transfers(squad, rankings, bank, top_n=2)
            if suggestions.empty:
                st.info("No positive-gain transfer suggestions found within budget.")
            else:
                st.dataframe(
                    suggestions.rename(columns={
                        "sell": "Sell", "sell_position": "Pos", "sell_predicted": "Sell predicted",
                        "sell_price": "Sell price", "buy": "Buy", "buy_predicted": "Buy predicted",
                        "buy_price": "Buy price", "net_gain": "Net gain",
                    }),
                    use_container_width=True, hide_index=True,
                )

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
    "Predictions come from a Random Forest model trained on 3 historical seasons; see "
    "notebooks/06_model_comparison.ipynb for backtested accuracy. Squad optimization / "
    "auto-substitution and full transfer-in-out comparisons are Phase 5 (decision engine), "
    "not built yet."
)
