"""Fergie's Regression - Phase 3 MVP dashboard.

Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import settings
from src.features.historical_features import FEATURE_COLS as ALL_HISTORICAL_FEATURE_COLS
from src.features.historical_features import build_training_frame
from src.ingestion.db import get_connection, get_engine
from src.models.experiment import MODEL_TYPES, evaluate_feature_subset, paired_bootstrap_p_value
from src.models.monte_carlo import simulate_squad, summarize_simulation
from src.models.train import TEST_SEASON, TRAIN_SEASONS, direct_points_baseline
from src.recommendations.horizon import DEFAULT_HORIZON, compute_horizon_points, load_fixture_window
from src.recommendations.squad_builder import build_optimal_squad
from src.recommendations.squad_optimizer import best_starting_xi
from src.recommendations.transfers import compute_free_transfers, suggest_transfer_plan

st.set_page_config(page_title="Fergie's Regression", layout="wide")

# get_connection()/get_engine() come from src/ingestion/db.py - the shared, retry- and
# pool-health-aware DB helpers every part of this project uses now, not a dashboard-local copy.


@st.cache_data(ttl=300)
def load_player_rankings(season: str) -> pd.DataFrame:
    engine = get_engine()
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
        engine, params={"season": season},
    )
    position_names = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    df["position"] = df["element_type"].map(position_names)
    df["price"] = df["now_cost"] / 10
    df["predicted_value"] = df["predicted_points"] / df["price"]
    return df


@st.cache_data(ttl=300)
def load_fixture_difficulty(season: str) -> pd.DataFrame:
    engine = get_engine()
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
        engine, params={"season": season},
    ).sort_values("fixture_difficulty_score")


@st.cache_data(ttl=300)
def load_horizon_fixtures(season: str, start_gw: int, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    return load_fixture_window(get_engine(), season, start_gw, horizon)


def with_horizon(rankings_df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Adds horizon_points/horizon_fixtures (see src.recommendations.horizon) to any
    player-keyed frame that already has predicted_points + predicted_gw. Falls back to
    treating the horizon as 1 flat gameweek when there's no next-GW prediction yet to anchor
    the window on (e.g. before the first `predict_live` run)."""
    predicted_gw = rankings_df["predicted_gw"].dropna().max() if not rankings_df.empty else None
    if predicted_gw is None or pd.isna(predicted_gw):
        result = rankings_df.copy()
        result["horizon_points"] = result["predicted_points"]
        result["horizon_fixtures"] = 1
        return result
    fixtures = load_horizon_fixtures(season, int(predicted_gw), DEFAULT_HORIZON)
    return compute_horizon_points(rankings_df, fixtures, DEFAULT_HORIZON)


@st.cache_data(ttl=300)
def load_squad(season: str, entry_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (squad_df, manager_gw_df). manager_gw_df has ALL ingested gameweeks for this
    entry (needed for compute_free_transfers, which needs the whole event_transfers history,
    not just the latest row) - iloc[0] is still the most recent gameweek, ordered descending."""
    engine = get_engine()
    manager_gw = pd.read_sql_query(
        """
        SELECT * FROM manager_gameweeks
        WHERE entry_id = %(entry_id)s AND season = %(season)s
        ORDER BY event_id DESC
        """,
        engine, params={"entry_id": entry_id, "season": season},
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
        engine, params={"entry_id": entry_id, "season": season},
    )
    position_names = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    squad["position"] = squad["element_type"].map(position_names)
    squad["price"] = squad["now_cost"] / 10
    return squad, manager_gw


@st.cache_data(ttl=300)
def load_gw_comparison(season: str, entry_id: int) -> pd.DataFrame:
    """One row per elapsed gameweek: actual points vs GW average/highest, our model's pre-GW
    prediction, FPL's own ep_next-based prediction, and this GW's percentile — from
    v_manager_gw_comparison (sql/analytics.sql). model/epl predicted points are NaN for any
    gameweek where no prediction existed before that gameweek's deadline (structurally true
    for GW1, and for any gameweek before ep_next capture was added) - the UI must show that
    as "—", not 0."""
    return pd.read_sql_query(
        "SELECT * FROM v_manager_gw_comparison WHERE entry_id = %(entry_id)s AND season = %(season)s "
        "ORDER BY event_id",
        get_engine(), params={"entry_id": entry_id, "season": season},
    )


@st.cache_data(ttl=300)
def solve_optimal_squad(season: str, budget: float, value_col: str = "predicted_points"):
    """Cached on (season, budget, value_col) so an unrelated widget interaction elsewhere on
    the page doesn't re-run the ~1s solver - Streamlit re-executes the whole script on every
    interaction. value_col="horizon_points" builds for the multi-week window instead of just
    the next gameweek (src.recommendations.horizon)."""
    rankings_df = load_player_rankings(season)
    if value_col == "horizon_points":
        rankings_df = with_horizon(rankings_df, season)
    return build_optimal_squad(rankings_df, budget, value_col=value_col)


@st.cache_data(ttl=300)
def load_last_ingested_gw(season: str) -> int | None:
    row = pd.read_sql_query(
        "SELECT MAX(event_id) AS gw FROM player_gameweek_stats WHERE season = %(season)s",
        get_engine(), params={"season": season},
    )
    return None if row["gw"].isna().all() else int(row["gw"].iloc[0])


@st.cache_data(ttl=3600, show_spinner="Loading and engineering historical training data (first load only)...")
def load_model_lab_frame() -> tuple[pd.DataFrame, list[str]]:
    """The same historical training frame train.py fits on, cached for the session - this is
    the ~75s expensive part (load 6 seasons of CSVs + engineer every rolling/fixture feature),
    so every Model Lab "Run experiment" click only pays for re-fitting models, not
    re-engineering features."""
    df = build_training_frame()
    df["season_points_baseline"] = direct_points_baseline(df)
    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    return df, ALL_HISTORICAL_FEATURE_COLS + pos_cols


@st.cache_data(ttl=3600)
def load_deployed_baseline_results(_df: pd.DataFrame, all_features: list[str]) -> dict:
    """The 'all features' reference run, computed once per session - every experiment's paired
    significance test compares against this same fixed baseline. _df is excluded from
    Streamlit's cache key (leading underscore) since hashing the full training frame on every
    call would be slower than just re-running; all_features (small, hashable) is what actually
    varies this result."""
    return evaluate_feature_subset(_df, all_features, compute_extended=False)


def group_features(all_features: list[str]) -> dict[str, list[str]]:
    """Groups the ~45 model input columns for the Model Lab checkbox UI. Built from explicit
    membership + _roll3/_roll5 suffix matching (not a hardcoded full list) so it stays correct
    if FEATURE_COLS in historical_features.py grows - anything unmatched lands in "Other"
    rather than silently vanishing from the UI."""
    groups = {
        "Price & data flags": [c for c in all_features if c in ("price", "dc_data_available", "xg_data_available")],
        "Season-to-date": [c for c in all_features if c in ("season_points_per90_avg", "season_minutes_avg")],
        "Prior season": [c for c in all_features if c == "had_prior_season" or c.startswith("prev_season_")],
        "Fixture & opponent strength": [c for c in all_features if c in (
            "was_home", "own_attack_form", "own_defense_form", "opp_attack_form", "opp_defense_form"
        )],
        "Rolling form (3 GW)": [c for c in all_features if c.endswith("_roll3")],
        "Rolling form (5 GW)": [c for c in all_features if c.endswith("_roll5")],
        "Position": [c for c in all_features if c.startswith("pos_")],
    }
    grouped = {c for cols in groups.values() for c in cols}
    other = [c for c in all_features if c not in grouped]
    if other:
        groups["Other"] = other
    return groups


season = settings.SEASON
st.title("Fergie's Regression")
st.caption("An FPL Analytics Report")

last_gw = load_last_ingested_gw(season)
rankings = load_player_rankings(season)
predicted_gw = rankings["predicted_gw"].dropna().max() if not rankings.empty else None
rankings = with_horizon(rankings, season)

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

tab_names = ["Player Rankings", "Fixture Planner", "Transfer Targets", "Optimal Squad", "Model Lab"]
if settings.ENTRY_ID:
    tab_names.insert(0, "My Squad")
tabs = st.tabs(tab_names)
tab_lookup = dict(zip(tab_names, tabs))
tab_rankings = tab_lookup["Player Rankings"]
tab_fixtures = tab_lookup["Fixture Planner"]
tab_targets = tab_lookup["Transfer Targets"]
tab_optimal = tab_lookup["Optimal Squad"]
tab_model_lab = tab_lookup["Model Lab"]

if settings.ENTRY_ID:
    with tab_lookup["My Squad"]:
        squad, manager_gw = load_squad(season, settings.ENTRY_ID)
        squad = squad.merge(
            rankings[["player_code", "horizon_points", "horizon_fixtures"]], on="player_code", how="left"
        )
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

            best_ceiling_idx = (
                starting["p_haul_10plus"].idxmax() if not starting["p_haul_10plus"].isna().all() else None
            )
            if (
                best_ceiling_idx is not None and not current_captain.empty
                and starting.loc[best_ceiling_idx, "web_name"] != current_captain.iloc[0]["web_name"]
            ):
                st.info(
                    f"Highest-ceiling option in your XI: **{starting.loc[best_ceiling_idx, 'web_name']}** "
                    f"({starting.loc[best_ceiling_idx, 'p_haul_10plus']:.0%} chance of 10+ points) - "
                    "worth considering as a differential captain pick if you're chasing rank rather "
                    "than protecting it. The expected-points suggestion above is the safer play."
                )

            def _format_squad(df: pd.DataFrame) -> pd.DataFrame:
                labeled = df.copy()
                labeled["role"] = labeled.apply(
                    lambda r: "C" if r["is_captain"] else ("VC" if r["is_vice_captain"] else ""), axis=1
                )
                return labeled[
                    ["web_name", "position", "team", "price", "predicted_points", "p_haul_10plus", "role"]
                ].rename(columns={
                    "web_name": "Player", "position": "Pos", "team": "Team", "price": "Price (£m)",
                    "predicted_points": "Predicted (next GW)", "p_haul_10plus": "Ceiling %", "role": "",
                })

            st.markdown("**Starting XI**")
            st.dataframe(_format_squad(starting), use_container_width=True, hide_index=True)
            st.markdown("**Bench**")
            st.dataframe(_format_squad(bench), use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Monte Carlo: range of outcomes for your starting XI")
            st.caption(
                "10,000 simulated gameweeks, sampling each starter's points independently from "
                "their own floor/median/ceiling (quantile regression, not a guessed distribution "
                "shape). **Known simplification**: players are sampled independently - real "
                "outcomes correlate within a team and across a fixture, so the true spread is "
                "somewhat wider than shown here on both ends."
            )
            sim_totals = simulate_squad(starting, n_samples=10000)
            sim_stats = summarize_simulation(sim_totals)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Median", f"{sim_stats['p50']:.1f}")
            c2.metric("10th pct (bad week)", f"{sim_stats['p10']:.1f}")
            c3.metric("90th pct (great week)", f"{sim_stats['p90']:.1f}")
            c4.metric("P(80+ points)", f"{sim_stats['p_beat_80']:.1%}")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(sim_totals, bins=50, color="#4C72B0", alpha=0.85)
            for pct, label in [(10, "p10"), (50, "median"), (90, "p90")]:
                ax.axvline(np.percentile(sim_totals, pct), color="black", linestyle="--", linewidth=1)
            ax.set_xlabel("Simulated starting XI points (captain doubled)")
            ax.set_ylabel("Simulations")
            ax.set_title("Next-GW outcome distribution for your actual starting XI")
            plt.tight_layout()
            st.pyplot(fig)

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
            st.subheader("Recommended transfer plan")
            free_transfers = compute_free_transfers(manager_gw)
            bank = float(gw_row["bank"]) / 10 if gw_row is not None else 0.0
            plan_horizon = st.radio(
                "Judge transfers by", ["Next GW only", f"Next {DEFAULT_HORIZON} GWs (fixture-weighted)"],
                horizontal=True, key="plan_horizon",
            )
            plan_value_col = "horizon_points" if "GWs" in plan_horizon else "predicted_points"
            st.caption(
                f"You have **{free_transfers}** free transfer(s) available (computed from your "
                "transfer history — FPL's 2026/27 rule allows banking up to 5). This is a "
                "coordinated plan, not independent suggestions: each transfer accounts for the "
                "ones before it, so you'll never see the same buy target twice. Greedy — picks "
                "the single best transfer at each step, not a global search over combinations — "
                "and stops as soon as the next transfer wouldn't survive its hit cost. "
                + (
                    f"Ranking players on a {DEFAULT_HORIZON}-gameweek fixture-weighted total "
                    "(same next-GW model prediction, scaled per week by that team's fixture "
                    "difficulty — see the Fixture Planner tab) rather than just next gameweek, "
                    "so a good run of fixtures can justify a transfer a single-GW view would miss."
                    if plan_value_col == "horizon_points" else
                    "Ranking players on next gameweek's prediction only."
                )
            )
            plan, _, remaining_bank = suggest_transfer_plan(
                squad, rankings, bank, free_transfers, value_col=plan_value_col
            )
            if plan.empty:
                st.info("No positive-net-gain transfer found — your squad already looks efficient.")
            else:
                st.dataframe(
                    plan.rename(columns={
                        "transfer_num": "#", "sell": "Sell", "buy": "Buy", "gain": "Predicted gain",
                        "hit": "Hit cost", "net": "Net gain", "free_transfer_used": "Free transfer",
                    }),
                    use_container_width=True, hide_index=True,
                )
                st.caption(f"Bank after this plan: £{remaining_bank:.1f}m")

            st.divider()
            st.subheader("Gameweek performance comparison")
            st.caption(
                "**Model Prediction** and **FPL Prediction** are only populated once a "
                "prediction / `ep_next` snapshot existed *before* that gameweek's deadline — "
                "there's no way to retroactively reconstruct either for earlier gameweeks. "
                "GW1's Model Prediction is permanently unavailable, not just unbackfilled: the "
                "model's features need at least one prior current-season gameweek of rolling "
                "form, so GW1 is excluded from training and backtesting too, every season, not "
                "just this dashboard. **—** means unavailable, not zero. Both prediction "
                "columns are for the starting XI you actually fielded that week, not the "
                "model's own optimal lineup (see 'Recommended starting XI' above for that — the "
                "two can differ). **Overall Percentile** is your cumulative season-rank "
                "percentile as of that gameweek (FPL's own figure), not an isolated "
                "single-gameweek percentile."
            )
            comparison = load_gw_comparison(season, settings.ENTRY_ID)
            if comparison.empty:
                st.info("No gameweek history ingested yet. Run `python -m src.ingestion.load_manager` first.")
            else:
                display = comparison.copy()
                display["delta_vs_model"] = display["actual_points"] - display["model_predicted_points"]

                def _fmt_partial(points, matched: float) -> str:
                    if pd.isna(points):
                        return "—"
                    suffix = f" ({int(matched)}/11)" if pd.notna(matched) and matched < 11 else ""
                    return f"{points:.1f}{suffix}"

                display["Model Prediction"] = display.apply(
                    lambda r: _fmt_partial(r["model_predicted_points"], r["model_players_matched"]), axis=1
                )
                display["FPL Prediction"] = display.apply(
                    lambda r: _fmt_partial(r["epl_predicted_points"], r["epl_players_matched"]), axis=1
                )
                display["Δ vs Model"] = display["delta_vs_model"].apply(
                    lambda v: f"{v:+.1f}" if pd.notna(v) else "—"
                )
                display["Overall Percentile"] = display["percentile"].apply(
                    lambda v: f"Top {v:.0f}%" if pd.notna(v) else "—"
                )
                st.dataframe(
                    display[[
                        "gw_name", "Model Prediction", "FPL Prediction", "actual_points",
                        "Δ vs Model", "gw_average", "gw_highest", "Overall Percentile",
                    ]].rename(columns={
                        "gw_name": "Gameweek", "actual_points": "Points Scored",
                        "gw_average": "Gameweek Average", "gw_highest": "Gameweek Highest",
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

    st.caption(
        "**Ceiling %** = P(10+ points next GW), from a separate haul-probability classifier "
        "(ROC-AUC ~0.85 on held-out backtesting) - not derived from Predicted points. The point "
        "estimate alone is haul-blind (see notebooks/09_error_analysis.ipynb): it hedges toward "
        "the mean on big scores, so Ceiling % is what actually surfaces differential/captaincy "
        "upside, e.g. a modest point estimate with a disproportionately high ceiling. "
        "**Floor–Ceiling** is the 10th–90th percentile range from a separate quantile "
        "regression pass (`notebooks/11_monte_carlo.ipynb`), giving a real spread rather than "
        "a single number — a nailed starter and a rotation risk can share the same point "
        "estimate while having very different floors."
    )
    view_display = view.copy()
    view_display["range"] = view_display.apply(
        lambda r: f"{r['floor_points']:.1f}–{r['ceiling_points']:.1f}"
        if pd.notna(r["floor_points"]) and pd.notna(r["ceiling_points"]) else "—",
        axis=1,
    )
    st.dataframe(
        view_display.sort_values("predicted_points", ascending=False, na_position="last")[
            ["web_name", "position", "team", "price", "predicted_points", "range", "p_haul_10plus",
             "total_points", "points_per_90", "points_per_million", "selected_by_percent", "status"]
        ].rename(columns={
            "web_name": "Player", "position": "Pos", "team": "Team", "price": "Price (£m)",
            "predicted_points": "Predicted (next GW)", "range": "Floor–Ceiling", "p_haul_10plus": "Ceiling %",
            "total_points": "Total pts", "points_per_90": "Pts/90", "points_per_million": "Pts/£m",
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
    st.caption(
        f"'Next {DEFAULT_HORIZON} GWs' sums the next-GW prediction across that window, scaled "
        "week-by-week by the player's team's fixture difficulty (0 for a blank gameweek, "
        "double-counted for a double gameweek) — see src/recommendations/horizon.py."
    )
    min_predicted = st.slider("Minimum predicted points", 0.0, 10.0, 1.0, step=0.5, key="targets_min")
    max_ownership = st.slider("Maximum ownership % (differentials)", 0.0, 100.0, 100.0, step=5.0, key="targets_own")
    sort_horizon = st.checkbox(f"Sort by next {DEFAULT_HORIZON} GWs instead of pts/£m", key="targets_sort_horizon")

    targets = rankings[
        (rankings["predicted_points"] >= min_predicted) &
        (rankings["selected_by_percent"] <= max_ownership) &
        (rankings["status"] == "a")
    ].sort_values("horizon_points" if sort_horizon else "predicted_value", ascending=False)

    st.dataframe(
        targets[["web_name", "position", "team", "price", "predicted_points", "horizon_points",
                  "horizon_fixtures", "predicted_value", "selected_by_percent"]].rename(columns={
            "web_name": "Player", "position": "Pos", "team": "Team", "price": "Price (£m)",
            "predicted_points": "Predicted (next GW)", "horizon_points": f"Predicted (next {DEFAULT_HORIZON} GWs)",
            "horizon_fixtures": "Fixtures in window", "predicted_value": "Predicted pts/£m",
            "selected_by_percent": "Owned %",
        }),
        use_container_width=True, height=500,
    )

with tab_optimal:
    st.subheader("Optimal squad from scratch")
    st.caption(
        "A real integer program (PuLP/CBC), not a heuristic: picks the 15-player squad "
        "(exactly 2 GKP / 5 DEF / 5 MID / 3 FWD, max 3 per real team) that maximizes what its "
        "*starting XI* can actually score — squad and starting-XI selection are solved "
        "together, with the captain's double points included in the objective. Budget is a "
        "ceiling the solver has no reason to exhaust: bench players score nothing in the "
        "objective, so it only spends beyond the cheapest legal bench if doing so improves a "
        "starter."
    )
    budget = st.number_input("Budget (£m)", min_value=80.0, max_value=100.0, value=100.0, step=0.5)
    opt_horizon = st.radio(
        "Optimize for", ["Next GW only", f"Next {DEFAULT_HORIZON} GWs (fixture-weighted)"],
        horizontal=True, key="opt_horizon",
    )
    opt_value_col = "horizon_points" if "GWs" in opt_horizon else "predicted_points"
    opt_squad, opt_xi, opt_captain, opt_objective = solve_optimal_squad(season, budget, opt_value_col)
    opt_label = f"Predicted (next {DEFAULT_HORIZON} GWs)" if opt_value_col == "horizon_points" else "Predicted (next GW)"

    if opt_squad is None:
        st.error("No feasible squad found at this budget — try raising it.")
    else:
        st.metric(f"Projected starting XI points (incl. captain double) — {opt_label}", round(opt_objective, 2))
        display_squad = opt_squad.copy()
        display_squad["role"] = display_squad.apply(
            lambda r: "C" if r["captain"] else ("Start" if r["starting"] else "Bench"), axis=1
        )
        st.dataframe(
            display_squad.sort_values(["position", "starting"], ascending=[True, False])[
                ["web_name", "position", "team", "price", opt_value_col, "role"]
            ].rename(columns={
                "web_name": "Player", "position": "Pos", "team": "Team",
                "price": "Price (£m)", opt_value_col: opt_label, "role": "Role",
            }),
            use_container_width=True, hide_index=True, height=560,
        )
        st.caption(f"Total spend: £{opt_squad['price'].sum():.1f}m of £{budget:.1f}m budget")

        if settings.ENTRY_ID:
            actual_squad, _ = load_squad(season, settings.ENTRY_ID)
            actual_squad = actual_squad.merge(
                rankings[["player_code", "horizon_points", "horizon_fixtures"]], on="player_code", how="left"
            )
            actual_xi, _ = best_starting_xi(actual_squad, value_col=opt_value_col)
            actual_total = actual_xi[opt_value_col].sum()
            st.info(
                f"For context (not apples-to-apples — this doesn't account for the cost of "
                f"actually making these transfers): your current starting XI projects to "
                f"**{actual_total:.2f}** points (uncaptained, {opt_label.lower()}); this "
                f"from-scratch squad at the same £{budget:.1f}m budget projects to "
                f"**{opt_objective:.2f}** (captained)."
            )

with tab_model_lab:
    st.subheader("Model Lab")
    st.caption(
        "Exploratory only. Toggle which input features go into Linear Regression / Random "
        "Forest / XGBoost, retrain on the exact same 2020-21..2024-25 train / 2025-26 held-out "
        "split the deployed model uses, and see the real effect on accuracy. Every run logs a "
        "row to `model_versions` (`is_experiment=true`, no artifact saved) so nothing here can "
        "ever be picked up as the live model, but the record persists across sessions."
    )

    lab_df, all_features = load_model_lab_frame()
    feature_groups = group_features(all_features)

    with st.expander("Feature correlation map", expanded=False):
        numeric_features = [c for c in all_features if not c.startswith("pos_")]
        corr = lab_df[numeric_features].corr()
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(numeric_features)))
        ax.set_xticklabels(numeric_features, rotation=90, fontsize=6)
        ax.set_yticks(range(len(numeric_features)))
        ax.set_yticklabels(numeric_features, fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
        st.pyplot(fig)
        st.caption(
            "Feature-to-feature correlation, computed once on the cached training frame - a "
            "map of redundancy worth a glance before deciding what to toggle below."
        )

    st.markdown("**Select features**")
    selected: list[str] = []
    group_cols_layout = st.columns(2)
    for i, (group_name, cols) in enumerate(feature_groups.items()):
        with group_cols_layout[i % 2]:
            with st.expander(f"{group_name} ({len(cols)})", expanded=(group_name == "Fixture & opponent strength")):
                btn_a, btn_b = st.columns(2)
                if btn_a.button("All", key=f"ml_all_{group_name}"):
                    for c in cols:
                        st.session_state[f"ml_feat_{c}"] = True
                if btn_b.button("None", key=f"ml_none_{group_name}"):
                    for c in cols:
                        st.session_state[f"ml_feat_{c}"] = False
                for c in cols:
                    checked = st.checkbox(c, value=st.session_state.get(f"ml_feat_{c}", True), key=f"ml_feat_{c}")
                    if checked:
                        selected.append(c)

    st.caption(f"**{len(selected)} of {len(all_features)}** features selected")

    opt_a, opt_b = st.columns(2)
    fast_preview = opt_a.checkbox("Fast preview (25% training sample)", value=False, key="ml_fast_preview")
    extended = opt_b.checkbox(
        "Compute extended diagnostics (OLS p-values / permutation importance — slower)",
        value=False, key="ml_extended",
    )

    run_clicked = st.button("Run experiment", type="primary", disabled=(len(selected) == 0))
    if len(selected) == 0:
        st.warning("Select at least one feature to run an experiment.")

    if run_clicked:
        with st.spinner(f"Training on {len(selected)} features..."):
            results = evaluate_feature_subset(
                lab_df, selected, sample_frac=0.25 if fast_preview else None, compute_extended=extended,
            )
            baseline_results = load_deployed_baseline_results(lab_df, all_features)

        y_test = results["_y_test"]

        p_values_by_model: dict[str, float] = {}
        for model_type in MODEL_TYPES:
            if model_type == "baseline":
                continue
            p_values_by_model[model_type] = paired_bootstrap_p_value(
                y_test, baseline_results[model_type]["y_test_pred"], results[model_type]["y_test_pred"],
            )

        conn = get_connection()
        with conn.cursor() as cur:
            for model_type in MODEL_TYPES:
                res = results[model_type]
                tm = res["test_metrics"]
                diagnostics = {
                    "train_metrics": res["train_metrics"],
                    "overfit_gap": res["overfit_gap"],
                    "r2_ci_95": list(res["r2_ci"]),
                    "fast_preview": fast_preview,
                }
                if model_type in p_values_by_model:
                    diagnostics["p_value_vs_session_baseline"] = p_values_by_model[model_type]
                if res["extended"] is not None:
                    diagnostics["extended"] = res["extended"]
                cur.execute(
                    """
                    INSERT INTO model_versions (
                        model_type, target, training_seasons, test_season, features,
                        hyperparameters, mae, rmse, r2, artifact_path, is_experiment, diagnostics
                    ) VALUES (%s, 'total_points_direct', %s, %s, %s, NULL, %s, %s, %s, NULL, true, %s)
                    """,
                    (
                        model_type, TRAIN_SEASONS, TEST_SEASON, json.dumps(selected),
                        tm["mae"], tm["rmse"], tm["r2"], json.dumps(diagnostics),
                    ),
                )
        conn.commit()
        conn.close()
        st.success(f"Logged {len(MODEL_TYPES)} rows to model_versions (is_experiment=true).")

        rows = []
        for model_type in MODEL_TYPES:
            res = results[model_type]
            tm = res["test_metrics"]
            base_r2 = baseline_results[model_type]["test_metrics"]["r2"]
            fit_flag = "—"
            if model_type != "baseline":
                if res["overfit_gap"] is not None and res["overfit_gap"] > 0.15:
                    fit_flag = "⚠️ possible overfit"
                elif tm["r2"] <= results["baseline"]["test_metrics"]["r2"]:
                    fit_flag = "⚠️ possible underfit"
                else:
                    fit_flag = "OK"
            rows.append({
                "Model": model_type,
                "Test R²": tm["r2"],
                "Δ vs full feature set": round(tm["r2"] - base_r2, 4) if model_type != "baseline" else None,
                "Test MAE": tm["mae"],
                "Test RMSE": tm["rmse"],
                "Train R²": res["train_metrics"]["r2"] if res["train_metrics"] else None,
                "Overfit gap": res["overfit_gap"],
                "Fit": fit_flag,
                "95% CI (R²)": f"[{res['r2_ci'][0]:.3f}, {res['r2_ci'][1]:.3f}]",
                "p vs. session baseline": (
                    round(p_values_by_model[model_type], 4) if model_type in p_values_by_model else None
                ),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "Δ and p-value are both vs. a fixed 'all features' reference run computed once per "
            "session — not an older ledger row. Fit flags are rules of thumb (overfit gap > "
            "0.15 R², or underfit = at/below the naive baseline), not a formal test."
        )

        importance_cols = st.columns(2)
        for idx, model_type in enumerate(("random_forest", "xgboost")):
            model = results[model_type]["model"]
            with importance_cols[idx]:
                st.markdown(f"**{model_type} feature importance**")
                imp = pd.Series(model.feature_importances_, index=selected).sort_values(ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.barh(imp.index[::-1], imp.values[::-1])
                ax.set_xlabel("Importance")
                st.pyplot(fig)

        if extended:
            st.markdown("**Extended diagnostics**")
            ols = results["linear_regression"]["extended"]
            if ols:
                st.caption(f"OLS overall F-test p-value: {ols['f_pvalue']:.2e}")
                ols_df = pd.DataFrame([
                    {"feature": name, **vals} for name, vals in ols["coefficients"].items() if name != "const"
                ]).sort_values("p_value")
                st.dataframe(ols_df, use_container_width=True, hide_index=True, height=300)
            for model_type in ("random_forest", "xgboost"):
                perm = results[model_type]["extended"]
                if perm:
                    st.markdown(f"**{model_type} permutation importance (mean ± std)**")
                    perm_df = pd.DataFrame([
                        {"feature": name, **vals} for name, vals in perm.items()
                    ]).sort_values("importance_mean", ascending=False)
                    st.dataframe(perm_df, use_container_width=True, hide_index=True, height=300)

    st.markdown("**Experiment ledger**")
    ledger = pd.read_sql_query(
        """
        SELECT model_type, features, mae, rmse, r2, diagnostics, trained_at
        FROM model_versions
        WHERE is_experiment AND target = 'total_points_direct'
        ORDER BY trained_at DESC
        LIMIT 50
        """,
        get_engine(),
    )
    if ledger.empty:
        st.caption("No experiments logged yet — run one above.")
    else:
        ledger["feature_count"] = ledger["features"].apply(len)
        ledger["overfit_gap"] = ledger["diagnostics"].apply(
            lambda d: d.get("overfit_gap") if isinstance(d, dict) else None
        )
        st.dataframe(
            ledger[["trained_at", "model_type", "feature_count", "r2", "mae", "rmse", "overfit_gap"]],
            use_container_width=True, hide_index=True, height=400,
        )
        st.caption(
            "Every past Model Lab run, this session or an earlier one — queryable directly via "
            "`SELECT * FROM model_versions WHERE is_experiment`."
        )

st.divider()
st.caption(
    "Predictions come from a Random Forest model trained on 3 historical seasons; see "
    "notebooks/06_model_comparison.ipynb for backtested accuracy. Squad optimization / "
    "auto-substitution and full transfer-in-out comparisons are Phase 5 (decision engine), "
    "not built yet."
)
