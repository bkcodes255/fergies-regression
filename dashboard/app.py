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
import streamlit.components.v1 as components

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
def load_fixture_grid(season: str, start_gw: int, horizon: int) -> pd.DataFrame:
    """Every team's fixture(s) across [start_gw, start_gw + horizon) - not just the immediate
    next one (v_fixture_difficulty already carries one row per upcoming fixture; the Fixtures
    tab used to throw all but the earliest away via DISTINCT ON, which is why it could only
    ever show one gameweek). A team with two fixtures in one event_id (a double gameweek)
    naturally produces two rows here; a team with none (a blank) produces zero - the caller
    handles both."""
    engine = get_engine()
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
        engine, params={"season": season, "start_gw": start_gw, "horizon": horizon},
    )


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


@st.cache_data(ttl=300)
def load_next_deadline(season: str) -> dict | None:
    df = pd.read_sql_query(
        """
        SELECT event_id, name, deadline_time FROM gameweeks
        WHERE season = %(season)s AND NOT finished AND deadline_time > now()
        ORDER BY deadline_time ASC LIMIT 1
        """,
        get_engine(), params={"season": season},
    )
    if df.empty:
        return None
    row = df.iloc[0]
    return {"event_id": int(row["event_id"]), "name": row["name"], "deadline_time": row["deadline_time"]}


def format_time_left(deadline_time) -> str:
    delta = deadline_time - pd.Timestamp.now(tz="UTC")
    total_seconds = delta.total_seconds()
    if total_seconds <= 0:
        return "deadline passed"
    days, rem = divmod(int(total_seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days > 0:
        return f"{days}d {hours}h"
    return f"{hours}h {minutes}m"


# Design tokens - lifted directly from .streamlit/config.toml, not invented: this is the
# app's real dark theme (background/card/border/text), and the position/status colors below
# are chosen from its own chartCategoricalColors / semantic-colors sets so nothing here
# introduces a palette Streamlit's own chrome doesn't already use. Position colors deliberately
# avoid green/orange/red - those are reserved for status (good/warning) and the captain badge -
# so a position accent never collides with a status ring around the same player.
BG = "#0F1420"
CARD_BG = "#161B28"
CARD_BORDER = "#232838"
TEXT_PRIMARY = "#E6EDF3"
TEXT_MUTED = "#8B94A8"
CAPTAIN_BADGE = "#E11D2E"  # brand primary - the one thing on a pitch worth a bold color
STATUS_GOOD = "#22C55E"
STATUS_WARNING = "#F59E0B"
POSITION_ACCENT = {"GKP": "#8B94A8", "DEF": "#3B82F6", "MID": "#A855F7", "FWD": "#06B6D4"}


def captain_recommendation(starting: pd.DataFrame) -> dict | None:
    """Compares the model's top-predicted starter against the currently-armbanded captain -
    None if either side can't be determined (no predictions yet, or no captain flagged)."""
    if starting.empty or starting["predicted_points"].isna().all():
        return None
    current = starting[starting["is_captain"]]
    if current.empty:
        return None
    best_row = starting.loc[starting["predicted_points"].idxmax()]
    current_row = current.iloc[0]
    return {
        "matches": best_row["player_code"] == current_row["player_code"],
        "recommended_code": best_row["player_code"], "recommended_name": best_row["web_name"],
        "recommended_points": best_row["predicted_points"],
        "current_name": current_row["web_name"], "current_points": current_row["predicted_points"],
    }


def lineup_changes(squad: pd.DataFrame) -> dict:
    """Diffs the actual starting XI against best_starting_xi's recommendation by player_code -
    empty bring_in/bench_out means the fielded lineup is already optimal."""
    starting_codes = set(squad.loc[squad["multiplier"] > 0, "player_code"])
    optimal_xi, formation = best_starting_xi(squad)
    optimal_codes = set(optimal_xi["player_code"])
    return {
        "bring_in_codes": optimal_codes - starting_codes,
        "bench_out_codes": starting_codes - optimal_codes,
        "optimal_points": optimal_xi["predicted_points"].sum(),
        "formation": formation,
    }


def render_stat_tile_html(label: str, value: str, sublabel: str | None = None, accent: str = TEXT_PRIMARY) -> str:
    sub = f'<div style="font-size:12px;font-weight:500;color:{TEXT_MUTED};">{sublabel}</div>' if sublabel else ""
    return f"""
    <div style="background:{CARD_BG};border:1px solid {CARD_BORDER};border-radius:16px;padding:18px 20px;
                display:flex;flex-direction:column;gap:6px;height:100%;box-sizing:border-box;justify-content:center;">
      <div style="font-size:12px;font-weight:500;color:{TEXT_MUTED};letter-spacing:0.02em;">{label}</div>
      <div style="font-size:28px;font-weight:700;color:{accent};line-height:1.1;">{value}</div>
      {sub}
    </div>
    """


def render_action_tile_html(status: str, headline: str, subtext: str) -> str:
    """status: 'good' (green check) or 'attention' (red chevron, brand primary - reserved for
    the one thing on a screen actually worth acting on, per the design canvas mockup)."""
    is_good = status == "good"
    accent = STATUS_GOOD if is_good else CAPTAIN_BADGE
    icon = (
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0F1420" stroke-width="3" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l4 4L19 7"/></svg>'
        if is_good else
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg>'
    )
    return f"""
    <div style="background:{CARD_BG};border:1px solid {CARD_BORDER};border-left:3px solid {accent};
                border-radius:16px;padding:20px 22px;display:flex;gap:14px;align-items:flex-start;
                height:100%;box-sizing:border-box;">
      <div style="flex:0 0 auto;width:32px;height:32px;border-radius:50%;background:{accent};
                  display:flex;align-items:center;justify-content:center;">{icon}</div>
      <div style="display:flex;flex-direction:column;gap:4px;min-width:0;">
        <div style="font-size:15px;font-weight:600;color:{TEXT_PRIMARY};">{headline}</div>
        <div style="font-size:13px;color:{TEXT_MUTED};line-height:1.4;">{subtext}</div>
      </div>
    </div>
    """


def _player_badge_html(name: str, points, position: str, size: int, is_captain: bool = False,
                        is_vice: bool = False, swap_flag: str | None = None) -> str:
    accent = POSITION_ACCENT.get(position, TEXT_MUTED)
    points_str = f"{points:.1f} pts" if pd.notna(points) else "— pts"
    ring = ""
    footer = ""
    if swap_flag == "start":
        ring = f"box-shadow:0 0 0 3px {STATUS_GOOD};"
        footer = f'<div style="font-size:10px;color:{STATUS_GOOD};font-weight:700;">&#9650; consider starting</div>'
    elif swap_flag == "bench":
        ring = f"box-shadow:0 0 0 3px {STATUS_WARNING};"
        footer = f'<div style="font-size:10px;color:{STATUS_WARNING};font-weight:700;">&#9660; consider bench</div>'

    badge = ""
    if is_captain:
        badge = (f'<div style="position:absolute;top:-6px;right:-6px;width:20px;height:20px;'
                  f'border-radius:50%;background:{CAPTAIN_BADGE};color:#fff;font-size:11px;'
                  f'font-weight:800;display:flex;align-items:center;justify-content:center;'
                  f'border:2px solid {BG};">C</div>')
    elif is_vice:
        badge = (f'<div style="position:absolute;top:-6px;right:-6px;width:20px;height:20px;'
                  f'border-radius:50%;background:{TEXT_MUTED};color:{BG};font-size:11px;'
                  f'font-weight:800;display:flex;align-items:center;justify-content:center;'
                  f'border:2px solid {BG};">V</div>')

    return f"""
    <div style="text-align:center;width:{max(size + 34, 76)}px;">
      <div style="position:relative;width:{size}px;height:{size}px;margin:0 auto;">
        <div style="width:{size}px;height:{size}px;border-radius:50%;background:{CARD_BG};
                    border:3px solid {accent};{ring}"></div>
        {badge}
      </div>
      <div style="margin-top:4px;background:rgba(15,20,32,0.85);border-radius:8px;padding:3px 6px;">
        <div style="color:{TEXT_PRIMARY};font-size:11px;font-weight:600;white-space:nowrap;overflow:hidden;
                    text-overflow:ellipsis;">{name}</div>
        <div style="color:{TEXT_PRIMARY};font-size:11px;font-weight:700;">{points_str}</div>
      </div>
      {footer}
    </div>
    """


def render_pitch_html(starting: pd.DataFrame, bench_out_codes: set, value_col: str = "predicted_points") -> str:
    """An FPL-style pitch view of a starting XI (no external kit/crest images - those would
    need FPL's own CDN, unnecessary and unavailable here). Position = a fixed accent color;
    the captain gets a red armband badge; a lineup swap the model would make is called out
    with BOTH a status-colored ring and a text label - never color alone, per the dataviz
    skill's accessibility rule. value_col lets callers reuse this for either the next-GW
    prediction or the fixture-weighted horizon total."""
    row_top_pct = {"FWD": 8, "MID": 34, "DEF": 60, "GKP": 84}
    cards = []
    for pos in ("FWD", "MID", "DEF", "GKP"):
        row_players = starting[starting["position"] == pos].reset_index(drop=True)
        n = len(row_players)
        for i, player in row_players.iterrows():
            left_pct = (i + 0.5) / n * 100 if n else 50
            swap_flag = "bench" if player["player_code"] in bench_out_codes else None
            card = _player_badge_html(
                player["web_name"], player.get(value_col), player["position"], size=48,
                is_captain=bool(player.get("is_captain", False)), is_vice=bool(player.get("is_vice_captain", False)),
                swap_flag=swap_flag,
            )
            cards.append(
                f'<div style="position:absolute;top:{row_top_pct[pos]}%;left:{left_pct}%;'
                f'transform:translate(-50%,-50%);">{card}</div>'
            )
    return f"""
    <div style="position:relative;width:100%;height:480px;border-radius:16px;overflow:hidden;
                background:repeating-linear-gradient(180deg,#1f9d4a 0,#1f9d4a 48px,#1c8f43 48px,#1c8f43 96px);">
      <div style="position:absolute;left:0;right:0;top:50%;height:2px;background:rgba(255,255,255,0.45);"></div>
      <div style="position:absolute;left:50%;top:50%;width:100px;height:100px;
                  border:2px solid rgba(255,255,255,0.45);border-radius:50%;transform:translate(-50%,-50%);"></div>
      <div style="position:absolute;left:50%;top:0;width:180px;height:36px;
                  border:2px solid rgba(255,255,255,0.45);border-top:none;transform:translateX(-50%);"></div>
      <div style="position:absolute;left:50%;bottom:0;width:180px;height:36px;
                  border:2px solid rgba(255,255,255,0.45);border-bottom:none;transform:translateX(-50%);"></div>
      {''.join(cards)}
    </div>
    """


def render_bench_html(bench: pd.DataFrame, bring_in_codes: set, value_col: str = "predicted_points") -> str:
    items = []
    for _, player in bench.iterrows():
        swap_flag = "start" if player["player_code"] in bring_in_codes else None
        items.append(_player_badge_html(
            player["web_name"], player.get(value_col), player["position"], size=38,
            is_vice=bool(player.get("is_vice_captain", False)), swap_flag=swap_flag,
        ))
    return f"""
    <div style="display:flex;gap:16px;align-items:flex-start;justify-content:center;
                padding:16px 10px;background:{CARD_BG};border:1px solid {CARD_BORDER};
                border-radius:12px;margin-top:12px;">
      <div style="font-size:11px;color:{TEXT_MUTED};font-weight:700;align-self:center;">BENCH</div>
      {''.join(items)}
    </div>
    """


def render_transfer_card_html(sell_row: pd.Series, buy_row: pd.Series, value_col: str = "predicted_points") -> str:
    def _card(player: pd.Series, accent: str) -> str:
        return f"""
        <div style="text-align:center;width:120px;">
          <div style="width:56px;height:56px;margin:0 auto;border-radius:50%;background:{CARD_BG};
                      border:3px solid {accent};"></div>
          <div style="margin-top:6px;font-weight:700;font-size:13px;color:{TEXT_PRIMARY};">{player['web_name']}</div>
          <div style="font-size:11px;color:{TEXT_MUTED};">£{player['price']:.1f}m</div>
          <div style="font-size:13px;font-weight:700;color:{accent};margin-top:2px;">
            {player[value_col]:.1f} pts
          </div>
        </div>
        """
    return f"""
    <div style="display:flex;align-items:center;justify-content:center;gap:18px;padding:12px 0;">
      {_card(sell_row, TEXT_MUTED)}
      <div style="font-size:22px;color:{STATUS_GOOD};">&#8594;</div>
      {_card(buy_row, STATUS_GOOD)}
    </div>
    """


def render_player_row_html(name: str, team: str, price: float, position: str, predicted: float,
                            floor: float, ceiling: float, ceiling_pct: float) -> str:
    """One row of the Players tab: a predicted-points bar and a floor-ceiling range bar sharing
    one 0-12pt scale, so rows compare at a glance without reading twelve columns of numbers.
    The range bar's right edge (not its width) is what gets clamped to the scale - clamping
    width independently let a star player's ceiling (this model runs to 20+ pts) push the bar
    past the track's right edge with nothing to stop it."""
    scale_max = 12.0

    def pct(v: float) -> float:
        return max(0.0, min(100.0, (v / scale_max) * 100.0))

    predicted_pct = pct(predicted)
    range_left = pct(floor)
    range_right = pct(ceiling)
    range_width = max(0.0, range_right - range_left)
    ceiling_color = STATUS_GOOD if ceiling_pct >= 20 else TEXT_PRIMARY
    return f"""
    <div style="display:grid;grid-template-columns:230px 1fr 200px 92px;gap:20px;align-items:center;
                background:{CARD_BG};border:1px solid {CARD_BORDER};border-radius:12px;padding:14px 18px;">
      <div style="display:flex;align-items:center;gap:12px;min-width:0;">
        <div style="flex:0 0 auto;width:10px;height:10px;border-radius:50%;background:{POSITION_ACCENT.get(position, TEXT_MUTED)};"></div>
        <div style="min-width:0;">
          <div style="font-size:14px;font-weight:600;color:{TEXT_PRIMARY};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name}</div>
          <div style="font-size:12px;color:{TEXT_MUTED};">{team} · £{price:.1f}m</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:12px;">
        <div style="flex:1;height:8px;border-radius:999px;background:{BG};overflow:hidden;">
          <div style="width:{predicted_pct}%;height:100%;border-radius:999px;background:{POSITION_ACCENT.get(position, TEXT_MUTED)};"></div>
        </div>
        <div style="font-size:15px;font-weight:700;color:{TEXT_PRIMARY};width:38px;text-align:right;">{predicted:.1f}</div>
      </div>
      <div>
        <div style="position:relative;height:8px;border-radius:999px;background:{BG};overflow:hidden;">
          <div style="position:absolute;left:{range_left}%;width:{range_width}%;top:0;height:100%;border-radius:999px;background:{CARD_BORDER};"></div>
          <div style="position:absolute;left:{predicted_pct}%;top:-3px;width:3px;height:14px;border-radius:2px;background:{TEXT_MUTED};"></div>
        </div>
        <div style="font-size:11px;color:{TEXT_MUTED};margin-top:5px;">{floor:.1f} – {ceiling:.1f} pts</div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:15px;font-weight:700;color:{ceiling_color};">{ceiling_pct:.0f}%</div>
        <div style="font-size:11px;color:{TEXT_MUTED};">ceiling</div>
      </div>
    </div>
    """


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

tab_names = ["Players", "Fixtures", "Optimal Squad", "Model Lab"]
if settings.ENTRY_ID:
    tab_names = ["This Week"] + tab_names[:2] + ["Season"] + tab_names[2:]
tabs = st.tabs(tab_names)
tab_lookup = dict(zip(tab_names, tabs))
tab_players = tab_lookup["Players"]
tab_fixtures = tab_lookup["Fixtures"]
tab_optimal = tab_lookup["Optimal Squad"]
tab_model_lab = tab_lookup["Model Lab"]

if settings.ENTRY_ID:
    with tab_lookup["This Week"]:
        squad, manager_gw = load_squad(season, settings.ENTRY_ID)
        deadline = load_next_deadline(season)

        if squad.empty:
            st.warning("No squad data ingested yet. Run `python -m src.ingestion.load_manager` first.")
        else:
            starting = squad[squad["multiplier"] > 0].copy()
            bench = squad[squad["multiplier"] == 0].copy()

            header_cols = st.columns([2, 1, 1])
            if deadline is not None:
                header_cols[0].markdown(
                    f"<div style='font-size:12px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;"
                    f"color:{TEXT_MUTED};margin-bottom:6px;'>{deadline['name']} deadline</div>"
                    f"<div style='font-size:44px;font-weight:800;line-height:1;color:{TEXT_PRIMARY};'>"
                    f"{format_time_left(deadline['deadline_time'])}</div>",
                    unsafe_allow_html=True,
                )
            else:
                header_cols[0].markdown("### No upcoming deadline found")
            gw_row = manager_gw.iloc[0] if not manager_gw.empty else None
            free_transfers = compute_free_transfers(manager_gw)
            bank = float(gw_row["bank"]) / 10 if gw_row is not None else 0.0
            with header_cols[1]:
                components.html(render_stat_tile_html("Free transfers", str(free_transfers)), height=96)
            with header_cols[2]:
                components.html(render_stat_tile_html("Bank", f"£{bank:.1f}m"), height=96)

            cap = captain_recommendation(starting)
            changes = lineup_changes(squad)

            st.markdown("#### Your team")
            components.html(render_pitch_html(starting, changes["bench_out_codes"]), height=500)
            if not bench.empty:
                components.html(render_bench_html(bench, changes["bring_in_codes"]), height=120)

            action_cols = st.columns(2)
            with action_cols[0]:
                if cap is None:
                    tile = render_action_tile_html(
                        "good", "Captain recommendation unavailable", "No prediction for this gameweek yet."
                    )
                elif cap["matches"]:
                    tile = render_action_tile_html(
                        "good", f"Captain: {cap['current_name']}", "Already the model's top pick in your XI."
                    )
                else:
                    tile = render_action_tile_html(
                        "attention", f"Switch captain to {cap['recommended_name']}",
                        f"{cap['recommended_points']:.1f} predicted vs. {cap['current_points']:.1f} for "
                        f"{cap['current_name']}, currently armbanded.",
                    )
                components.html(tile, height=110)
            with action_cols[1]:
                if not changes["bring_in_codes"] and not changes["bench_out_codes"]:
                    tile = render_action_tile_html(
                        "good", "Lineup is optimal", "No swap improves this formation and squad."
                    )
                else:
                    gain = changes["optimal_points"] - starting["predicted_points"].sum()
                    tile = render_action_tile_html(
                        "attention", f"+{gain:.1f} pts available in your lineup",
                        "See the highlighted swap on the pitch above.",
                    )
                components.html(tile, height=110)

            st.divider()
            st.markdown("#### Recommended transfer")
            plan, _, remaining_bank = suggest_transfer_plan(squad, rankings, bank, free_transfers)
            if plan.empty:
                components.html(
                    render_action_tile_html("good", "No transfer recommended", "Your squad already looks efficient."),
                    height=110,
                )
            else:
                top = plan.iloc[0]
                sell_row = squad[squad["web_name"] == top["sell"]].iloc[0]
                buy_row = rankings[rankings["web_name"] == top["buy"]].iloc[0]
                components.html(render_transfer_card_html(sell_row, buy_row), height=140)
                components.html(
                    render_stat_tile_html("Net gain, after any hit", f"{top['net']:+.1f} pts", accent=STATUS_GOOD),
                    height=96,
                )

            with st.expander("Full coordinated transfer plan"):
                st.caption(
                    f"You have **{free_transfers}** free transfer(s) available. Coordinated, not "
                    "independent suggestions — each transfer accounts for the ones before it. Greedy: "
                    "picks the single best transfer at each step, stops once the next one wouldn't "
                    "survive its hit cost."
                )
                plan_horizon = st.radio(
                    "Judge transfers by", ["Next GW only", f"Next {DEFAULT_HORIZON} GWs (fixture-weighted)"],
                    horizontal=True, key="plan_horizon",
                )
                plan_value_col = "horizon_points" if "GWs" in plan_horizon else "predicted_points"
                full_plan, _, plan_remaining_bank = suggest_transfer_plan(
                    squad, rankings, bank, free_transfers, value_col=plan_value_col
                )
                if full_plan.empty:
                    st.info("No positive-net-gain transfer found at this horizon.")
                else:
                    st.dataframe(
                        full_plan.rename(columns={
                            "transfer_num": "#", "sell": "Sell", "buy": "Buy", "gain": "Predicted gain",
                            "hit": "Hit cost", "net": "Net gain", "free_transfer_used": "Free transfer",
                        }),
                        use_container_width=True, hide_index=True,
                    )
                    st.caption(f"Bank after this plan: £{plan_remaining_bank:.1f}m")

            st.divider()
            st.markdown("#### Range of outcomes")
            sim_totals = simulate_squad(starting, n_samples=10000)
            sim_stats = summarize_simulation(sim_totals)
            range_tiles = "".join(
                f'<div style="flex:1;">{render_stat_tile_html(label, value, accent=accent)}</div>'
                for label, value, accent in [
                    ("Likely (median)", f"{sim_stats['p50']:.0f}", TEXT_PRIMARY),
                    ("Bad week (p10)", f"{sim_stats['p10']:.0f}", TEXT_PRIMARY),
                    ("Great week (p90)", f"{sim_stats['p90']:.0f}", STATUS_GOOD),
                    ("P(80+ pts)", f"{sim_stats['p_beat_80']:.0%}", TEXT_PRIMARY),
                ]
            )
            components.html(f'<div style="display:flex;gap:16px;">{range_tiles}</div>', height=118)

            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)
            ax.hist(sim_totals, bins=50, color="#3B82F6", alpha=0.85, edgecolor="none")
            for pct, label, color in [(10, "p10", TEXT_MUTED), (50, "median", TEXT_PRIMARY), (90, "p90", TEXT_MUTED)]:
                x = np.percentile(sim_totals, pct)
                ax.axvline(x, color=color, linestyle="--", linewidth=1)
                ax.text(x, ax.get_ylim()[1] * 0.95, label, color=color, fontsize=9, ha="center")
            ax.set_xlabel("Simulated starting XI points (captain doubled)", color=TEXT_MUTED)
            ax.tick_params(colors=TEXT_MUTED)
            ax.set_yticks([])
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color(CARD_BORDER)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("See the **Season** tab for gameweek-by-gameweek history against this model's own predictions.")

if settings.ENTRY_ID:
    with tab_lookup["Season"]:
        _, manager_gw = load_squad(season, settings.ENTRY_ID)
        comparison = load_gw_comparison(season, settings.ENTRY_ID)

        if manager_gw.empty:
            st.warning("No squad data ingested yet. Run `python -m src.ingestion.load_manager` first.")
        else:
            latest = manager_gw.sort_values("event_id").iloc[-1]
            best_gw = manager_gw.loc[manager_gw["points"].idxmax()]

            stat_cols = st.columns(4)
            with stat_cols[0]:
                components.html(render_stat_tile_html(
                    "Total points", str(int(latest["total_points"])),
                    sublabel=f"GW average {manager_gw['points'].mean():.1f}",
                ), height=118)
            with stat_cols[1]:
                rank_str = f"{int(latest['overall_rank']):,}" if pd.notna(latest["overall_rank"]) else "—"
                components.html(render_stat_tile_html("Overall rank", rank_str), height=118)
            with stat_cols[2]:
                components.html(render_stat_tile_html(
                    "Best gameweek", str(int(best_gw["points"])), sublabel=f"GW{int(best_gw['event_id'])}",
                    accent=STATUS_GOOD,
                ), height=118)
            with stat_cols[3]:
                components.html(render_stat_tile_html(
                    "Points left on bench", str(int(manager_gw["points_on_bench"].sum())),
                    sublabel=f"Across {len(manager_gw)} gameweek(s)", accent=STATUS_WARNING,
                ), height=118)

            st.markdown("#### Model accuracy — predicted vs. what actually happened")
            st.caption(
                "Model/FPL prediction lines are only populated once a prediction existed *before* "
                "that gameweek's deadline — GW1 is permanently unavailable for either (the model's "
                "features need at least one prior gameweek of rolling form). Gaps mean unavailable, "
                "not zero. Both are for the starting XI actually fielded that week."
            )
            if comparison.empty:
                st.info("No gameweek history ingested yet.")
            else:
                fig, ax = plt.subplots(figsize=(9, 3.5))
                fig.patch.set_facecolor(BG)
                ax.set_facecolor(BG)
                ax.plot(comparison["gw_name"], comparison["actual_points"], color=CAPTAIN_BADGE,
                        linewidth=2.5, marker="o", markersize=4, label="Actual")
                ax.plot(comparison["gw_name"], comparison["model_predicted_points"], color="#3B82F6",
                        linewidth=2, marker="o", markersize=3, label="This model")
                ax.plot(comparison["gw_name"], comparison["epl_predicted_points"], color=TEXT_MUTED,
                        linewidth=2, linestyle="--", marker="o", markersize=3, label="FPL's own")
                ax.set_ylabel("Points", color=TEXT_MUTED)
                ax.tick_params(colors=TEXT_MUTED)
                for spine in ax.spines.values():
                    spine.set_color(CARD_BORDER)
                legend = ax.legend(facecolor=CARD_BG, edgecolor=CARD_BORDER, labelcolor=TEXT_PRIMARY, fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)

                valid = comparison.dropna(subset=["model_predicted_points", "epl_predicted_points"]).copy()
                insight_cols = st.columns(2)
                if valid.empty:
                    with insight_cols[0]:
                        components.html(render_action_tile_html(
                            "good", "Not enough gameweeks yet",
                            "Model-accuracy comparisons need at least one gameweek with a pre-deadline prediction.",
                        ), height=110)
                else:
                    model_err = (valid["actual_points"] - valid["model_predicted_points"]).abs()
                    epl_err = (valid["actual_points"] - valid["epl_predicted_points"]).abs()
                    beats = int((model_err < epl_err).sum())
                    with insight_cols[0]:
                        components.html(render_action_tile_html(
                            "good" if model_err.mean() < epl_err.mean() else "attention",
                            f"Beating FPL's own projection {beats} of {len(valid)} weeks",
                            f"Mean error {model_err.mean():.1f} pts/GW vs. their {epl_err.mean():.1f}.",
                        ), height=110)
                    with insight_cols[1]:
                        miss = valid.loc[(valid["actual_points"] - valid["model_predicted_points"]).idxmax()]
                        miss_gap = miss["actual_points"] - miss["model_predicted_points"]
                        components.html(render_action_tile_html(
                            "attention", f"{miss['gw_name']}: biggest miss this season",
                            f"Predicted {miss['model_predicted_points']:.0f}, actual {miss['actual_points']:.0f} "
                            f"({miss_gap:+.0f}) — the model hedges toward the mean by design.",
                        ), height=110)

            with st.expander("Full gameweek table"):
                if comparison.empty:
                    st.info("No gameweek history ingested yet.")
                else:
                    display = comparison.copy()
                    display["Δ vs Model"] = (display["actual_points"] - display["model_predicted_points"]).apply(
                        lambda v: f"{v:+.1f}" if pd.notna(v) else "—"
                    )
                    display["Overall Percentile"] = display["percentile"].apply(
                        lambda v: f"Top {v:.0f}%" if pd.notna(v) else "—"
                    )
                    st.dataframe(
                        display[[
                            "gw_name", "model_predicted_points", "epl_predicted_points", "actual_points",
                            "Δ vs Model", "gw_average", "gw_highest", "Overall Percentile",
                        ]].rename(columns={
                            "gw_name": "Gameweek", "model_predicted_points": "Model Prediction",
                            "epl_predicted_points": "FPL Prediction", "actual_points": "Points Scored",
                            "gw_average": "Gameweek Average", "gw_highest": "Gameweek Highest",
                        }),
                        use_container_width=True, hide_index=True,
                    )

with tab_players:
    available = rankings[rankings["status"] == "a"]
    st.markdown(
        f"<div style='font-size:12px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;"
        f"color:{TEXT_MUTED};margin-bottom:6px;'>Players</div>"
        f"<div style='font-size:32px;font-weight:800;line-height:1;color:{TEXT_PRIMARY};'>"
        f"{len(available)} available</div>",
        unsafe_allow_html=True,
    )

    filter_cols = st.columns([1, 1, 1, 1, 1])
    pos_filter = filter_cols[0].selectbox(
        "Position", ["All"] + sorted(rankings["position"].dropna().unique().tolist()), key="players_pos"
    )
    sort_choice = filter_cols[1].selectbox(
        "Sort by", ["Predicted (next GW)", f"Predicted (next {DEFAULT_HORIZON} GWs)", "Value per £m", "Ceiling %"],
        key="players_sort",
    )
    min_predicted = filter_cols[2].slider("Min. predicted pts", 0.0, 10.0, 0.0, step=0.5, key="players_min_pred")
    differentials_only = filter_cols[3].checkbox("Differentials (<10% owned)", key="players_diff")
    available_only = filter_cols[4].checkbox("Available only", value=True, key="players_avail")

    view = rankings.copy()
    if pos_filter != "All":
        view = view[view["position"] == pos_filter]
    view = view[view["predicted_points"] >= min_predicted]
    if differentials_only:
        view = view[view["selected_by_percent"] < 10]
    if available_only:
        view = view[view["status"] == "a"]

    sort_col = {
        "Predicted (next GW)": "predicted_points", f"Predicted (next {DEFAULT_HORIZON} GWs)": "horizon_points",
        "Value per £m": "predicted_value", "Ceiling %": "p_haul_10plus",
    }[sort_choice]
    view = view.sort_values(sort_col, ascending=False, na_position="last")

    top_rows = view.head(25)
    if top_rows.empty:
        st.info("No players match these filters.")
    else:
        rows_html = "".join(
            render_player_row_html(
                r["web_name"], r["team"], r["price"], r["position"],
                r["predicted_points"] if pd.notna(r["predicted_points"]) else 0.0,
                r["floor_points"] if pd.notna(r["floor_points"]) else 0.0,
                r["ceiling_points"] if pd.notna(r["ceiling_points"]) else 0.0,
                (r["p_haul_10plus"] * 100) if pd.notna(r["p_haul_10plus"]) else 0.0,
            )
            for _, r in top_rows.iterrows()
        )
        components.html(
            f'<div style="display:flex;flex-direction:column;gap:10px;">{rows_html}</div>',
            height=72 * len(top_rows) + 10 * (len(top_rows) - 1) + 20,
        )
    st.caption(
        "Predicted/floor/ceiling bars share one 0–12pt scale (a real player's ceiling can run "
        "higher; the bar clips there but the number alongside it doesn't). Ceiling % is a "
        "separate haul-probability classifier, not derived from the point estimate — see "
        "notebooks/09_error_analysis.ipynb."
    )

    with st.expander(f"Full table ({len(view)} players)"):
        view_display = view.copy()
        view_display["range"] = view_display.apply(
            lambda r: f"{r['floor_points']:.1f}–{r['ceiling_points']:.1f}"
            if pd.notna(r["floor_points"]) and pd.notna(r["ceiling_points"]) else "—",
            axis=1,
        )
        st.dataframe(
            view_display[[
                "web_name", "position", "team", "price", "predicted_points", "horizon_points", "range",
                "p_haul_10plus", "predicted_value", "total_points", "selected_by_percent", "status",
            ]].rename(columns={
                "web_name": "Player", "position": "Pos", "team": "Team", "price": "Price (£m)",
                "predicted_points": "Predicted (next GW)", "horizon_points": f"Predicted (next {DEFAULT_HORIZON} GWs)",
                "range": "Floor–Ceiling", "p_haul_10plus": "Ceiling %", "predicted_value": "Predicted pts/£m",
                "total_points": "Total pts", "selected_by_percent": "Owned %", "status": "Status",
            }),
            use_container_width=True, height=500,
        )

with tab_fixtures:
    deadline_gw = load_next_deadline(season)
    start_gw = deadline_gw["event_id"] if deadline_gw is not None else 2
    grid = load_fixture_grid(season, start_gw, DEFAULT_HORIZON)

    st.markdown(
        f"<div style='font-size:12px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;"
        f"color:{TEXT_MUTED};margin-bottom:6px;'>Fixtures</div>"
        f"<div style='font-size:32px;font-weight:800;line-height:1;color:{TEXT_PRIMARY};'>"
        f"Next {DEFAULT_HORIZON} gameweeks</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "This project's own fixture-difficulty score (rolling team attack/defense form), not "
        "FPL's published FDR — the two disagree often. A double gameweek's two fixtures are "
        "shown joined; a blank shows as a dash."
    )

    if grid.empty:
        st.info("No upcoming fixtures ingested yet.")
    else:
        gw_cols = sorted(grid["event_id"].unique())

        def _cell(rows: pd.DataFrame) -> tuple[str, float]:
            opp_label = " + ".join(f"{r['opponent']} ({'H' if r['is_home'] else 'A'})" for _, r in rows.iterrows())
            return opp_label, rows["fixture_difficulty_score"].mean()

        grid_rows = []
        for team, team_rows in grid.groupby("team"):
            row = {"Team": team}
            scores = []
            for gw in gw_cols:
                gw_rows = team_rows[team_rows["event_id"] == gw]
                if gw_rows.empty:
                    row[f"GW{gw}"] = "—"
                else:
                    label, score = _cell(gw_rows)
                    row[f"GW{gw}"] = label
                    scores.append(score)
            avg_score = np.mean(scores) if scores else 0.0
            row["_avg"] = avg_score
            row["Run rating"] = (
                "Easy" if avg_score <= -0.2 else "Brutal" if avg_score >= 0.25 else "Mixed"
            )
            grid_rows.append(row)
        grid_df = pd.DataFrame(grid_rows).sort_values("_avg")

        # Cell background by that team-gameweek's own difficulty score - a real, per-cell scale
        # (green=easy through red=brutal), not just the row-level run rating.
        score_lookup = {
            (t, gw): grid[(grid["team"] == t) & (grid["event_id"] == gw)]["fixture_difficulty_score"].mean()
            for t in grid_df["Team"] for gw in gw_cols
        }

        display_cols = ["Team"] + [f"GW{gw}" for gw in gw_cols] + ["Run rating"]

        def _style_row(row: pd.Series) -> list[str]:
            # Must return exactly one style per column in `subset` below (display_cols), not
            # per column in the full grid_df - Styler.apply with a subset maps the returned
            # list positionally onto the subset, and a length mismatch raises ValueError.
            styles = []
            for col in display_cols:
                if col not in [f"GW{gw}" for gw in gw_cols]:
                    styles.append("")
                    continue
                gw = int(col.replace("GW", ""))
                score = score_lookup.get((row["Team"], gw))
                if score is None or pd.isna(score):
                    styles.append(f"background-color:{CARD_BG};color:{TEXT_MUTED};")
                elif score <= -0.35:
                    styles.append(f"background-color:{STATUS_GOOD};color:{BG};font-weight:600;")
                elif score <= -0.1:
                    styles.append(f"background-color:#15803D;color:{TEXT_PRIMARY};font-weight:600;")
                elif score <= 0.1:
                    styles.append(f"background-color:{CARD_BORDER};color:{TEXT_PRIMARY};")
                elif score <= 0.35:
                    styles.append(f"background-color:{STATUS_WARNING};color:{BG};font-weight:600;")
                else:
                    styles.append(f"background-color:{CAPTAIN_BADGE};color:#fff;font-weight:600;")
            return styles

        styled = grid_df[display_cols].style.apply(_style_row, axis=1, subset=display_cols)
        st.dataframe(styled, use_container_width=True, hide_index=True, height=(len(grid_df) + 1) * 38)

        best_team = grid_df.iloc[0]
        worst_team = grid_df.iloc[-1]
        insight_cols = st.columns(2)
        with insight_cols[0]:
            components.html(render_action_tile_html(
                "good", f"{best_team['Team']}: best run in the window",
                f"Lowest average fixture difficulty over the next {DEFAULT_HORIZON} gameweeks.",
            ), height=110)
        with insight_cols[1]:
            components.html(render_action_tile_html(
                "attention", f"{worst_team['Team']}: hardest run in the window",
                f"Highest average fixture difficulty over the next {DEFAULT_HORIZON} gameweeks.",
            ), height=110)

        if settings.ENTRY_ID:
            owned_squad, _ = load_squad(season, settings.ENTRY_ID)
            if not owned_squad.empty:
                owned_teams = set(owned_squad["team"])
                if worst_team["Team"] in owned_teams:
                    st.warning(
                        f"You own players from **{worst_team['Team']}**, this window's hardest run — "
                        "worth checking if any are worth rotating out before it starts."
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
        opt_squad = opt_squad.copy()
        opt_squad["is_captain"] = opt_squad["captain"]
        # The MILP only selects a captain, not a vice - a real gap in squad_builder.py, not
        # papered over here: every badge below renders without a vice-captain marker.
        opt_squad["is_vice_captain"] = False
        opt_squad["multiplier"] = opt_squad["starting"].astype(int)
        opt_starting = opt_squad[opt_squad["starting"]]
        opt_bench = opt_squad[~opt_squad["starting"]]

        stat_cols = st.columns(3 if settings.ENTRY_ID else 2)
        with stat_cols[0]:
            components.html(render_stat_tile_html(
                "Projected XI points", f"{opt_objective:.1f}", sublabel="Captain doubled", accent=STATUS_GOOD,
            ), height=124)
        with stat_cols[1]:
            spend = opt_squad["price"].sum()
            components.html(render_stat_tile_html(
                "Spend", f"£{spend:.1f}m", sublabel=f"£{budget - spend:.1f}m unspent",
            ), height=124)

        actual_total = None
        if settings.ENTRY_ID:
            actual_squad, _ = load_squad(season, settings.ENTRY_ID)
            actual_squad = actual_squad.merge(
                rankings[["player_code", "horizon_points", "horizon_fixtures"]], on="player_code", how="left"
            )
            if not actual_squad.empty:
                actual_xi, _ = best_starting_xi(actual_squad, value_col=opt_value_col)
                actual_total = actual_xi[opt_value_col].sum()
                with stat_cols[2]:
                    components.html(render_stat_tile_html(
                        "Your squad, same basis", f"{actual_total:.1f}",
                        sublabel=f"{opt_objective - actual_total:+.1f} vs. this squad", accent=STATUS_WARNING,
                    ), height=124)

        st.markdown("#### The solver's XI")
        components.html(render_pitch_html(opt_starting, set(), value_col=opt_value_col), height=500)
        if not opt_bench.empty:
            components.html(render_bench_html(opt_bench, set(), value_col=opt_value_col), height=120)

        if settings.ENTRY_ID and not actual_squad.empty:
            overlap = set(opt_squad["player_code"]) & set(actual_squad["player_code"])
            transfers_needed = 15 - len(overlap)
            components.html(render_action_tile_html(
                "good" if transfers_needed == 0 else "attention",
                f"{len(overlap)} of your 15 already match this squad",
                "Your squad already matches." if transfers_needed == 0 else
                f"Getting the rest costs {transfers_needed} transfers — a wildcard decision, not a weekly one.",
            ), height=110)

        st.caption(
            "Solved as one integer program: squad and starting XI chosen together, captain's "
            f"double included, max 3 per club. Total spend £{opt_squad['price'].sum():.1f}m of "
            f"£{budget:.1f}m budget."
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
