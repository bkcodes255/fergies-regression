"""Multi-week transfer-planning horizon: projects each player's existing next-GW prediction
across the next N gameweeks, scaled by their team's fixture difficulty each week
(v_fixture_difficulty, from Phase 2).

Deliberately NOT recursive re-prediction. The trained model only outputs total_points and
minutes directly - it has no sub-models for the ~10 other rolling-feature inputs (goals,
assists, bps, xg, ict_index, etc.), so simulating those forward would mean crudely
approximating most of the feature vector at every step, with error compounding over the
window. Repeating the model's own current-form estimate and letting only the fixture run vary
avoids that compounding and - as a bonus - adds a signal (fixture swings, blank/double
gameweeks) the single-GW prediction doesn't have at all today, since fixture difficulty isn't
one of the model's training features.

v_fixture_difficulty gives one row per (team, upcoming fixture) - a team with two fixtures in
one event_id (a double gameweek) naturally produces two rows for that event_id, and a team with
none (a blank gameweek) produces zero, so summing over the window handles both without special
casing.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_HORIZON = 5  # matches the original plan's "5-GW transfer horizon"

# fixture_difficulty_score is roughly zero-centered with observed std ~0.22 (season-to-date);
# this sensitivity gives a genuinely hard/easy fixture (~2 std out) a +/-20% swing rather than
# a token nudge, without letting one score outlier dominate the sum over the whole window.
DIFFICULTY_SENSITIVITY = 0.4
MULTIPLIER_BOUNDS = (0.6, 1.5)

FIXTURE_WINDOW_QUERY = """
    SELECT ps.player_code, fd.event_id, fd.fixture_difficulty_score
    FROM v_fixture_difficulty fd
    JOIN player_seasons ps ON ps.team_code = fd.team_code AND ps.season = fd.season
    WHERE fd.season = %(season)s
      AND fd.event_id >= %(start_gw)s AND fd.event_id < %(start_gw)s + %(horizon)s
"""


def difficulty_to_multiplier(score: pd.Series) -> pd.Series:
    return (1 - DIFFICULTY_SENSITIVITY * score).clip(*MULTIPLIER_BOUNDS)


def load_fixture_window(conn, season: str, start_gw: int, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """One row per (player_code, fixture) over [start_gw, start_gw + horizon)."""
    return pd.read_sql_query(
        FIXTURE_WINDOW_QUERY, conn, params={"season": season, "start_gw": start_gw, "horizon": horizon}
    )


def compute_horizon_points(rankings: pd.DataFrame, fixtures: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """rankings needs player_code + predicted_points (the existing next-GW prediction, used as
    the flat per-fixture baseline). Adds:
      - horizon_points: sum over the window of predicted_points * that fixture's difficulty
        multiplier (0 for a player with no fixtures in the window - a blank gameweek run)
      - horizon_fixtures: fixture count in the window, so blanks/doubles are visible, not just
        baked silently into the points total
    """
    result = rankings.copy()
    if fixtures.empty:
        result["horizon_points"] = result["predicted_points"] * horizon
        result["horizon_fixtures"] = horizon
        return result

    fixtures = fixtures.copy()
    fixtures["multiplier"] = difficulty_to_multiplier(fixtures["fixture_difficulty_score"])
    per_player = fixtures.groupby("player_code").agg(
        horizon_multiplier_sum=("multiplier", "sum"),
        horizon_fixtures=("event_id", "count"),
    ).reset_index()

    result = result.merge(per_player, on="player_code", how="left")
    result["horizon_fixtures"] = result["horizon_fixtures"].fillna(0).astype(int)
    result["horizon_multiplier_sum"] = result["horizon_multiplier_sum"].fillna(0.0)
    result["horizon_points"] = result["predicted_points"] * result["horizon_multiplier_sum"]
    return result.drop(columns=["horizon_multiplier_sum"])
