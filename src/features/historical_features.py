"""Builds a leak-free training dataset from historical FPL seasons
(data/historical/*/merged_gw.csv, sourced from the vaastav/Fantasy-Premier-League archive).

Every rolling/lag feature uses ONLY prior gameweeks (shift(1) before any rolling window,
computed per (season, element) group) - a row's features never see that row's own outcome
or any future gameweek. Same no-leakage discipline as the rest of this project's backtesting
design (data/data_dictionary.md, sql/analytics.sql).

Double-gameweek fixtures are pre-aggregated to one row per (season, element, GW) by summing
the additive stats across fixtures, matching how the live FPL API's event/{gw}/live/ endpoint
already aggregates DGWs for the current season.

defensive_contribution and its raw-count inputs (clearances_blocks_interceptions, tackles,
recoveries) only exist in FPL data from the 2025-26 season onward - older seasons don't have
the columns at all, not just zeros. Rows from those seasons get dc_data_available=0 and the
dc_roll* features zeroed, rather than fabricating a value that never existed.

Same gap, same fix, for `starts` and the expected_* (xG-family) stats: FPL didn't track them at
all before the 2022-23 season, so 2020-21/2021-22 (added to broaden training data beyond the
original 3-season window) get xg_data_available=0 and their derived roll features zeroed rather
than a fabricated 0.0 that looks like a real "no expected-goal involvement" reading.

Fixture-strength features (was_home, own/opp_attack_form, own/opp_defense_form): the model was
otherwise blind to who a player is about to face - added after the Phase 6.5 error-analysis
notebook flagged "opponent defensive weakness" as an untested direction. Computed from each
team's own leak-free rolling goals-for/against (_compute_team_form, shift(1) before rolling,
same discipline as everything else here), then attached at the per-fixture level before the
DGW aggregation below so a double-gameweek's two fixtures average together like every other
per-fixture stat.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HISTORICAL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "historical"
SEASONS_WITH_DC = {"2025-26"}
ROLLING_WINDOWS = (3, 5)
TEAM_FORM_WINDOW = 5  # matches v_team_form's window (sql/analytics.sql) - same design choice,
# reused here so the live dashboard's Fixture Difficulty Score and this model's opponent
# features are built on the same notion of "recent form", just computed independently in
# Python since training reads from the historical CSV archive, not our own live DB.

SUM_COLS = [
    "minutes", "goals_scored", "assists", "bps", "bonus", "total_points",
    "clean_sheets", "goals_conceded", "own_goals", "saves", "yellow_cards", "red_cards",
    "starts", "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "influence", "creativity", "threat", "ict_index",
]
DC_SUM_COLS = ["clearances_blocks_interceptions", "defensive_contribution", "recoveries", "tackles"]
XG_SUM_COLS = ["starts", "expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded"]
FIRST_COLS = ["name", "position", "value", "selected"]


def _load_team_name_mapping(season_dir: Path) -> dict:
    """merged_gw.csv's own `team` column is the full team NAME ('Man Utd'), but its
    `opponent_team` column is that season's numeric team id (1-20, from teams.csv) - not the
    same space, despite naming that suggests otherwise. teams.csv's `name` column matches
    merged_gw.csv's `team` strings exactly, so this maps name -> id to put both columns in
    the same space before any fixture-level join can work."""
    path = season_dir / "teams.csv"
    if not path.exists():
        return {}
    teams = pd.read_csv(path, usecols=["id", "name"])
    return dict(zip(teams["name"], teams["id"]))


def _load_code_mapping(season_dir: Path) -> dict:
    """element id (season-local, resets every season) -> code (stable across seasons),
    from that season's players_raw.csv. Same season-scoped-id gotcha as the live FPL API
    (data/data_dictionary.md) - vaastav's archive has the identical issue and the identical
    fix (a stable `code` field)."""
    path = season_dir / "players_raw.csv"
    if not path.exists():
        return {}
    raw = pd.read_csv(path, usecols=["id", "code"])
    return dict(zip(raw["id"], raw["code"]))


def _extract_fixture_results(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, fixture): that team's own goals for/against and home/away for the
    match. Deduped from the per-player rows - every player on a team shares an identical
    view of their own team's fixture-level facts, so this just picks out the distinct ones."""
    cols = ["fixture", "team", "opponent_team", "was_home", "team_h_score", "team_a_score", "kickoff_time"]
    fx = df[cols].drop_duplicates(subset=["fixture", "team"]).copy()
    fx["kickoff_time"] = pd.to_datetime(fx["kickoff_time"])
    fx["goals_for"] = np.where(fx["was_home"], fx["team_h_score"], fx["team_a_score"])
    fx["goals_against"] = np.where(fx["was_home"], fx["team_a_score"], fx["team_h_score"])
    return fx[["fixture", "team", "opponent_team", "was_home", "kickoff_time", "goals_for", "goals_against"]]


def _compute_team_form(fixture_results: pd.DataFrame, window: int = TEAM_FORM_WINDOW) -> pd.DataFrame:
    """Leak-free rolling attack/defense form per team, ordered by kickoff time - shift(1)
    before rolling, same no-leakage discipline as every other feature in this module, so a
    team's form entering a fixture never includes that fixture's own result. A team's first
    fixture(s) of the season (no prior result to roll over) get the league-average goals
    figure instead of 0 - 0 would read as 'guaranteed to face the weakest possible
    attack/defense', a worse assumption than 'unknown, assume average'."""
    fixture_results = fixture_results.sort_values(["team", "kickoff_time"]).copy()
    league_avg_goals = fixture_results["goals_for"].mean()
    grouped = fixture_results.groupby("team", sort=False)
    fixture_results["attack_form"] = (
        grouped["goals_for"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .fillna(league_avg_goals)
    )
    fixture_results["defense_form"] = (
        grouped["goals_against"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .fillna(league_avg_goals)
    )
    return fixture_results


def _attach_fixture_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds was_home + each row's own team's and opponent's rolling attack/defense form
    (leak-free as of entering that specific fixture) to the per-player-fixture rows, before
    they get aggregated to one row per (season, element, GW). Unlike the rolling player-stat
    features elsewhere in this module, these are NOT shifted again at the player level - the
    upcoming fixture and the opponent's rolling form going into it are legitimately known
    before kickoff, so no additional lag is needed on top of _compute_team_form's own shift."""
    form = _compute_team_form(_extract_fixture_results(df))
    own = form[["fixture", "team", "attack_form", "defense_form"]].rename(
        columns={"attack_form": "own_attack_form", "defense_form": "own_defense_form"}
    )
    opp = form[["fixture", "team", "attack_form", "defense_form"]].rename(
        columns={"team": "opponent_team", "attack_form": "opp_attack_form", "defense_form": "opp_defense_form"}
    )
    df = df.merge(own, on=["fixture", "team"], how="left")
    df = df.merge(opp, on=["fixture", "opponent_team"], how="left")
    df["was_home"] = df["was_home"].astype(float)
    return df


def _load_season(season_dir: Path) -> pd.DataFrame:
    season = season_dir.name
    df = pd.read_csv(season_dir / "merged_gw.csv")
    df["season"] = season
    df["code"] = df["element"].map(_load_code_mapping(season_dir))
    df["team"] = df["team"].map(_load_team_name_mapping(season_dir))
    df = _attach_fixture_features(df)

    has_dc = season in SEASONS_WITH_DC
    has_xg = "expected_goals" in df.columns  # FPL didn't track xG-family stats or `starts`
    # at all before 2022-23 - detected from the raw file rather than a hardcoded season list,
    # so adding another pre-2022-23 season later doesn't need a code change here too.
    for col in DC_SUM_COLS:
        if col not in df.columns:
            df[col] = np.nan
    for col in XG_SUM_COLS:
        if col not in df.columns:
            df[col] = np.nan

    FIXTURE_COLS = ["was_home", "own_attack_form", "own_defense_form", "opp_attack_form", "opp_defense_form"]
    agg = {col: "sum" for col in SUM_COLS + DC_SUM_COLS}
    agg.update({col: "mean" for col in FIXTURE_COLS})
    agg.update({col: "first" for col in FIRST_COLS + ["code"]})

    grouped = df.groupby(["season", "element", "GW"], as_index=False).agg(agg)
    grouped["dc_data_available"] = int(has_dc)
    grouped["xg_data_available"] = int(has_xg)
    return grouped


def _prior_season_str(season: str) -> str:
    start_year = int(season.split("-")[0])
    return f"{start_year - 1}-{str(start_year % 100).zfill(2)}"


def _prior_season_summary(season: str) -> pd.DataFrame:
    """Each returning player's FULL prior-season performance (by stable `code`), used as a
    'how did they do last year' prior - without this, a proven veteran and a total unknown
    look identical for however many gameweeks it takes the current season's own rolling
    history to build up. Matched by code so this works whether `season` is a historical
    training season or 2026/27 live data (same function, same join key, on purpose)."""
    empty = pd.DataFrame({
        "code": pd.Series(dtype="int64"),
        "prev_season_points_per90": pd.Series(dtype="float64"),
        "prev_season_minutes_avg": pd.Series(dtype="float64"),
        "prev_season_total_points": pd.Series(dtype="float64"),
    })
    prior_dir = HISTORICAL_DIR / _prior_season_str(season)
    if not prior_dir.exists() or not (prior_dir / "merged_gw.csv").exists():
        return empty

    prior = _load_season(prior_dir)
    summary = prior.groupby("code").agg(
        total_points=("total_points", "sum"), total_minutes=("minutes", "sum"), gameweeks=("GW", "nunique"),
    ).reset_index()
    summary["prev_season_points_per90"] = np.where(
        summary["total_minutes"] > 0, summary["total_points"] / summary["total_minutes"] * 90, 0.0
    )
    summary["prev_season_minutes_avg"] = summary["total_minutes"] / summary["gameweeks"].clip(lower=1)
    summary = summary.rename(columns={"total_points": "prev_season_total_points"})
    return summary[["code", "prev_season_points_per90", "prev_season_minutes_avg", "prev_season_total_points"]]


def load_all_seasons() -> pd.DataFrame:
    frames = [_load_season(p) for p in sorted(HISTORICAL_DIR.iterdir()) if p.is_dir()]
    df = pd.concat(frames, ignore_index=True)
    df["GW"] = df["GW"].astype(int)
    # "AM" = the experimental Manager position (real club managers were pickable in some
    # historical seasons) - not present in 2026/27's element_types (verified live: only
    # GKP/DEF/MID/FWD), so irrelevant to any squad we'd actually build this season.
    df = df[df["position"] != "AM"]
    return df.sort_values(["season", "element", "GW"]).reset_index(drop=True)


def _prior_rolling(series: pd.Series, window: int, agg: str) -> pd.Series:
    shifted = series.shift(1)
    if agg == "mean":
        return shifted.rolling(window, min_periods=1).mean()
    if agg == "sum":
        return shifted.rolling(window, min_periods=1).sum()
    raise ValueError(agg)


def _engineer_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("GW").copy()

    for window in ROLLING_WINDOWS:
        min_sum = _prior_rolling(g["minutes"], window, "sum")

        def per90(col: str) -> pd.Series:
            col_sum = _prior_rolling(g[col], window, "sum")
            return np.where(min_sum > 0, col_sum / min_sum * 90, np.nan)

        g[f"points_per90_roll{window}"] = per90("total_points")
        g[f"goals_per90_roll{window}"] = per90("goals_scored")
        g[f"assists_per90_roll{window}"] = per90("assists")
        g[f"xg_per90_roll{window}"] = per90("expected_goals")
        g[f"xa_per90_roll{window}"] = per90("expected_assists")
        g[f"xgc_per90_roll{window}"] = per90("expected_goals_conceded")
        g[f"minutes_roll{window}"] = _prior_rolling(g["minutes"], window, "mean")
        g[f"bps_roll{window}"] = _prior_rolling(g["bps"], window, "mean")
        g[f"ict_index_roll{window}"] = _prior_rolling(g["ict_index"], window, "mean")
        # threat/creativity are FPL's own attacking sub-indices, already summed into
        # ict_index but never exposed separately before - added specifically to probe the
        # "haul-blindness" finding (see notebooks/09_error_analysis.ipynb): a rising shot/
        # chance-creation trend is a plausible precursor to a big score that a blended index
        # or a pure past-points average wouldn't isolate on its own.
        g[f"threat_roll{window}"] = _prior_rolling(g["threat"], window, "mean")
        g[f"creativity_roll{window}"] = _prior_rolling(g["creativity"], window, "mean")
        g[f"dc_roll{window}"] = _prior_rolling(g["defensive_contribution"], window, "mean")
        g[f"starts_rate_roll{window}"] = _prior_rolling(g["starts"], window, "mean")

    prior_minutes_sum = g["minutes"].shift(1).expanding().sum()
    prior_points_sum = g["total_points"].shift(1).expanding().sum()
    g["season_points_per90_avg"] = np.where(
        prior_minutes_sum.fillna(0) > 0, prior_points_sum / prior_minutes_sum * 90, np.nan
    )
    g["season_minutes_avg"] = g["minutes"].shift(1).expanding().mean()
    g["gw_number"] = g["GW"]
    return g


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    groups = [
        _engineer_group(g)
        for _, g in df.groupby(["season", "element"], sort=False)
    ]
    result = pd.concat(groups, ignore_index=True)

    dc_cols = [c for c in result.columns if c.startswith("dc_roll")]
    for col in dc_cols:
        result.loc[result["dc_data_available"] == 0, col] = 0.0
        result[col] = result[col].fillna(0.0)

    xg_cols = [c for c in result.columns if c.startswith(("xg_per90_roll", "xa_per90_roll", "xgc_per90_roll", "starts_rate_roll"))]
    for col in xg_cols:
        result.loc[result["xg_data_available"] == 0, col] = 0.0
        result[col] = result[col].fillna(0.0)

    prior_frames = []
    for season in result["season"].unique():
        summary = _prior_season_summary(season)
        summary = summary.copy()
        summary["season"] = season
        prior_frames.append(summary)
    prior_all = pd.concat(prior_frames, ignore_index=True) if prior_frames else pd.DataFrame(
        columns=["code", "season", "prev_season_points_per90", "prev_season_minutes_avg", "prev_season_total_points"]
    )
    result = result.merge(prior_all, on=["season", "code"], how="left")
    result["had_prior_season"] = result["prev_season_points_per90"].notna().astype(int)
    for col in ["prev_season_points_per90", "prev_season_minutes_avg", "prev_season_total_points"]:
        result[col] = result[col].fillna(0.0).astype(float)

    result["target_points_per90"] = np.where(
        result["minutes"] > 0, result["total_points"] / result["minutes"] * 90, np.nan
    )
    result = result.rename(columns={"value": "price", "total_points": "target_total_points",
                                     "minutes": "target_minutes"})
    return result


FEATURE_COLS = [
    # NOTE: "selected" (ownership) deliberately excluded - vaastav's historical data stores
    # it as a raw manager COUNT, not a percentage, so it's on a different scale per season
    # and would not transfer cleanly to our own live pipeline's selected_by_percent when this
    # model is later applied to 2026/27 data. Price and rolling performance are scale-stable.
    "price", "dc_data_available", "xg_data_available",
    "season_points_per90_avg", "season_minutes_avg",
    "had_prior_season", "prev_season_points_per90", "prev_season_minutes_avg", "prev_season_total_points",
    "was_home", "own_attack_form", "own_defense_form", "opp_attack_form", "opp_defense_form",
] + [f"{stat}_roll{w}" for w in ROLLING_WINDOWS for stat in (
    "points_per90", "goals_per90", "assists_per90", "xg_per90", "xa_per90",
    "xgc_per90", "minutes", "bps", "ict_index", "dc", "starts_rate", "threat", "creativity",
)]


def build_training_frame() -> pd.DataFrame:
    """Loads all historical seasons, engineers features, and one-hot encodes position.
    Returns one row per (season, element, GW) with FEATURE_COLS + position dummies +
    target_total_points, target_minutes, target_points_per90, plus season/gw_number/name
    for bookkeeping. Rows are NOT filtered here (e.g. GW1 rows with no prior history are
    still present, with NaN/0 features) - callers decide what to drop for training.
    """
    raw = load_all_seasons()
    engineered = engineer_features(raw)
    position_dummies = pd.get_dummies(engineered["position"], prefix="pos")
    return pd.concat([engineered, position_dummies], axis=1)
