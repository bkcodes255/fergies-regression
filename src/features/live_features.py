"""Builds the same feature set historical_features.py trains on, but sourced from our own
live-ingested Postgres data instead of the historical CSV archive - the bridge from a trained
model to an actual prediction for our current 2026/27 squad.

Reuses engineer_features() unchanged, by first reshaping our own tables into the exact column
shape historical_features.py expects (season, element, GW, minutes, total_points, ...). This
guarantees the rolling-feature math is identical between training and serving - the single
easiest way to introduce silent bias is to have serve-time features computed with subtly
different logic than train-time features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.historical_features import (
    CONGESTED_REST_THRESHOLD_DAYS,
    CONGESTION_WINDOW_DAYS,
    DC_SUM_COLS,
    SUM_COLS,
    TEAM_FORM_WINDOW,
    compute_injury_features,
    engineer_features,
    is_muscle_tendon_injury,
    load_player_injuries,
)

# Historical training data used "GK" for goalkeepers (vaastav archive convention); our own
# schema's element_type uses FPL's own GKP/DEF/MID/FWD labels. Map to match what the model
# was actually trained on - get this wrong and every goalkeeper's position one-hot is silently
# all-zero instead of pos_GK=1.
ELEMENT_TYPE_TO_TRAINED_POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

LIVE_QUERY = """
    SELECT
        s.season, s.player_code AS element, s.event_id AS "GW",
        p.web_name AS name, ps.element_type, s.team_code,
        s.minutes, s.goals_scored, s.assists, s.bps, s.bonus, s.total_points,
        s.clean_sheets, s.goals_conceded, s.own_goals, s.saves, s.yellow_cards, s.red_cards,
        s.starts, s.expected_goals, s.expected_assists, s.expected_goal_involvements,
        s.expected_goals_conceded, s.influence, s.creativity, s.threat, s.ict_index,
        s.defensive_contribution,
        lp.now_cost AS value
    FROM player_gameweek_stats s
    JOIN players p ON p.player_code = s.player_code
    JOIN player_seasons ps ON ps.player_code = s.player_code AND ps.season = s.season
    JOIN LATERAL (
        SELECT now_cost FROM player_price_snapshots pps
        WHERE pps.player_code = s.player_code AND pps.season = s.season
        ORDER BY snapshot_date DESC LIMIT 1
    ) lp ON true
    WHERE s.season = %(season)s
"""


FIXTURES_QUERY = """
    SELECT fpl_fixture_id AS fixture, event_id, kickoff_time, team_h_code, team_a_code,
           team_h_score, team_a_score, finished_provisional
    FROM fixtures WHERE season = %(season)s
"""


def _team_perspectives(fx: pd.DataFrame) -> pd.DataFrame:
    """Reshapes one row per fixture (home team's + away team's columns together) into one
    row per (team, fixture) - the same shape historical_features._extract_fixture_results
    produces from the CSV archive, so both sources feed identical rolling-form math."""
    home = fx.rename(columns={
        "team_h_code": "team_code", "team_a_code": "opponent_code",
        "team_h_score": "goals_for", "team_a_score": "goals_against",
    }).copy()
    home["was_home"] = True
    away = fx.rename(columns={
        "team_a_code": "team_code", "team_h_code": "opponent_code",
        "team_a_score": "goals_for", "team_h_score": "goals_against",
    }).copy()
    away["was_home"] = False
    cols = ["fixture", "event_id", "kickoff_time", "team_code", "opponent_code", "was_home",
            "goals_for", "goals_against", "finished_provisional"]
    return pd.concat([home[cols], away[cols]], ignore_index=True)


def _load_team_form_now(conn, season: str) -> pd.DataFrame:
    """Each team's rolling attack/defense form as of the most recent finished result - the
    same 'form entering the next fixture' notion v_team_latest_form computes for the
    dashboard's Fixture Difficulty Score (sql/analytics.sql), built independently here so it
    stays directly joinable against the model's feature columns. No extra shift needed (unlike
    historical_features._compute_team_form): every row here already happened before 'now', so
    the rolling mean over the last TEAM_FORM_WINDOW played fixtures already IS the form
    entering whatever comes next - v_team_latest_form makes the identical choice."""
    fx = pd.read_sql_query(FIXTURES_QUERY, conn, params={"season": season})
    played = _team_perspectives(fx)
    played = played[played["finished_provisional"] & played["goals_for"].notna()].copy()
    played["kickoff_time"] = pd.to_datetime(played["kickoff_time"], utc=True)
    played = played.sort_values(["team_code", "kickoff_time"])
    league_avg = played["goals_for"].mean() if len(played) else 1.3

    def _tail_rolling_mean(s: pd.Series) -> float:
        rolled = s.rolling(TEAM_FORM_WINDOW, min_periods=1).mean()
        return rolled.iloc[-1] if len(rolled) else np.nan

    form = played.groupby("team_code").agg(
        attack_form=("goals_for", _tail_rolling_mean),
        defense_form=("goals_against", _tail_rolling_mean),
    ).reset_index()
    form[["attack_form", "defense_form"]] = form[["attack_form", "defense_form"]].fillna(league_avg)
    return form


def _load_team_congestion(conn, season: str) -> pd.DataFrame:
    """Each team's days_since_last_match/matches_last_14d/is_congested as of their NEXT
    unplayed fixture - mirrors historical_features._compute_fixture_congestion's per-fixture
    notion, just computed from the live `fixtures` table instead of the CSV archive. Same
    EPL-only limitation stated there: only counts EPL matches we've ingested, not cup/European
    fixtures."""
    fx = pd.read_sql_query(FIXTURES_QUERY, conn, params={"season": season})
    perspectives = _team_perspectives(fx)
    perspectives["kickoff_time"] = pd.to_datetime(perspectives["kickoff_time"], utc=True)

    played = perspectives[perspectives["finished_provisional"] & perspectives["kickoff_time"].notna()]
    unplayed = perspectives[~perspectives["finished_provisional"] & perspectives["kickoff_time"].notna()]

    last_played = played.sort_values("kickoff_time").groupby("team_code")["kickoff_time"].last()
    next_kickoff = unplayed.sort_values("kickoff_time").groupby("team_code")["kickoff_time"].first()

    rows = []
    for team_code, next_kt in next_kickoff.items():
        last_kt = last_played.get(team_code)
        days_since = (next_kt - last_kt) / np.timedelta64(1, "D") if pd.notna(last_kt) else CONGESTION_WINDOW_DAYS
        team_played = played[played["team_code"] == team_code]
        window_start = next_kt - pd.Timedelta(days=CONGESTION_WINDOW_DAYS)
        matches_in_window = int((team_played["kickoff_time"] >= window_start).sum())
        rows.append({"team_code": team_code, "days_since_last_match": max(days_since, 0),
                     "matches_last_14d": matches_in_window})

    congestion = pd.DataFrame(rows, columns=["team_code", "days_since_last_match", "matches_last_14d"])
    congestion["is_congested"] = (congestion["days_since_last_match"] <= CONGESTED_REST_THRESHOLD_DAYS).astype(float)
    return congestion


def _load_next_fixture(conn, season: str) -> pd.DataFrame:
    """Each team's next unplayed fixture - opponent + home/away, for the synthetic
    prediction row. Picks the earliest by kickoff_time, so a team in a blank gameweek (no
    fixture in the very next event_id) still gets its real next fixture, not a spurious one."""
    fx = pd.read_sql_query(FIXTURES_QUERY, conn, params={"season": season})
    perspectives = _team_perspectives(fx)
    unplayed = perspectives[~perspectives["finished_provisional"] & perspectives["kickoff_time"].notna()].copy()
    unplayed["kickoff_time"] = pd.to_datetime(unplayed["kickoff_time"], utc=True)
    unplayed = unplayed.sort_values("kickoff_time")
    return unplayed.drop_duplicates(subset=["team_code"], keep="first")[
        ["team_code", "opponent_code", "was_home", "kickoff_time"]
    ]


def load_live_gameweeks(conn, season: str) -> pd.DataFrame:
    df = pd.read_sql_query(LIVE_QUERY, conn, params={"season": season})
    df["position"] = df["element_type"].map(ELEMENT_TYPE_TO_TRAINED_POSITION)
    df["dc_data_available"] = 1  # 2026/27 has the DC stat; live pipeline always sets this
    df["xg_data_available"] = 1  # ...and the xG-family stats/`starts`, unlike 2020-21/2021-22
    # our own player_code IS the stable cross-season identity already (that's the whole point
    # of keying the schema on it) - engineer_features()'s prior-season lookup expects a `code`
    # column by that same name, shared with the historical CSV path where element != code.
    df["code"] = df["element"]
    for col in DC_SUM_COLS:
        if col not in df.columns:
            df[col] = 0.0
    for col in SUM_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _append_synthetic_next_gw_row(raw: pd.DataFrame) -> pd.DataFrame:
    """For each player, adds one placeholder row for the next (unplayed) gameweek with all
    stat columns blank. This is what makes the rolling features come out right: engineer_features
    shifts by 1 before rolling, so a player's actual last-played row has NO history of its own
    (nothing precedes it yet) - it's the row AFTER it whose shifted rolling window correctly
    pulls in that real data. Without this, build_live_feature_frame would grab each player's
    last real row and get all-NaN/zero rolling features, i.e. predict as if nobody has ever
    played before - exactly the bug this function exists to avoid."""
    latest_per_player = raw.sort_values("GW").groupby("element", as_index=False).tail(1)
    synthetic = latest_per_player.copy()
    synthetic["GW"] = synthetic["GW"] + 1
    for col in SUM_COLS + DC_SUM_COLS:
        synthetic[col] = 0.0
    return pd.concat([raw, synthetic], ignore_index=True)


def build_live_feature_frame(conn, season: str, feature_cols: list[str]) -> pd.DataFrame:
    """Returns one row per player: rolling feature state as of right before the NEXT
    (not-yet-played) gameweek, correctly built from all real prior gameweeks. `feature_cols`
    should come from the trained model_versions row so column order/presence always matches
    what the model expects."""
    raw = load_live_gameweeks(conn, season)
    augmented = _append_synthetic_next_gw_row(raw)
    engineered = engineer_features(augmented)

    latest = (
        engineered.sort_values("GW")
        .groupby("element", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    next_fixture = _load_next_fixture(conn, season)
    team_form_now = _load_team_form_now(conn, season)
    latest = latest.merge(next_fixture, on="team_code", how="left")
    latest = latest.merge(
        team_form_now.rename(columns={"attack_form": "own_attack_form", "defense_form": "own_defense_form"}),
        on="team_code", how="left",
    )
    latest = latest.merge(
        team_form_now.rename(columns={
            "team_code": "opponent_code", "attack_form": "opp_attack_form", "defense_form": "opp_defense_form",
        }),
        on="opponent_code", how="left",
    )
    latest["was_home"] = latest["was_home"].astype(float)

    congestion = _load_team_congestion(conn, season)
    latest = latest.merge(congestion, on="team_code", how="left")
    latest["days_since_last_match"] = latest["days_since_last_match"].fillna(CONGESTION_WINDOW_DAYS)
    latest["matches_last_14d"] = latest["matches_last_14d"].fillna(0)
    latest["is_congested"] = latest["is_congested"].fillna(0.0)

    # Injury features "as of" the player's own upcoming fixture kickoff - same leak-free notion
    # as training (compute_injury_features), just using the real live fixtures table's next
    # unplayed kickoff_time instead of a historical row's own kickoff_time. Falls back to "now"
    # for any player with no resolved next fixture (e.g. season just ended).
    as_of = latest["kickoff_time"].fillna(pd.Timestamp.utcnow())
    injuries = load_player_injuries()
    injury_features = compute_injury_features(latest["element"], as_of, injuries)
    for col in ("days_since_last_injury", "injuries_last_365d", "is_returning_from_injury", "injury_data_available"):
        latest[col] = injury_features[col].to_numpy()

    muscle_injuries = injuries[injuries["injury_type"].map(is_muscle_tendon_injury)]
    muscle_features = compute_injury_features(latest["element"], as_of, muscle_injuries)
    latest["muscle_injuries_last_365d"] = muscle_features["injuries_last_365d"].to_numpy()
    latest["injury_congestion_risk"] = latest["muscle_injuries_last_365d"] * latest["is_congested"]

    position_dummies = pd.get_dummies(latest["position"], prefix="pos")
    latest = pd.concat([latest, position_dummies], axis=1)

    for col in feature_cols:
        if col not in latest.columns:
            latest[col] = 0.0

    X = latest[feature_cols].fillna(0.0)
    # "GW" here is the synthetic row's gameweek number - i.e. the gameweek being predicted,
    # not the last one actually played.
    meta = latest[["element", "name", "GW"]].rename(columns={"element": "player_code", "GW": "predicting_gw"})
    return meta, X
