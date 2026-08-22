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

import pandas as pd

from src.features.historical_features import DC_SUM_COLS, SUM_COLS, engineer_features

# Historical training data used "GK" for goalkeepers (vaastav archive convention); our own
# schema's element_type uses FPL's own GKP/DEF/MID/FWD labels. Map to match what the model
# was actually trained on - get this wrong and every goalkeeper's position one-hot is silently
# all-zero instead of pos_GK=1.
ELEMENT_TYPE_TO_TRAINED_POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

LIVE_QUERY = """
    SELECT
        s.season, s.player_code AS element, s.event_id AS "GW",
        p.web_name AS name, ps.element_type,
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


def load_live_gameweeks(conn, season: str) -> pd.DataFrame:
    df = pd.read_sql_query(LIVE_QUERY, conn, params={"season": season})
    df["position"] = df["element_type"].map(ELEMENT_TYPE_TO_TRAINED_POSITION)
    df["dc_data_available"] = 1  # 2026/27 has the DC stat; live pipeline always sets this
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
