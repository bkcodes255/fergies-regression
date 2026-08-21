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
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HISTORICAL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "historical"
SEASONS_WITH_DC = {"2025-26"}
ROLLING_WINDOWS = (3, 5)

SUM_COLS = [
    "minutes", "goals_scored", "assists", "bps", "bonus", "total_points",
    "clean_sheets", "goals_conceded", "own_goals", "saves", "yellow_cards", "red_cards",
    "starts", "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "influence", "creativity", "threat", "ict_index",
]
DC_SUM_COLS = ["clearances_blocks_interceptions", "defensive_contribution", "recoveries", "tackles"]
FIRST_COLS = ["name", "position", "value", "selected"]


def _load_season(season_dir: Path) -> pd.DataFrame:
    season = season_dir.name
    df = pd.read_csv(season_dir / "merged_gw.csv")
    df["season"] = season

    has_dc = season in SEASONS_WITH_DC
    for col in DC_SUM_COLS:
        if col not in df.columns:
            df[col] = np.nan

    agg = {col: "sum" for col in SUM_COLS + DC_SUM_COLS}
    agg.update({col: "first" for col in FIRST_COLS})

    grouped = df.groupby(["season", "element", "GW"], as_index=False).agg(agg)
    grouped["dc_data_available"] = int(has_dc)
    return grouped


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
    "price", "dc_data_available",
    "season_points_per90_avg", "season_minutes_avg",
] + [f"{stat}_roll{w}" for w in ROLLING_WINDOWS for stat in (
    "points_per90", "goals_per90", "assists_per90", "xg_per90", "xa_per90",
    "xgc_per90", "minutes", "bps", "ict_index", "dc", "starts_rate",
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
