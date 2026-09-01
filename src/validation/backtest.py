"""Phase 6: decision-engine backtest - simulates a full held-out season week by week using
only leak-free at-the-time predictions, and scores three policies against real historical
results:

  1. full_engine   - the actual decision engine: an MILP squad pick at the first tracked
                      gameweek, then each week the greedy transfer plan + auto-substitution +
                      captain choice, all driven by the model's predictions.
  2. static_squad  - the SAME starting squad as (1), never transferred again; auto-sub/captain
                      still driven by predictions. Isolates the marginal value of the transfer
                      plan by holding squad composition fixed.
  3. static_oracle - the same starting squad again, never transferred, but auto-sub/captain are
                      chosen using ACTUAL results (hindsight) instead of predictions. A
                      same-squad ceiling that isolates how much is left on the table by
                      imperfect weekly judgment vs. how much is baked into squad composition
                      itself (comparing 2 vs 3 answers "how good is the model's weekly
                      captain/bench judgment, holding the squad fixed?").

Uses the 2025-26 season, which src.models.train already holds out entirely from training - this
is a genuine backtest, not an in-sample fit. Every week's prediction is generated from
historical_features.py's leak-free rolling features (shift(1) before any window), the same
discipline the model was trained and evaluated with - a gameweek's prediction never sees that
gameweek's own result.

Known simplifications (documented in README, not silently swept under the rug):
  - No historical injury/availability data survives in the vaastav archive's per-gameweek CSVs,
    so every player is treated as selectable every week (unlike the live dashboard's 'a'-status
    filter). Real managers had information this backtest doesn't.
  - No reactive real-FPL auto-substitution for a starter who blanks (0 minutes) - the lineup
    picked before the gameweek is scored as-is, not corrected after the fact.
  - The fixture-difficulty multi-week horizon (src.recommendations.horizon) isn't used here:
    it depends on the live Postgres `fixtures`/`teams` tables, which only cover 2026/27 - no
    historical season has fixture data ingested. This backtest is single-gameweek-horizon only.

Run directly:
    python -m src.validation.backtest
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

from config.settings import MODELS_DIR
from src.features.historical_features import HISTORICAL_DIR, build_training_frame
from src.ingestion.db import get_connection
from src.models.predict_live import get_best_model
from src.recommendations.squad_builder import build_optimal_squad
from src.recommendations.squad_optimizer import best_starting_xi
from src.recommendations.transfers import FREE_TRANSFER_CAP, suggest_transfer_plan

BACKTEST_SEASON = "2025-26"
BUDGET = 100.0
POSITION_RELABEL = {"GK": "GKP"}  # match FPL's/squad_builder's own convention, not vaastav's


def _load_team_lookup(season: str) -> pd.DataFrame:
    """merged_gw.csv has each player's real team per gameweek (handles mid-season transfers
    automatically) - historical_features.py's aggregation drops it since it's not a training
    feature, so pull it back in separately for squad-building's max-3-per-team constraint."""
    raw = pd.read_csv(HISTORICAL_DIR / season / "merged_gw.csv", usecols=["element", "GW", "team"])
    return raw.groupby(["element", "GW"], as_index=False)["team"].first()


def build_predictions_frame(conn, season: str = BACKTEST_SEASON) -> pd.DataFrame:
    """One row per (player, gameweek) for `season`, GW>1 (GW1 has no rolling history to predict
    from - the same convention src.models.train uses to exclude it from evaluation). Columns:
    player_code, web_name, position ('GKP'/'DEF'/'MID'/'FWD'), team, price, status, GW,
    predicted_points (leak-free, what the model would have said before that gameweek),
    actual_points (real total_points, known only in hindsight - used for scoring and for the
    static_oracle track)."""
    with conn.cursor() as cur:
        model_id, model_type, features_json, artifact_path = get_best_model(cur)
    feature_cols = json.loads(features_json) if isinstance(features_json, str) else features_json
    print(f"Using model_id={model_id} ({model_type}), {len(feature_cols)} features, from {artifact_path}")
    model = joblib.load(MODELS_DIR / artifact_path)

    df = build_training_frame()
    season_df = df[(df["season"] == season) & (df["gw_number"] > 1)].copy()
    X = season_df[feature_cols].fillna(0.0)
    season_df["predicted_points"] = model.predict(X)
    season_df["actual_points"] = season_df["target_total_points"]
    season_df["player_code"] = season_df["code"]
    season_df["web_name"] = season_df["name"]
    # historical "price" (from merged_gw's `value`) is in tenths of £m, like live now_cost -
    # squad_builder/transfers work in real £m against a real-£m budget, same as the dashboard.
    season_df["price"] = season_df["price"] / 10
    season_df["position"] = season_df["position"].replace(POSITION_RELABEL)
    season_df["status"] = "a"  # no historical injury/availability data - see module docstring

    team_lookup = _load_team_lookup(season)
    season_df = season_df.merge(team_lookup, on=["element", "GW"], how="left")
    season_df["team"] = season_df["team"].fillna("UNK")

    return season_df[[
        "player_code", "web_name", "position", "team", "price", "status",
        "GW", "predicted_points", "actual_points",
    ]].reset_index(drop=True)


def _refresh(squad_ids: pd.DataFrame, week_rankings: pd.DataFrame) -> pd.DataFrame:
    """squad_ids: just player_code for the 15 currently owned. Re-attaches THIS week's
    price/predicted/actual/team/status - all of which change week to week - so downstream
    transfer/lineup logic always compares players on the same gameweek's numbers."""
    return squad_ids[["player_code"]].merge(week_rankings, on="player_code", how="left")


def _score_week(starting_xi: pd.DataFrame, value_col: str) -> tuple[float, str]:
    """Returns (actual points scored this gameweek incl. captain double, captain's name).
    Captain = whoever ranks top on value_col within the starting XI, matching the dashboard's
    own captain-check logic (src/dashboard: 'model suggests X as captain')."""
    ranked = starting_xi.sort_values(value_col, ascending=False)
    captain = ranked.iloc[0]
    captain_bonus = 0.0 if pd.isna(captain["actual_points"]) else float(captain["actual_points"])
    total = float(starting_xi["actual_points"].fillna(0).sum()) + captain_bonus
    return total, str(captain["web_name"])


def run_backtest(season: str = BACKTEST_SEASON, budget: float = BUDGET) -> pd.DataFrame:
    conn = get_connection()
    try:
        predictions = build_predictions_frame(conn, season)
    finally:
        conn.close()

    gws = sorted(predictions["GW"].unique())
    start_gw = gws[0]
    print(f"Backtesting {season}, GW{start_gw}-GW{gws[-1]} ({len(gws)} gameweeks)...")

    start_rankings = predictions[predictions["GW"] == start_gw]
    squad0, _, _, obj0 = build_optimal_squad(start_rankings, budget, value_col="predicted_points")
    if squad0 is None:
        raise RuntimeError(f"No feasible starting squad at budget={budget} for GW{start_gw}")
    print(f"Initial squad (GW{start_gw}): {squad0['price'].sum():.1f}m spent, "
          f"predicted starting XI (incl. captain) = {obj0:.2f}")

    squad_ids = squad0[["player_code"]].copy()
    engine_squad = squad0.copy()
    static_squad_ids = squad_ids.copy()  # frozen for tracks 2 and 3

    bank = budget - squad0["price"].sum()
    free_transfers = 1
    rows = []

    for gw in gws:
        week_rankings = predictions[predictions["GW"] == gw]

        # --- Track 1: full decision engine ---
        if gw == start_gw:
            engine_week = _refresh(squad_ids, week_rankings)
            transfers_made = 0
            hit_cost = 0
        else:
            refreshed = _refresh(squad_ids, week_rankings)
            plan, engine_week, bank = suggest_transfer_plan(
                refreshed, week_rankings, bank, free_transfers, value_col="predicted_points"
            )
            transfers_made = len(plan)
            hit_cost = int(plan["hit"].sum()) if not plan.empty else 0
            free_transfers = min(FREE_TRANSFER_CAP, max(0, free_transfers - transfers_made) + 1)
        squad_ids = engine_week[["player_code"]]
        engine_xi, engine_formation = best_starting_xi(engine_week, value_col="predicted_points")
        engine_points, engine_captain = _score_week(engine_xi, "predicted_points")
        engine_points -= hit_cost  # transfer-hit cost is a real deduction from the actual score,
        # not just a factor in deciding whether the transfer was worth making

        # --- Track 2: static squad (same start, no transfers, predicted-driven weekly picks) ---
        static_week = _refresh(static_squad_ids, week_rankings)
        static_xi, static_formation = best_starting_xi(static_week, value_col="predicted_points")
        static_points, static_captain = _score_week(static_xi, "predicted_points")

        # --- Track 3: static squad, oracle (hindsight) weekly picks ---
        oracle_xi, oracle_formation = best_starting_xi(static_week, value_col="actual_points")
        oracle_points, oracle_captain = _score_week(oracle_xi, "actual_points")

        rows.append({
            "GW": gw, "transfers_made": transfers_made, "hit_cost": hit_cost,
            "free_transfers_after": free_transfers,
            "full_engine_points": engine_points, "full_engine_captain": engine_captain,
            "static_squad_points": static_points, "static_squad_captain": static_captain,
            "static_oracle_points": oracle_points, "static_oracle_captain": oracle_captain,
        })

    result = pd.DataFrame(rows)
    for col in ["full_engine_points", "static_squad_points", "static_oracle_points"]:
        result[col.replace("_points", "_cumulative")] = result[col].cumsum()
    return result


def summarize(result: pd.DataFrame) -> None:
    totals = {
        "Full decision engine": result["full_engine_points"].sum(),
        "Static squad (no transfers)": result["static_squad_points"].sum(),
        "Static squad, oracle picks": result["static_oracle_points"].sum(),
    }
    print(f"\n=== Season totals ({len(result)} gameweeks) ===")
    for label, total in totals.items():
        print(f"  {label:32s} {total:8.1f} pts")

    transfer_value = totals["Full decision engine"] - totals["Static squad (no transfers)"]
    judgment_gap = totals["Static squad, oracle picks"] - totals["Static squad (no transfers)"]
    print(f"\nTransfer plan's contribution (engine - static):        {transfer_value:+.1f} pts")
    print(f"Same-squad judgment ceiling left on the table:          {judgment_gap:+.1f} pts")
    print(f"Total transfers made across the season: {int(result['transfers_made'].sum())}")


def run() -> None:
    result = run_backtest()
    summarize(result)


if __name__ == "__main__":
    run()
