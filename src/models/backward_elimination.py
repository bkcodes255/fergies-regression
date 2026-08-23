"""Backward feature elimination for total_points_direct, driven by Random Forest permutation
importance - built after Brian asked whether the Model Lab dashboard's feature correlation
heatmap (feature-to-feature redundancy, not predictive value) meant near-zero-correlation
features should be dropped. It doesn't: that map answers a different question. This is the
principled alternative - rank features by actual contribution to held-out accuracy, drop the
weakest, repeat - instead of manually toggling 45 checkboxes or brute-forcing 2^45 combinations.

Critical design choice: every elimination round is scored against an INTERNAL validation split
carved out of the training seasons alone (train on 2020-21..2023-24, validate on 2024-25) -
never against the real TEST_SEASON (2025-26). Repeatedly checking ~10 feature subsets against
the same fixed holdout and keeping whichever looks best would itself be a form of overfitting to
that one season's noise - the same "best of many tries" issue that makes the Model Lab's paired
bootstrap p-value only valid for one planned comparison, not a search over many. Only the FINAL
selected feature set gets one evaluation against the true TEST_SEASON, exactly like every other
model in this project, and that single result (plus the all-features baseline for comparison) is
what gets logged to model_versions - not every internal round.

Uses a lightweight RF-only fit + permutation_importance per round (not the full
evaluate_feature_subset(), which would also fit linear_regression/xgboost and OLS diagnostics
every round for no benefit here - reuses experiment.py's _permutation_diagnostics() and
train.py's RF_PARAMS/_metrics directly instead of duplicating that logic).

Run directly:
    python -m src.models.backward_elimination
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.features.historical_features import FEATURE_COLS
from src.models.experiment import _permutation_diagnostics, evaluate_feature_subset
from src.models.train import RF_PARAMS, TEST_SEASON, TRAIN_SEASONS, _metrics, direct_points_baseline
from src.ingestion.db import get_connection

INTERNAL_TRAIN_SEASONS = TRAIN_SEASONS[:-1]  # 2020-21..2023-24
INTERNAL_VAL_SEASON = TRAIN_SEASONS[-1]  # 2024-25 - held back from the elimination loop entirely,
# distinct from the real TEST_SEASON (2025-26), which never gets touched until the final check.
BATCH_SIZE = 4
MIN_FEATURES = 8
PERM_N_REPEATS = 5  # lower than experiment.py's default 10 - this runs ~10 rounds, not once


def _fit_and_rank(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str],
                   target_col: str = "target_total_points") -> tuple[dict, dict]:
    """One RF fit + permutation importance on the internal validation split. Returns
    (val_metrics, {feature: {"importance_mean":..., "importance_std":...}})."""
    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df[target_col]
    X_val = val_df[feature_cols].fillna(0.0)
    y_val = val_df[target_col].to_numpy()

    rf = RandomForestRegressor(**RF_PARAMS).fit(X_train, y_train)
    val_pred = rf.predict(X_val)
    metrics = _metrics(y_val, val_pred)
    importance = _permutation_diagnostics(rf, X_val, pd.Series(y_val), n_repeats=PERM_N_REPEATS)
    return metrics, importance


def run_backward_elimination(df: pd.DataFrame, all_features: list[str]) -> dict:
    """Returns {"rounds": [{"features": [...], "dropped": [...], "val_r2": ..., "val_mae":...,
    "val_rmse": ...}], "best_round_idx": int}. Round 0 is always the full feature set (nothing
    dropped yet) - kept in the results so "don't drop anything" is a real, comparable option,
    not assumed away."""
    train_df = df[df["season"].isin(INTERNAL_TRAIN_SEASONS) & (df["gw_number"] > 1)].dropna(
        subset=["target_total_points"]
    )
    val_df = df[(df["season"] == INTERNAL_VAL_SEASON) & (df["gw_number"] > 1)].dropna(
        subset=["target_total_points"]
    )

    current = list(all_features)
    rounds = []
    round_idx = 0
    while True:
        metrics, importance = _fit_and_rank(train_df, val_df, current)
        rounds.append({
            "features": list(current),
            "dropped": [],
            "val_r2": metrics["r2"], "val_mae": metrics["mae"], "val_rmse": metrics["rmse"],
        })
        print(f"Round {round_idx}: {len(current)} features, val R²={metrics['r2']:.4f}")

        if len(current) <= MIN_FEATURES:
            break
        ranked = sorted(importance.items(), key=lambda kv: kv[1]["importance_mean"])
        drop = [name for name, _ in ranked[:min(BATCH_SIZE, len(current) - MIN_FEATURES)]]
        print(f"  dropping (lowest permutation importance): {drop}")
        current = [c for c in current if c not in drop]
        rounds[-1]["dropped"] = drop  # what got dropped AFTER this round's evaluation
        round_idx += 1

    best_round_idx = max(range(len(rounds)), key=lambda i: rounds[i]["val_r2"])
    return {"rounds": rounds, "best_round_idx": best_round_idx}


def _log_to_ledger(cur, model_type: str, feature_cols: list[str], metrics: dict, diagnostics: dict) -> None:
    cur.execute(
        """
        INSERT INTO model_versions (
            model_type, target, training_seasons, test_season, features,
            hyperparameters, mae, rmse, r2, artifact_path, is_experiment, diagnostics
        ) VALUES (%s, 'total_points_direct', %s, %s, %s, NULL, %s, %s, %s, NULL, true, %s)
        """,
        (
            model_type, TRAIN_SEASONS, TEST_SEASON, json.dumps(feature_cols),
            metrics["mae"], metrics["rmse"], metrics["r2"], json.dumps(diagnostics),
        ),
    )


def run() -> None:
    from src.features.historical_features import build_training_frame

    print("Loading and engineering historical features...")
    df = build_training_frame()
    df["season_points_baseline"] = direct_points_baseline(df)
    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    all_features = FEATURE_COLS + pos_cols
    print(f"{len(all_features)} candidate features. Internal validation split: "
          f"train={INTERNAL_TRAIN_SEASONS}, validate={INTERNAL_VAL_SEASON} (real TEST_SEASON "
          f"'{TEST_SEASON}' is untouched until the final check below).")

    result = run_backward_elimination(df, all_features)
    rounds = result["rounds"]
    best = rounds[result["best_round_idx"]]

    print("\n=== Elimination summary (internal validation R², not the real test season) ===")
    for i, r in enumerate(rounds):
        marker = "  <-- best" if i == result["best_round_idx"] else ""
        print(f"  round {i}: {len(r['features'])} features, val R²={r['val_r2']:.4f}{marker}")

    dropped_overall = [c for c in all_features if c not in best["features"]]
    print(f"\nBest round: {len(best['features'])}/{len(all_features)} features "
          f"(dropped {len(dropped_overall)}): {dropped_overall}")

    print(f"\nFinal check against the real held-out {TEST_SEASON} season "
          f"(one evaluation, both the winning subset and the full feature set for comparison)...")
    winning_results = evaluate_feature_subset(df, best["features"], compute_extended=False)
    full_results = evaluate_feature_subset(df, all_features, compute_extended=False)

    print(f"\n{'model':20s} {'features':>9s}  {'test R² (reduced)':>18s}  {'test R² (full)':>15s}")
    conn = get_connection()
    with conn.cursor() as cur:
        for model_type in ("baseline", "linear_regression", "random_forest", "xgboost"):
            reduced_r2 = winning_results[model_type]["test_metrics"]["r2"]
            full_r2 = full_results[model_type]["test_metrics"]["r2"]
            print(f"{model_type:20s} {len(best['features']):9d}  {reduced_r2:18.4f}  {full_r2:15.4f}")

            _log_to_ledger(
                cur, model_type, best["features"], winning_results[model_type]["test_metrics"],
                {
                    "source": "backward_elimination",
                    "internal_validation_rounds": [
                        {"n_features": len(r["features"]), "val_r2": r["val_r2"]} for r in rounds
                    ],
                    "dropped_features": dropped_overall,
                    "compared_to_full_feature_set_test_r2": full_r2,
                },
            )
            _log_to_ledger(
                cur, model_type, all_features, full_results[model_type]["test_metrics"],
                {"source": "backward_elimination_full_feature_reference"},
            )
    conn.commit()
    print(f"\nLogged {2 * 4} rows to model_versions (is_experiment=true) - reduced set + full "
          f"set, both model types, visible in the Model Lab ledger.")
    print(
        "\nNOTE: this only picked features; it did not change the deployed model. Review the "
        "numbers above before adopting the reduced feature set as the new default in "
        "historical_features.FEATURE_COLS."
    )


if __name__ == "__main__":
    run()
