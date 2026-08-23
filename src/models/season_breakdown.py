"""Per-gameweek accuracy breakdown for the deployed total_points_direct model(s) across the
full held-out 2025-26 season. A single pooled R²/MAE (the number quoted everywhere else in this
project) doesn't say whether the model gets better or worse as the season progresses, has
particular rough patches, or is roughly uniform throughout - this breaks the exact same
held-out evaluation train.py already does down by gameweek instead of pooling all rows.

Uses the ALREADY-TRAINED deployed artifacts (no retraining): loads whichever model_versions row
is currently best (lowest RMSE) for random_forest and xgboost on total_points_direct, applies
each to the real per-gameweek test rows. Also includes the naive season-average baseline for
the same floor-reference every other evaluation in this project shows.

Run directly:
    python -m src.models.season_breakdown
"""
from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.historical_features import build_training_frame
from src.ingestion.db import get_connection
from src.models.train import TEST_SEASON, direct_points_baseline

MODEL_TYPES = ("baseline", "random_forest", "xgboost")
MIN_ROWS_PER_GW = 20  # skip gameweeks with too few rows for a stable R² (e.g. a mostly-blank GW)


def _load_model_row(cur, model_type: str):
    cur.execute(
        """
        SELECT model_id, features, artifact_path FROM model_versions
        WHERE target = 'total_points_direct' AND model_type = %s AND artifact_path IS NOT NULL
        ORDER BY rmse ASC, trained_at DESC LIMIT 1
        """,
        (model_type,),
    )
    return cur.fetchone()


def run() -> None:
    print("Loading and engineering historical features...")
    df = build_training_frame()
    df["season_points_baseline"] = direct_points_baseline(df)
    test_df = df[(df["season"] == TEST_SEASON) & (df["gw_number"] > 1)].dropna(subset=["target_total_points"])
    y_test = test_df["target_total_points"]
    gws = test_df["gw_number"]

    conn = get_connection()
    predictions = {"baseline": test_df["season_points_baseline"].fillna(0.0)}
    with conn.cursor() as cur:
        for model_type in ("random_forest", "xgboost"):
            row = _load_model_row(cur, model_type)
            if row is None:
                print(f"No trained {model_type} model found - skipping.")
                continue
            model_id, feature_cols, artifact_path = row
            model = joblib.load(artifact_path)
            X_test = test_df[feature_cols].fillna(0.0)
            predictions[model_type] = pd.Series(model.predict(X_test), index=test_df.index)
            print(f"{model_type}: model_id={model_id}, {len(feature_cols)} features, {artifact_path}")
    conn.close()

    rows = []
    for gw in sorted(gws.unique()):
        mask = gws == gw
        n = int(mask.sum())
        if n < MIN_ROWS_PER_GW:
            continue
        row = {"GW": int(gw), "n_players": n}
        for model_type in predictions:
            yt, yp = y_test[mask], predictions[model_type][mask]
            row[f"{model_type}_r2"] = r2_score(yt, yp)
            row[f"{model_type}_mae"] = mean_absolute_error(yt, yp)
            row[f"{model_type}_rmse"] = mean_squared_error(yt, yp) ** 0.5
        rows.append(row)
    breakdown = pd.DataFrame(rows)

    pd.set_option("display.width", 160)
    print(f"\n=== Per-gameweek breakdown, {TEST_SEASON}, held-out (never trained on) ===")
    print(breakdown.round(3).to_string(index=False))

    print("\n=== Season-pooled (for comparison to the single number quoted elsewhere) ===")
    for model_type in predictions:
        yt, yp = y_test, predictions[model_type]
        print(f"  {model_type:14s} R²={r2_score(yt, yp):.4f}  MAE={mean_absolute_error(yt, yp):.4f}  "
              f"RMSE={mean_squared_error(yt, yp)**0.5:.4f}")

    fig, (ax_r2, ax_mae) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = {"baseline": "gray", "random_forest": "tab:blue", "xgboost": "tab:orange"}
    for model_type in predictions:
        ax_r2.plot(breakdown["GW"], breakdown[f"{model_type}_r2"], marker="o", label=model_type,
                   color=colors.get(model_type))
        ax_mae.plot(breakdown["GW"], breakdown[f"{model_type}_mae"], marker="o", label=model_type,
                    color=colors.get(model_type))
    ax_r2.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax_r2.set_ylabel("R² (per gameweek)")
    ax_r2.set_title(f"total_points_direct accuracy by gameweek, {TEST_SEASON} (held-out, never trained on)")
    ax_r2.legend()
    ax_mae.set_ylabel("MAE (per gameweek)")
    ax_mae.set_xlabel("Gameweek")
    fig.tight_layout()
    out_path = "notebooks/season_breakdown_2025-26.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nChart saved to {out_path}")


if __name__ == "__main__":
    run()
