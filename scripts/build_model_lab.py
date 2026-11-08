"""Builds the Model Lab results page's data: web/src/data/model-lab.json.

Run by hand, offline, not in CI and not by src/web_data/export.py. It loads six seasons of
history and fits three models per experiment, which a Cloudflare build cannot do. Commit the
JSON it writes; the page shows it as a fixed run with its date.

    python scripts/build_model_lab.py [--sample-frac 0.25] [--out web/src/data/model-lab.json]

Experiments: the all-features reference, then each feature group removed in turn (same grouping
as the old Streamlit tab's group_features). Each is scored with experiment.evaluate_feature_subset
on the deployed model's own split and hyperparameters, and tested against the all-features run
with the tab's paired bootstrap. Nothing is written except --out: no joblib dumps, no database
writes. Experiments run one at a time and memory is freed between them.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.features.historical_features import FEATURE_COLS, build_training_frame  # noqa: E402
from src.models.experiment import (  # noqa: E402
    MODEL_TYPES, OVERFIT_GAP_THRESHOLD, evaluate_feature_subset, paired_bootstrap_p_value,
)
from src.models.train import TEST_SEASON, TRAIN_SEASONS, direct_points_baseline  # noqa: E402

DEFAULT_OUT = ROOT / "web" / "src" / "data" / "model-lab.json"
TARGET = "target_total_points"
FITTED = tuple(m for m in MODEL_TYPES if m != "baseline")
TOP_IMPORTANCES = 15


def group_features(all_features: list[str]) -> dict[str, list[str]]:
    """Copy of dashboard/app.py's group_features (that file imports Streamlit)."""
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


def _log(msg: str) -> None:
    print(msg, flush=True)


def _num(x, digits: int = 4):
    return None if x is None or not np.isfinite(x) else round(float(x), digits)


def _fit_flag(model_type: str, res: dict, naive_r2: float) -> str | None:
    """The tab's rule of thumb: overfit gap above 0.15, or no better than the naive baseline."""
    if model_type == "baseline":
        return None
    if res["overfit_gap"] is not None and res["overfit_gap"] > OVERFIT_GAP_THRESHOLD:
        return "possible_overfit"
    if res["test_metrics"]["r2"] <= naive_r2:
        return "possible_underfit"
    return "ok"


def _summarise(results: dict, ref: dict | None, ref_r2: dict | None) -> dict:
    """One experiment's numbers, JSON-ready. ref is the all-features run, for delta and p-value."""
    naive_r2 = results["baseline"]["test_metrics"]["r2"]
    out = {}
    for mt in MODEL_TYPES:
        res = results[mt]
        tm = res["test_metrics"]
        row = {
            "r2": _num(tm["r2"]), "mae": _num(tm["mae"]), "rmse": _num(tm["rmse"]),
            "train_r2": _num(res["train_metrics"]["r2"]) if res["train_metrics"] else None,
            "overfit_gap": _num(res["overfit_gap"]),
            "r2_ci": [_num(res["r2_ci"][0]), _num(res["r2_ci"][1])],
            "fit": _fit_flag(mt, res, naive_r2),
        }
        if ref is not None and mt != "baseline":
            row["delta_r2"] = _num(tm["r2"] - ref_r2[mt])
            row["p_value"] = _num(
                min(1.0, paired_bootstrap_p_value(results["_y_test"], ref[mt]["y_test_pred"], res["y_test_pred"]))
            )
        out[mt] = row
    return out


def _importances(results: dict, features: list[str]) -> dict:
    out = {}
    for mt in ("random_forest", "xgboost"):
        imp = np.asarray(results[mt]["model"].feature_importances_, dtype=float)
        order = np.argsort(imp)[::-1][:TOP_IMPORTANCES]
        out[mt] = [{"feature": features[i], "importance": _num(imp[i], 5)} for i in order]
    return out


def _slim(results: dict) -> dict:
    """Keep only what later p-value tests need from the reference run; drop the fitted models."""
    slim = {"_y_test": results["_y_test"]}
    for mt in MODEL_TYPES:
        slim[mt] = {"y_test_pred": results[mt]["y_test_pred"], "test_metrics": results[mt]["test_metrics"]}
    return slim


def build_payload(df, all_features: list[str], sample_frac: float | None = None) -> dict:
    groups = group_features(all_features)
    gen = datetime.now(timezone.utc).replace(microsecond=0)

    train_mask = df["season"].isin(TRAIN_SEASONS) & (df["gw_number"] > 1) & df[TARGET].notna()
    test_mask = (df["season"] == TEST_SEASON) & (df["gw_number"] > 1) & df[TARGET].notna()
    n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
    if sample_frac is not None:
        n_train = int(round(n_train * sample_frac))

    _log(f"Reference run: all {len(all_features)} features")
    results = evaluate_feature_subset(df, all_features, sample_frac=sample_frac)
    ref_summary = _summarise(results, None, None)
    importances = _importances(results, all_features)
    ref = _slim(results)
    ref_r2 = {mt: results[mt]["test_metrics"]["r2"] for mt in MODEL_TYPES}
    del results
    gc.collect()

    experiments = []
    for i, (name, cols) in enumerate(groups.items(), 1):
        kept = [c for c in all_features if c not in set(cols)]
        _log(f"[{i}/{len(groups)}] Without '{name}' ({len(cols)} removed, {len(kept)} kept)")
        if not kept:
            _log("  nothing left to fit, skipped")
            continue
        res = evaluate_feature_subset(df, kept, sample_frac=sample_frac)
        experiments.append({
            "group": name, "removed": cols, "n_kept": len(kept),
            "models": _summarise(res, ref, ref_r2),
        })
        del res
        gc.collect()

    return {
        "generated_at": gen.isoformat().replace("+00:00", "Z"),
        "test_season": TEST_SEASON,
        "train_seasons": list(TRAIN_SEASONS),
        "n_train": n_train, "n_test": n_test,
        "sample_frac": sample_frac,
        "n_features": len(all_features),
        "bootstrap_draws": 1000,
        "overfit_gap_threshold": OVERFIT_GAP_THRESHOLD,
        "groups": [{"name": n, "features": c} for n, c in groups.items()],
        "reference": ref_summary,
        "importances": importances,
        "experiments": experiments,
    }


def load_frame() -> tuple:
    _log("Building the training frame (six seasons; slow and memory-hungry)...")
    df = build_training_frame()
    df["season_points_baseline"] = direct_points_baseline(df)
    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    features = list(FEATURE_COLS) + pos_cols
    # Keep only the rows and columns the experiments use, then free the rest.
    keep = ["season", "gw_number", TARGET, "season_points_baseline"] + features
    keep = list(dict.fromkeys(keep))
    rows = df["season"].isin(list(TRAIN_SEASONS) + [TEST_SEASON]) & (df["gw_number"] > 1) & df[TARGET].notna()
    slim = df.loc[rows, keep].copy()
    del df
    gc.collect()
    _log(f"Frame ready: {len(slim)} rows, {len(features)} features.")
    return slim, features


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sample-frac", type=float, default=None,
                    help="fraction of training rows to use (0-1]; omit for the full run")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output JSON (default {DEFAULT_OUT})")
    args = ap.parse_args(argv)
    if args.sample_frac is not None and not 0 < args.sample_frac <= 1:
        ap.error("--sample-frac must be in (0, 1]")

    df, features = load_frame()
    payload = build_payload(df, features, args.sample_frac)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    _log(f"Wrote {args.out}")
    if args.sample_frac is not None:
        _log("NOTE: sampled run. Do not commit this as the real results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
