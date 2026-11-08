"""The Model Lab results script, on a small synthetic frame (no database, no real data)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.features.historical_features import FEATURE_COLS
from src.models.train import TEST_SEASON, TRAIN_SEASONS

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_model_lab.py"
spec = importlib.util.spec_from_file_location("build_model_lab", SCRIPT)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def fake_frame(n_per_season: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    seasons = list(TRAIN_SEASONS) + [TEST_SEASON]
    n = n_per_season * len(seasons)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    df["season"] = np.repeat(seasons, n_per_season)
    df["gw_number"] = rng.integers(1, 38, size=n)
    df["season_points_per90_avg"] = rng.uniform(0, 8, size=n)
    df["season_minutes_avg"] = rng.uniform(0, 90, size=n)
    for p in ("GK", "DEF", "MID", "FWD"):
        df[f"pos_{p}"] = False
    df.loc[:, "pos_MID"] = True
    df["target_total_points"] = df["season_points_per90_avg"] * 0.5 + df["price"] * 0.3 + rng.normal(size=n)
    return df


class ModelLab(unittest.TestCase):
    def test_groups_cover_every_feature_once(self):
        feats = list(FEATURE_COLS) + ["pos_GK", "pos_MID"]
        groups = lab.group_features(feats)
        flat = [c for cols in groups.values() for c in cols]
        self.assertCountEqual(flat, feats)

    def test_run_writes_one_experiment_per_group(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(lab, "build_training_frame", fake_frame):
            out = Path(d) / "ml.json"
            self.assertEqual(lab.main(["--sample-frac", "0.5", "--out", str(out)]), 0)
            data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(data["test_season"], TEST_SEASON)
        self.assertEqual(data["sample_frac"], 0.5)
        names = [g["name"] for g in data["groups"]]
        self.assertEqual([e["group"] for e in data["experiments"]], names)
        self.assertEqual(set(data["reference"]), {"baseline", "linear_regression", "random_forest", "xgboost"})
        for e in data["experiments"]:
            for mt in ("linear_regression", "random_forest", "xgboost"):
                m = e["models"][mt]
                self.assertIn("delta_r2", m)
                self.assertTrue(0 <= m["p_value"] <= 1)
            self.assertNotIn("delta_r2", e["models"]["baseline"])
        self.assertEqual(len(data["importances"]["random_forest"]), lab.TOP_IMPORTANCES)
        self.assertGreater(data["n_train"], 0)

    def test_bad_sample_frac_rejected(self):
        with self.assertRaises(SystemExit):
            lab.main(["--sample-frac", "0"])


if __name__ == "__main__":
    unittest.main()
