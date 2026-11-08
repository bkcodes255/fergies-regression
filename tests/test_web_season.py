"""The Season page's exporter, with the two queries patched (no database)."""
import json
import math
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.web_data import season_page

MANAGER = pd.DataFrame({
    "event_id": [1, 2, 3], "points": [50, 72, 61], "total_points": [50, 122, 183],
    "overall_rank": [900000, 500000, 450000], "points_on_bench": [4, 9, 2],
})
COMPARISON = pd.DataFrame({
    "event_id": [1, 2, 3], "gw_name": ["Gameweek 1", "Gameweek 2", "Gameweek 3"],
    "actual_points": [50, 72, 61],
    "model_predicted_points": [None, 55.0, 60.0], "epl_predicted_points": [None, 58.0, 52.0],
    "gw_average": [48.0, 50.0, 55.0], "gw_highest": [120, 130, 125], "percentile": [60.0, 12.0, 30.0],
})


def build(manager=MANAGER, comparison=COMPARISON, entry_id=1):
    with mock.patch.object(season_page, "load_manager_gameweeks", return_value=manager), \
         mock.patch.object(season_page, "load_gw_comparison", return_value=comparison):
        return season_page.payload(object(), "2026-27", entry_id)


class SeasonPayload(unittest.TestCase):
    def test_summary(self):
        s = build()["summary"]
        self.assertEqual(s["total_points"], 183)
        self.assertEqual(s["overall_rank"], 450000)
        self.assertEqual(s["best_gw"], {"gw": 2, "points": 72})
        self.assertEqual(s["bench_points"], 15)
        self.assertEqual(s["gameweeks_played"], 3)
        self.assertEqual(s["average_points"], 61.0)

    def test_missing_predictions_are_none_not_zero(self):
        rows = build()["gameweeks"]
        self.assertIsNone(rows[0]["model"])
        self.assertIsNone(rows[0]["fpl"])
        self.assertIsNone(rows[0]["delta"])
        self.assertEqual(rows[1]["delta"], 17.0)

    def test_accuracy(self):
        a = build()["accuracy"]
        self.assertEqual(a["weeks"], 2)
        self.assertEqual(a["beats"], 1)  # GW2 model error 17 vs FPL 14: no; GW3 1 vs 9: yes
        self.assertEqual(a["model_error"], 9.0)
        self.assertEqual(a["fpl_error"], 11.5)
        self.assertEqual(a["biggest_miss"], {"gw": 2, "predicted": 55, "actual": 72, "gap": 17})

    def test_no_accuracy_when_no_week_has_both(self):
        c = COMPARISON.iloc[:1]
        self.assertIsNone(build(MANAGER.iloc[:1], c)["accuracy"])

    def test_empty_states(self):
        for p in (build(entry_id=None), build(entry_id=0), build(manager=MANAGER.iloc[0:0])):
            self.assertIsNone(p["summary"])
            self.assertEqual(p["gameweeks"], [])
            self.assertIsNone(p["accuracy"])

    def test_json_safe_no_nan(self):
        json.dumps(build(), allow_nan=False)

    def test_payload_has_no_entry_id(self):
        self.assertNotIn("entry", json.dumps(build()))

    def test_sample_matches_real_shape(self):
        sample = json.loads((Path(__file__).parents[1] / "web/src/data/sample-season.json").read_text())
        real = build()
        self.assertEqual(set(sample), set(real))
        self.assertEqual(set(sample["summary"]), set(real["summary"]))
        self.assertEqual(set(sample["gameweeks"][0]), set(real["gameweeks"][0]))
        self.assertEqual(set(sample["accuracy"]), set(real["accuracy"]))
        self.assertEqual(set(sample["accuracy"]["biggest_miss"]), set(real["accuracy"]["biggest_miss"]))
        self.assertFalse(any(isinstance(v, float) and math.isnan(v) for v in sample["gameweeks"][1].values()))


if __name__ == "__main__":
    unittest.main()
