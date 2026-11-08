"""The Optimal Squad exporter, on a synthetic player frame (no database)."""
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.web_data import optimal_squad_page as page

SAMPLE = Path(__file__).resolve().parents[1] / "web" / "src" / "data" / "sample-optimal-squad.json"
TEAMS = [f"T{i:02d}" for i in range(20)]
PER_TEAM = {"GKP": 3, "DEF": 12, "MID": 14, "FWD": 6}  # 35 a side, 700 players


def synthetic_rankings(seed: int = 1, per_team: dict | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, code = [], 1
    for team in TEAMS:
        for pos, n in (per_team or PER_TEAM).items():
            for _ in range(n):
                base = {"GKP": 3.0, "DEF": 3.2, "MID": 3.6, "FWD": 3.8}[pos]
                pts = float(rng.gamma(1.6, base / 1.6 * 0.8))
                price = round(float(np.clip(3.9 + pts * 2.1 + rng.normal(0, 0.7), 4.0, 14.5)) * 2) / 2
                rows.append({
                    "player_code": code, "web_name": f"Player{code}", "position": pos, "team": team, "price": price,
                    "status": "a" if rng.random() < 0.9 else "i", "predicted_points": round(pts, 2),
                    "horizon_points": round(pts * rng.uniform(3.5, 5.5), 2), "predicted_gw": 8,
                })
                code += 1
    return pd.DataFrame(rows)


class OptimalSquadExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rankings = synthetic_rankings(per_team={"GKP": 2, "DEF": 4, "MID": 4, "FWD": 3})
        cls.budgets = [82.0, 90.0, 100.0]
        cls.data = page.build(cls.rankings, "2026-27", cls.budgets)

    def test_budget_grid(self):
        self.assertEqual(len(page.BUDGETS), 41)
        self.assertEqual((page.BUDGETS[0], page.BUDGETS[-1]), (80.0, 100.0))

    def test_shape_and_legal_squads(self):
        for key in page.HORIZONS:
            sol = self.data["solutions"][key]
            self.assertEqual(len(sol["by_budget"]), len(self.budgets))
            for budget, idx in zip(self.budgets, sol["by_budget"]):
                self.assertIsNotNone(idx)
                s = sol["squads"][idx]
                self.assertEqual(len(s["xi"]), 11)
                self.assertEqual(len(s["bench"]), 4)
                self.assertIn(s["cap"], s["xi"])
                self.assertLessEqual(s["spend"], budget + 0.05)
                pos = [self.data["players"][str(c)]["pos"] for c in s["xi"] + s["bench"]]
                self.assertEqual({p: pos.count(p) for p in set(pos)}, {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3})

    def test_each_distinct_squad_stored_once(self):
        for sol in self.data["solutions"].values():
            keys = [(tuple(s["xi"]), tuple(s["bench"]), s["cap"]) for s in sol["squads"]]
            self.assertEqual(len(keys), len(set(keys)))

    def test_same_squad_is_shared_between_budgets(self):
        # cheap players: the budget never binds, so every budget gets the one squad
        data = page.build(self.rankings.assign(price=4.0), "2026-27", [99.0, 99.5, 100.0])
        by = data["solutions"]["next"]["by_budget"]
        self.assertEqual(len(set(by)), len(data["solutions"]["next"]["squads"]))
        self.assertEqual(len(data["solutions"]["next"]["squads"]), 1)

    def test_players_cover_every_squad_member(self):
        for sol in self.data["solutions"].values():
            for s in sol["squads"]:
                for c in s["xi"] + s["bench"]:
                    self.assertIn(str(c), self.data["players"])

    def test_infeasible_budget_is_null_not_a_failure(self):
        data = page.build(self.rankings, "2026-27", [20.0, 100.0])
        for sol in data["solutions"].values():
            self.assertIsNone(sol["by_budget"][0])
            self.assertIsNotNone(sol["by_budget"][1])

    def test_a_solver_error_is_contained_to_that_option(self):
        real = page.build_optimal_squad
        calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return real(*a, **k)

        with mock.patch.object(page, "build_optimal_squad", flaky):
            data = page.build(self.rankings, "2026-27", [90.0, 100.0])
        # budgets are solved highest first, so the first call was 100.0
        self.assertIsNone(data["solutions"]["next"]["by_budget"][1])
        self.assertIsNotNone(data["solutions"]["next"]["by_budget"][0])
        self.assertIsNotNone(data["solutions"]["window"]["by_budget"][1])

    def test_payload_json_serialisable_and_has_no_entry_data(self):
        text = json.dumps(self.data, allow_nan=False)
        self.assertNotIn("entry", text)
        self.assertLess(len(text), 300_000)

    def test_payload_signature_reads_no_squad(self):
        with mock.patch.object(page.queries, "load_player_rankings", return_value=self.rankings), \
                mock.patch.object(page, "with_horizon", side_effect=lambda conn, r, s: r):
            out = page.payload(object(), "2026-27")
        self.assertEqual(out["season"], "2026-27")


class SampleFile(unittest.TestCase):
    def test_sample_has_the_real_shape(self):
        sample = json.loads(SAMPLE.read_text())
        real = page.build(synthetic_rankings(per_team={"GKP": 2, "DEF": 4, "MID": 4, "FWD": 3}), "2026-27", [90.0])
        self.assertEqual(set(sample), set(real))
        for key in page.HORIZONS:
            self.assertEqual(set(sample["solutions"][key]), {"squads", "by_budget"})
            self.assertEqual(set(sample["solutions"][key]["squads"][0]), set(real["solutions"][key]["squads"][0]))
            self.assertEqual(len(sample["solutions"][key]["by_budget"]), len(sample["budgets"]))
            for idx in sample["solutions"][key]["by_budget"]:
                self.assertTrue(idx is None or idx < len(sample["solutions"][key]["squads"]))
        self.assertEqual(set(next(iter(sample["players"].values()))), set(next(iter(real["players"].values()))))


if __name__ == "__main__":
    unittest.main()
