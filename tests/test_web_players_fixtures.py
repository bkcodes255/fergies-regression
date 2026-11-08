"""The Players and Fixtures exporters, with the query functions patched (no database)."""
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.web_data import fixtures_page, players_page

DATA = Path(__file__).resolve().parent.parent / "web" / "src" / "data"


def rankings() -> pd.DataFrame:
    return pd.DataFrame({
        "player_code": [1, 2, 3, 4],
        "web_name": ["Low", "High", "Gone", "Blank"],
        "team": ["AAA", "BBB", "AAA", "CCC"],
        "position": ["DEF", "FWD", "MID", "MID"],
        "price": [5.04, 10.0, 6.0, 4.5],
        "predicted_points": [3.04, 7.46, 9.0, np.nan],
        "horizon_points": [12.34, 30.0, 40.0, np.nan],
        "floor_points": [1.0, 2.0, 3.0, np.nan],
        "ceiling_points": [6.54, 12.0, 14.0, np.nan],
        "p_haul_10plus": [0.234, 0.5, 0.9, np.nan],
        "predicted_value": [0.6031, 0.746, 1.5, np.nan],
        "total_points": [40, 90, 70, 0],
        "selected_by_percent": [12.34, 50.0, 60.0, 0.25],
        "status": ["a", "a", "i", "a"],
        "predicted_gw": [12, 12, 12, 12],
    })


class PlayersExport(unittest.TestCase):
    def run_payload(self):
        with mock.patch("src.web_data.players_page.queries.load_player_rankings", return_value=rankings()), \
             mock.patch("src.web_data.players_page.with_horizon", side_effect=lambda conn, df, season: df):
            return players_page.payload(object(), "2026-27")

    def test_only_available_players_best_first_and_rounded(self):
        out = self.run_payload()
        self.assertEqual([p["name"] for p in out["players"]], ["High", "Low", "Blank"])
        low = out["players"][1]
        self.assertEqual((low["price"], low["pred"], low["horizon"], low["ceil"]), (5.0, 3.0, 12.3, 6.5))
        self.assertEqual((low["haul"], low["value"], low["owned"], low["pts"]), (23, 0.6, 12.3, 40))
        self.assertEqual(out["predicted_gw"], 12)

    def test_missing_predictions_are_null_and_json_is_strict(self):
        out = self.run_payload()
        blank = out["players"][2]
        self.assertIsNone(blank["pred"])
        self.assertIsNone(blank["haul"])
        json.dumps(out, allow_nan=False)

    def test_no_rankings_gives_an_empty_page(self):
        empty = rankings().iloc[0:0]
        with mock.patch("src.web_data.players_page.queries.load_player_rankings", return_value=empty):
            out = players_page.payload(object(), "2026-27")
        self.assertEqual(out["players"], [])
        self.assertIsNone(out["predicted_gw"])

    def test_nothing_squad_shaped_in_the_payload(self):
        self.assertNotIn("entry", json.dumps(self.run_payload()).lower())


def grid() -> pd.DataFrame:
    rows = [  # team, opponent, is_home, gw, score
        ("AAA", "BBB", True, 12, -0.5), ("AAA", "CCC", False, 14, -0.4), ("AAA", "BBB", True, 14, 0.2),
        ("BBB", "AAA", False, 12, 0.5), ("BBB", "CCC", True, 13, 0.4),
    ]
    return pd.DataFrame(rows, columns=["team", "opponent", "is_home", "event_id", "fixture_difficulty_score"])


class FixturesExport(unittest.TestCase):
    def run_payload(self, g=None, deadline=None):
        deadline = {"event_id": 12, "name": "Gameweek 12", "deadline_time": None} if deadline is None else deadline
        g = grid() if g is None else g
        with mock.patch("src.web_data.fixtures_page.queries.load_next_deadline", return_value=deadline), \
             mock.patch("src.web_data.fixtures_page.queries.load_fixture_grid", return_value=g) as lg:
            out = fixtures_page.payload(object(), "2026-27")
        return out, lg

    def test_grid_shape_blank_and_double_gameweeks(self):
        out, lg = self.run_payload()
        self.assertEqual(lg.call_args.args[2:], (12, 5))
        self.assertEqual(out["gws"], [12, 13, 14])
        aaa = out["teams"][0]
        self.assertEqual(aaa["team"], "AAA")  # easiest run first
        self.assertIsNone(aaa["cells"][1])  # blank in GW13
        double = aaa["cells"][2]
        self.assertEqual([f["opp"] for f in double["fx"]], ["CCC", "BBB"])
        self.assertEqual(double["grade"], 2)  # mean -0.1 -> grade 2; the two fixtures keep their own grades
        self.assertEqual([f["grade"] for f in double["fx"]], [1, 4])

    def test_run_rating_bands(self):
        self.assertEqual(fixtures_page.run_rating(-0.2), "Easy")
        self.assertEqual(fixtures_page.run_rating(0.0), "Mixed")
        self.assertEqual(fixtures_page.run_rating(0.25), "Brutal")
        out, _ = self.run_payload()
        self.assertEqual([t["run"] for t in out["teams"]], ["Easy", "Brutal"])

    def test_no_deadline_starts_at_gw2_and_empty_grid_is_fine(self):
        with mock.patch("src.web_data.fixtures_page.queries.load_next_deadline", return_value=None), \
             mock.patch("src.web_data.fixtures_page.queries.load_fixture_grid", return_value=grid().iloc[0:0]) as lg:
            out = fixtures_page.payload(object(), "2026-27")
        self.assertEqual(lg.call_args.args[2], 2)
        self.assertEqual(out["teams"], [])
        json.dumps(out, allow_nan=False)

    def test_never_reads_a_squad(self):
        with mock.patch("src.web_data.queries.load_squad") as squad:
            self.run_payload()
        squad.assert_not_called()


class Samples(unittest.TestCase):
    def test_sample_shapes_match_the_exporters(self):
        players = json.loads((DATA / "sample-players.json").read_text())
        fixtures = json.loads((DATA / "sample-fixtures.json").read_text())
        self.assertEqual(set(players), {"predicted_gw", "horizon", "players"})
        keys = {"name", "pos", "team", "price", "pred", "horizon", "floor", "ceil", "haul", "value", "pts", "owned"}
        for p in players["players"]:
            self.assertEqual(set(p), keys)
        self.assertEqual(set(fixtures), {"start_gw", "horizon", "gws", "teams"})
        for t in fixtures["teams"]:
            self.assertEqual(set(t), {"team", "avg", "run", "cells"})
            self.assertEqual(len(t["cells"]), len(fixtures["gws"]))


if __name__ == "__main__":
    unittest.main()
