import unittest

import pandas as pd

from src.recommendations.squad_optimizer import VALID_FORMATIONS, best_starting_xi


class SquadOptimizerTests(unittest.TestCase):
    def test_valid_formations_are_the_eight_fpl_shapes(self):
        self.assertEqual(len(VALID_FORMATIONS), 8)
        self.assertTrue(all(sum(formation) == 10 for formation in VALID_FORMATIONS))

    def test_selects_highest_scoring_formation_and_one_of_two_gkps(self):
        rows = [
            ("g1", "GKP", 5), ("g2", "GKP", 4),
            ("d1", "DEF", 8), ("d2", "DEF", 7), ("d3", "DEF", 6), ("d4", "DEF", 1), ("d5", "DEF", 1),
            ("m1", "MID", 10), ("m2", "MID", 9), ("m3", "MID", 8), ("m4", "MID", 7), ("m5", "MID", 3),
            ("f1", "FWD", 12), ("f2", "FWD", 11), ("f3", "FWD", 1),
        ]
        xi, formation = best_starting_xi(pd.DataFrame(rows, columns=["name", "position", "predicted_points"]))
        self.assertEqual(formation, (3, 5, 2))
        self.assertEqual(len(xi), 11)
        self.assertEqual((xi["position"] == "GKP").sum(), 1)
        self.assertEqual(xi["predicted_points"].sum(), 86)
        self.assertEqual(set(xi["name"]), {"g1", "d1", "d2", "d3", "m1", "m2", "m3", "m4", "m5", "f1", "f2"})

    def test_uses_requested_value_column(self):
        rows = [("g1", "GKP", 10, 10), ("g2", "GKP", 1, 1)]
        rows += [(f"d{i}", "DEF", p, h) for i, (p, h) in enumerate([(10, 10), (9, 9), (8, 8), (8, 8), (8, 8)], 1)]
        rows += [(f"m{i}", "MID", p, h) for i, (p, h) in enumerate([(20, 6), (19, 5), (19, 4), (20, 3), (40, 2)], 1)]
        rows += [(f"f{i}", "FWD", p, h) for i, (p, h) in enumerate([(20, 20), (19, 19), (18, 18)], 1)]
        squad = pd.DataFrame(rows, columns=["name", "position", "predicted_points", "horizon_points"])

        horizon_xi, horizon_formation = best_starting_xi(squad, value_col="horizon_points")
        predicted_xi, predicted_formation = best_starting_xi(squad)

        self.assertEqual(horizon_formation, (5, 2, 3))
        self.assertEqual(predicted_formation, (3, 5, 2))
        self.assertEqual(set(horizon_xi.loc[horizon_xi["position"] == "FWD", "name"]), {"f1", "f2", "f3"})
        self.assertNotEqual(set(horizon_xi["name"]), set(predicted_xi["name"]))


if __name__ == "__main__":
    unittest.main()
