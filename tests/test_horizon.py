import unittest

import numpy as np
import pandas as pd

from src.recommendations.horizon import compute_horizon_points, difficulty_to_multiplier, fixture_grade


class HorizonTests(unittest.TestCase):
    def test_difficulty_multiplier_is_scaled_and_clipped(self):
        result = difficulty_to_multiplier(pd.Series([-10.0, -0.5, 0.0, 0.5, 10.0]))
        np.testing.assert_allclose(result, [1.5, 1.2, 1.0, 0.8, 0.6])

    def test_fixture_grade_boundaries_and_blank_values(self):
        scores = [-1, -0.35, -0.349, -0.1, -0.099, 0.1, 0.101, 0.35, 0.351]
        self.assertEqual([fixture_grade(x) for x in scores], [1, 1, 2, 2, 3, 3, 4, 4, 5])
        self.assertIsNone(fixture_grade(None))
        self.assertIsNone(fixture_grade(float("nan")))

    def test_horizon_points_apply_difficulty_and_count_double_gameweeks(self):
        rankings = pd.DataFrame({"player_code": [1, 2, 3], "predicted_points": [10.0, 8.0, 6.0]})
        fixtures = pd.DataFrame([
            (1, 1, 0.25), (1, 2, 0.5), (1, 2, -0.5), (2, 1, -0.5),
        ], columns=["player_code", "event_id", "fixture_difficulty_score"])

        result = compute_horizon_points(rankings, fixtures, horizon=2).set_index("player_code")

        self.assertEqual(result.loc[1, "horizon_fixtures"], 3)
        self.assertAlmostEqual(result.loc[1, "horizon_points"], 29.0)
        self.assertEqual(result.loc[2, "horizon_fixtures"], 1)
        self.assertAlmostEqual(result.loc[2, "horizon_points"], 9.6)
        self.assertEqual(result.loc[3, "horizon_fixtures"], 0)
        self.assertAlmostEqual(result.loc[3, "horizon_points"], 0.0)

    def test_no_fixtures_use_flat_horizon_fallback(self):
        rankings = pd.DataFrame({"player_code": [1], "predicted_points": [10.0]})
        result = compute_horizon_points(rankings, pd.DataFrame(), horizon=3)
        self.assertEqual(result.loc[0, "horizon_points"], 30.0)
        self.assertEqual(result.loc[0, "horizon_fixtures"], 3)


if __name__ == "__main__":
    unittest.main()
