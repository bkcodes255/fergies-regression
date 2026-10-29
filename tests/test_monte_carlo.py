import unittest

import numpy as np
import pandas as pd

from src.models.monte_carlo import sample_player_points, simulate_squad, summarize_simulation


class MonteCarloTests(unittest.TestCase):
    def test_sample_has_flat_lower_tail_and_extrapolated_upper_tail(self):
        class FixedRng:
            def random(self, n):
                return np.array([0.05, 0.3, 0.7, 0.95])

        result = sample_player_points(2, 6, 10, 4, FixedRng())
        np.testing.assert_allclose(result, [2.0, 4.0, 8.0, 10.5])

    def test_sample_clips_negative_quantiles(self):
        class FixedRng:
            def random(self, n):
                return np.array([0.05, 0.5, 0.95])

        result = sample_player_points(-3, -1, 5, 3, FixedRng())
        self.assertTrue(np.all(result >= 0))
        np.testing.assert_allclose(result, [0.0, 0.0, 5.75])

    def test_simulate_excludes_bench_and_doubles_captain(self):
        squad = pd.DataFrame([
            {"floor_points": 2, "median_points": 2, "ceiling_points": 2, "multiplier": 1},
            {"floor_points": 3, "median_points": 3, "ceiling_points": 3, "multiplier": 2},
            {"floor_points": 100, "median_points": 100, "ceiling_points": 100, "multiplier": 0},
        ])
        np.testing.assert_allclose(simulate_squad(squad, n_samples=3), [8, 8, 8])

    def test_simulate_fills_missing_values_and_seed_changes_samples(self):
        squad = pd.DataFrame([{
            "floor_points": np.nan, "median_points": np.nan, "ceiling_points": np.nan,
            "predicted_points": 4, "multiplier": 1,
        }])
        first = simulate_squad(squad, n_samples=20, seed=7)
        second = simulate_squad(squad, n_samples=20, seed=7)
        other_seed = simulate_squad(squad, n_samples=20, seed=8)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other_seed))
        self.assertTrue(np.all(first >= 0))
        self.assertGreater(first.max(), 0)

    def test_summarize_simulation_separates_thresholds(self):
        result = summarize_simulation(np.array([0.0, 60.0, 70.0, 80.0, 100.0]))
        self.assertEqual(result, {"mean": 62.0, "p10": 24.0, "p50": 70.0, "p90": 92.0, "p_beat_60": 0.8, "p_beat_80": 0.4})


if __name__ == "__main__":
    unittest.main()
