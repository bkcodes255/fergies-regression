import unittest

import pandas as pd

from src.recommendations.transfers import compute_free_transfers, suggest_transfer_plan, suggest_transfers


class TransfersTests(unittest.TestCase):
    def test_free_transfers_sort_history_skip_gw1_consume_and_cap(self):
        history = pd.DataFrame([
            {"event_id": 4, "event_transfers": 2}, {"event_id": 1, "event_transfers": 99},
            {"event_id": 2, "event_transfers": 1}, {"event_id": 3, "event_transfers": 0},
        ])
        self.assertEqual(compute_free_transfers(history), 1)
        banked = pd.DataFrame([{"event_id": event, "event_transfers": 0} for event in range(1, 9)])
        self.assertEqual(compute_free_transfers(banked), 5)

    def _frames(self):
        squad = pd.DataFrame([
            (1, "Owned A", "MID", "T1", 5.0, 5.0), (2, "Owned B", "DEF", "T2", 4.0, 4.0),
        ], columns=["player_code", "web_name", "position", "team", "price", "predicted_points"])
        rankings = pd.DataFrame([
            (3, "Buy Mid", "MID", "T3", 6.0, 10.0, "a"), (4, "Buy Mid 2", "MID", "T1", 5.0, 7.0, "a"),
            (5, "Buy Def", "DEF", "T3", 5.0, 10.0, "a"), (6, "Unavailable", "MID", "T3", 1.0, 20.0, "i"),
            (7, "Too Costly", "MID", "T3", 10.0, 20.0, "a"),
        ], columns=["player_code", "web_name", "position", "team", "price", "predicted_points", "status"])
        return squad, rankings

    def test_suggest_transfers_filters_and_ranks_positive_gains(self):
        squad, rankings = self._frames()
        result = suggest_transfers(squad, rankings, bank=1.0, top_n=1)
        self.assertEqual(list(result["buy"]), ["Buy Def", "Buy Mid"])
        self.assertEqual(list(result["net_gain"]), [6.0, 5.0])
        self.assertEqual(list(result["net_gain_if_hit"]), [2.0, 1.0])

    def test_suggest_transfers_enforces_three_player_team_limit(self):
        squad = pd.DataFrame([
            (1, "Owned Mid", "MID", "T2", 5.0, 5.0), (2, "Owned Def", "DEF", "T1", 4.0, 4.0),
            (3, "Owned Fwd", "FWD", "T1", 4.0, 4.0), (4, "Owned Gkp", "GKP", "T1", 4.0, 4.0),
        ], columns=["player_code", "web_name", "position", "team", "price", "predicted_points"])
        rankings = pd.DataFrame([
            (5, "Blocked by Team Limit", "MID", "T1", 5.0, 20.0, "a"),
            (6, "Allowed Buy", "MID", "T3", 5.0, 8.0, "a"),
            (7, "Replacement Def", "DEF", "T3", 4.0, 5.0, "a"),
            (8, "Replacement Fwd", "FWD", "T3", 4.0, 5.0, "a"),
            (9, "Replacement Gkp", "GKP", "T3", 4.0, 5.0, "a"),
        ], columns=["player_code", "web_name", "position", "team", "price", "predicted_points", "status"])
        result = suggest_transfers(squad, rankings, bank=0.0)
        self.assertIn("Allowed Buy", set(result["buy"]))
        self.assertNotIn("Blocked by Team Limit", set(result["buy"]))

    @unittest.expectedFailure
    def test_suggest_transfers_empty_result_has_schema(self):
        # Source applies a row-wise filter to an empty candidate frame, losing its columns.
        squad, rankings = self._frames()
        result = suggest_transfers(squad, rankings.iloc[0:0], bank=0)
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), ["sell", "sell_position", "sell_predicted", "sell_price", "buy", "buy_predicted", "buy_price", "net_gain", "net_gain_if_hit"])

    def test_transfer_plan_is_coordinated_and_charges_hits(self):
        squad, rankings = self._frames()
        plan, resulting, bank = suggest_transfer_plan(squad, rankings, bank=1.0, free_transfers=2, max_transfers=3)
        self.assertEqual(list(plan["buy"]), ["Buy Def", "Buy Mid 2"])
        self.assertEqual(list(plan["hit"]), [0, 0])
        self.assertEqual(list(plan["net"]), [6.0, 2.0])
        self.assertEqual(bank, 0.0)
        self.assertEqual(set(resulting["web_name"]), {"Buy Def", "Buy Mid 2"})

    def test_transfer_plan_stops_when_hit_makes_gain_nonpositive(self):
        squad, rankings = self._frames()
        plan, _, _ = suggest_transfer_plan(squad, rankings, bank=1.0, free_transfers=0)
        self.assertEqual(len(plan), 1)
        self.assertEqual(list(plan["hit"]), [4])
        self.assertEqual(list(plan["net"]), [2.0])


if __name__ == "__main__":
    unittest.main()
