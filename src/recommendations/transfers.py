"""Transfer suggestions: for each squad player, find the best available replacement.

Deliberately simple (single-gameweek horizon, no multi-week planning) - our predictions table
only has next-gameweek projections right now, since the model's rolling features would need
real intervening gameweeks to project further out honestly. Multi-gameweek transfer planning
(the plan's "5-GW transfer horizon" section) needs iterated prediction, not built yet.
"""
from __future__ import annotations

import pandas as pd

MAX_PER_TEAM = 3


def suggest_transfers(squad: pd.DataFrame, rankings: pd.DataFrame, bank: float, top_n: int = 3) -> pd.DataFrame:
    """
    squad: one row per owned player - player_code, web_name, position, team, price, predicted_points
    rankings: full player pool with the same columns, plus status ('a' = available)
    bank: money in the bank, in £m (budget available beyond selling the player being replaced)

    Returns one row per (squad player, candidate replacement) with a positive net gain,
    top `top_n` candidates per squad player, sorted by net gain descending.
    """
    owned_codes = set(squad["player_code"])
    team_counts = squad["team"].value_counts().to_dict()

    suggestions = []
    for _, current in squad.iterrows():
        budget = current["price"] + bank
        same_position = rankings[
            (rankings["position"] == current["position"])
            & (~rankings["player_code"].isin(owned_codes))
            & (rankings["price"] <= budget)
            & (rankings["status"] == "a")
        ].copy()

        def team_ok(row) -> bool:
            count_if_bought = team_counts.get(row["team"], 0) + (0 if row["team"] == current["team"] else 1)
            return count_if_bought <= MAX_PER_TEAM

        same_position = same_position[same_position.apply(team_ok, axis=1)]
        same_position["net_gain"] = same_position["predicted_points"] - current["predicted_points"]
        candidates = same_position[same_position["net_gain"] > 0].sort_values("net_gain", ascending=False).head(top_n)

        for _, cand in candidates.iterrows():
            suggestions.append({
                "sell": current["web_name"], "sell_position": current["position"],
                "sell_predicted": current["predicted_points"], "sell_price": current["price"],
                "buy": cand["web_name"], "buy_predicted": cand["predicted_points"],
                "buy_price": cand["price"], "net_gain": round(cand["net_gain"], 3),
            })

    if not suggestions:
        return pd.DataFrame(columns=[
            "sell", "sell_position", "sell_predicted", "sell_price",
            "buy", "buy_predicted", "buy_price", "net_gain",
        ])
    return pd.DataFrame(suggestions).sort_values("net_gain", ascending=False).reset_index(drop=True)
