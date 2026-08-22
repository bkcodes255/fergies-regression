"""Transfer suggestions: for each squad player, find the best available replacement.

All comparisons are keyed off a `value_col` parameter (defaults to "predicted_points", the
single next-GW prediction) rather than hardcoding it, so the same greedy logic works for
multi-week planning too - pass value_col="horizon_points" (src.recommendations.horizon) to
compare players on a fixture-difficulty-adjusted sum over the next N gameweeks instead of just
the next one. hit/net-gain math (TRANSFER_HIT_COST) is unaffected either way: a transfer's hit
is paid once regardless of the horizon used to justify it.
"""
from __future__ import annotations

import pandas as pd

MAX_PER_TEAM = 3
TRANSFER_HIT_COST = 4  # points deducted for a transfer beyond your free ones
FREE_TRANSFER_CAP = 5  # 2026/27 rule change: up to 5 banked free transfers (was 2)


def compute_free_transfers(manager_gameweeks: pd.DataFrame, cap: int = FREE_TRANSFER_CAP) -> int:
    """manager_gameweeks: one entry's manager_gameweeks rows (needs event_id, event_transfers),
    any order. Returns free transfers available for the gameweek AFTER the last one played.

    GW1 doesn't consume a transfer (it's the initial squad pick, not a transfer) - everyone
    starts with 1 free transfer available for GW2, independent of what happened in GW1.
    """
    free_transfers = 1
    for _, row in manager_gameweeks.sort_values("event_id").iterrows():
        if row["event_id"] < 2:
            continue
        made = int(row["event_transfers"] or 0)
        free_transfers = min(cap, max(0, free_transfers - made) + 1)
    return free_transfers


def suggest_transfers(
    squad: pd.DataFrame, rankings: pd.DataFrame, bank: float, top_n: int = 3,
    value_col: str = "predicted_points",
) -> pd.DataFrame:
    """
    squad: one row per owned player - player_code, web_name, position, team, price, value_col
    rankings: full player pool with the same columns, plus status ('a' = available)
    bank: money in the bank, in £m (budget available beyond selling the player being replaced)
    value_col: column both frames rank players on - "predicted_points" (next-GW) by default,
        or "horizon_points" (src.recommendations.horizon) for a multi-week comparison.

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
        same_position["net_gain"] = same_position[value_col] - current[value_col]
        candidates = same_position[same_position["net_gain"] > 0].sort_values("net_gain", ascending=False).head(top_n)

        for _, cand in candidates.iterrows():
            suggestions.append({
                "sell": current["web_name"], "sell_position": current["position"],
                "sell_predicted": current[value_col], "sell_price": current["price"],
                "buy": cand["web_name"], "buy_predicted": cand[value_col],
                "buy_price": cand["price"], "net_gain": round(cand["net_gain"], 3),
                "net_gain_if_hit": round(cand["net_gain"] - TRANSFER_HIT_COST, 3),
            })

    if not suggestions:
        return pd.DataFrame(columns=[
            "sell", "sell_position", "sell_predicted", "sell_price",
            "buy", "buy_predicted", "buy_price", "net_gain", "net_gain_if_hit",
        ])
    return pd.DataFrame(suggestions).sort_values("net_gain", ascending=False).reset_index(drop=True)


def _best_single_transfer(squad: pd.DataFrame, rankings: pd.DataFrame, bank: float,
                            team_counts: dict, excluded_buys: set, value_col: str):
    """One step of the greedy planner: the single best (sell, buy) pair across the whole
    squad, not just one player - used by suggest_transfer_plan to pick one transfer at a time."""
    owned_codes = set(squad["player_code"]) | excluded_buys
    best = None
    for _, current in squad.iterrows():
        budget = current["price"] + bank
        candidates = rankings[
            (rankings["position"] == current["position"])
            & (~rankings["player_code"].isin(owned_codes))
            & (rankings["price"] <= budget)
            & (rankings["status"] == "a")
        ]
        for _, cand in candidates.iterrows():
            count_if_bought = team_counts.get(cand["team"], 0) + (0 if cand["team"] == current["team"] else 1)
            if count_if_bought > MAX_PER_TEAM:
                continue
            gain = cand[value_col] - current[value_col]
            if best is None or gain > best["gain"]:
                best = {"sell_row": current, "buy_row": cand, "gain": gain}
    return best


def suggest_transfer_plan(
    squad: pd.DataFrame, rankings: pd.DataFrame, bank: float, free_transfers: int, max_transfers: int = 5,
    value_col: str = "predicted_points",
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Greedily builds a COORDINATED multi-transfer plan, unlike suggest_transfers() (whose
    rows are independent 1-for-1 comparisons that can recommend the same buy target more than
    once). Picks the single best transfer, applies it to a working copy of the squad/bank/team
    counts, then repeats - so the 2nd transfer's suggestion already accounts for the 1st.
    Greedy, not globally optimal (a true optimum would need to search combinations, since an
    early greedy pick can block a better later combination) - but it's a real, honest
    coordinated plan rather than a set of suggestions that quietly conflict with each other.

    value_col: "predicted_points" (next-GW, default) or "horizon_points" - a transfer's gain
    is judged against whatever horizon that column represents, while the hit cost is still a
    flat one-time -4 either way.

    Stops when no remaining transfer has positive net gain after its hit cost, or
    `max_transfers` is reached. Transfers beyond `free_transfers` are charged TRANSFER_HIT_COST.

    Returns (plan_df, resulting_squad_df, remaining_bank_after_the_plan) - resulting_squad_df
    is the actual post-transfer 15 (same columns as the input `squad`), for callers that need
    to keep simulating forward (e.g. src.validation.backtest), not just display the plan.
    """
    working_squad = squad.copy()
    working_bank = bank
    team_counts = working_squad["team"].value_counts().to_dict()
    excluded_buys: set = set()
    plan = []

    for i in range(max_transfers):
        best = _best_single_transfer(working_squad, rankings, working_bank, team_counts, excluded_buys, value_col)
        if best is None:
            break
        hit = 0 if (i + 1) <= free_transfers else TRANSFER_HIT_COST
        net = best["gain"] - hit
        if net <= 0:
            break

        sell_row, buy_row = best["sell_row"], best["buy_row"]
        plan.append({
            "transfer_num": i + 1, "sell": sell_row["web_name"], "buy": buy_row["web_name"],
            "gain": round(best["gain"], 3), "hit": hit, "net": round(net, 3),
            "free_transfer_used": hit == 0,
        })

        working_bank = working_bank + sell_row["price"] - buy_row["price"]
        team_counts[sell_row["team"]] = team_counts.get(sell_row["team"], 1) - 1
        team_counts[buy_row["team"]] = team_counts.get(buy_row["team"], 0) + 1
        excluded_buys.add(buy_row["player_code"])
        working_squad = working_squad[working_squad["player_code"] != sell_row["player_code"]]
        working_squad = pd.concat([working_squad, buy_row.to_frame().T], ignore_index=True)

    plan_df = pd.DataFrame(plan) if plan else pd.DataFrame(
        columns=["transfer_num", "sell", "buy", "gain", "hit", "net", "free_transfer_used"]
    )
    return plan_df, working_squad.reset_index(drop=True), working_bank
