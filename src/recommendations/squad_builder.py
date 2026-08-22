"""Full squad-rebuild optimizer: "what is the optimal 15 I could build from the entire player
pool, given a budget?" - a real integer program (PuLP/CBC), not a heuristic, unlike
squad_optimizer.py (which only reorders an *existing* 15) or transfers.py's greedy planner
(which only searches one transfer at a time from the current squad).

Solves squad selection and starting-XI selection together in one MILP, rather than picking 15
players and then separately figuring out who starts - a squad that's great on paper but can't
field a legal starting XI within budget is not actually useful, so the objective is "maximize
what the squad can actually score," with the captain's double points included.
"""
from __future__ import annotations

import pandas as pd
import pulp

MAX_PER_TEAM = 3
SQUAD_COUNTS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}  # fixed - not a range (sums to 15)
STARTING_XI_RANGE = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}  # from
# bootstrap-static's element_types squad_min_play/squad_max_play, verified live against the
# 2026/27 API (same source squad_optimizer.py uses for the 8-formation brute force)


def build_optimal_squad(rankings: pd.DataFrame, budget: float, solver_time_limit: int = 30):
    """
    rankings: full player pool - player_code, web_name, position, team, price,
              predicted_points, status ('a' = available)
    budget: total spend allowed, in £m

    Returns (squad_df, starting_xi_df, captain_row, objective_value) or (None, None, None, None)
    if no feasible solution exists (e.g. budget too low to field a legal squad at all).
    """
    pool = rankings[rankings["status"] == "a"].reset_index(drop=True)
    pool = pool[pool["predicted_points"].notna()]
    idx = pool.index.tolist()

    prob = pulp.LpProblem("fpl_squad_optimizer", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", idx, cat="Binary")
    start = pulp.LpVariable.dicts("start", idx, cat="Binary")
    captain = pulp.LpVariable.dicts("captain", idx, cat="Binary")

    points = pool["predicted_points"].to_dict()
    prob += pulp.lpSum(points[i] * start[i] for i in idx) + pulp.lpSum(points[i] * captain[i] for i in idx)

    # squad composition: exactly 2 GKP / 5 DEF / 5 MID / 3 FWD (15 total)
    for pos, count in SQUAD_COUNTS.items():
        pos_idx = pool.index[pool["position"] == pos].tolist()
        prob += pulp.lpSum(squad[i] for i in pos_idx) == count

    # budget
    prices = pool["price"].to_dict()
    prob += pulp.lpSum(prices[i] * squad[i] for i in idx) <= budget

    # max 3 players per real team
    for team in pool["team"].unique():
        team_idx = pool.index[pool["team"] == team].tolist()
        prob += pulp.lpSum(squad[i] for i in team_idx) <= MAX_PER_TEAM

    # starting XI must be a subset of the squad, with valid formation
    for i in idx:
        prob += start[i] <= squad[i]
        prob += captain[i] <= start[i]
    prob += pulp.lpSum(captain[i] for i in idx) == 1
    prob += pulp.lpSum(start[i] for i in idx) == 11
    for pos, (lo, hi) in STARTING_XI_RANGE.items():
        pos_idx = pool.index[pool["position"] == pos].tolist()
        prob += pulp.lpSum(start[i] for i in pos_idx) >= lo
        prob += pulp.lpSum(start[i] for i in pos_idx) <= hi

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=solver_time_limit))

    if pulp.LpStatus[prob.status] != "Optimal":
        return None, None, None, None

    squad_idx = [i for i in idx if squad[i].value() == 1]
    starting_idx = [i for i in idx if start[i].value() == 1]
    captain_idx = [i for i in idx if captain[i].value() == 1]

    squad_df = pool.loc[squad_idx].copy()
    squad_df["starting"] = squad_df.index.isin(starting_idx)
    squad_df["captain"] = squad_df.index.isin(captain_idx)
    starting_df = pool.loc[starting_idx].copy()
    captain_row = pool.loc[captain_idx[0]] if captain_idx else None

    objective_value = pulp.value(prob.objective)
    return squad_df, starting_df, captain_row, objective_value
