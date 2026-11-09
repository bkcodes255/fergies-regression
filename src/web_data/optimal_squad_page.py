"""Data for the public Optimal Squad page. Public, so it takes no entry id and must not read a manager's squad.

The Streamlit tab solves live when the budget moves. A static site can't, so every option is solved
here, ahead of time: budgets 80.0 to 100.0 in 0.5 steps, for the next gameweek and for the
multi-week window. Many budgets give the same squad, so each distinct squad is stored once and the
budget maps to its index. The browser only looks the answer up.

Not ported from the tab: the "Your squad, same basis" and "already match" tiles, which read the
manager's own squad.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from src.recommendations.gameweek_plan import with_horizon
from src.recommendations.horizon import DEFAULT_HORIZON
from src.recommendations.squad_builder import build_optimal_squad
from src.web_data import queries

BUDGET_MIN, BUDGET_MAX, BUDGET_STEP = 80.0, 100.0, 0.5
BUDGETS = [BUDGET_MIN + i * BUDGET_STEP for i in range(int(round((BUDGET_MAX - BUDGET_MIN) / BUDGET_STEP)) + 1)]
# page key -> the rankings column the tab's "Optimize for" radio maps to
HORIZONS = {"next": "predicted_points", "window": "horizon_points"}
POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}


def _r(v, digits: int = 1) -> float:
    return round(float(v), digits)


def _squad_entry(squad_df: pd.DataFrame, objective: float, value_col: str) -> dict:
    """One solved squad: starting XI and bench as player codes (GKP, DEF, MID, FWD order), captain, points."""
    df = squad_df.assign(_pos=squad_df["position"].map(POSITION_ORDER)).sort_values(
        ["_pos", value_col], ascending=[True, False])
    xi, bench = df[df["starting"]], df[~df["starting"]]
    captain = df[df["captain"]]
    return {
        "xi": [int(c) for c in xi["player_code"]],
        "bench": [int(c) for c in bench["player_code"]],
        "cap": int(captain.iloc[0]["player_code"]) if not captain.empty else None,
        "pts": _r(objective),
        "spend": _r(df["price"].sum()),
    }


def solve_all(rankings: pd.DataFrame, budgets: list[float] = BUDGETS) -> dict:
    """Solves each (horizon, budget). Returns {key: {"squads": [...], "by_budget": [index or None]}}.
    Solves from the highest budget down. A squad that is optimal at budget B and spends S is also
    optimal at every budget from S up to B (still feasible, and a bigger budget only relaxes the
    problem), so those budgets are filled without another solve. Same answers as one solve per
    option, in a fraction of the time.

    A budget with no feasible squad, or whose solve raises, maps to None; the rest carry on."""
    out: dict = {}
    for key, value_col in HORIZONS.items():
        squads: list[dict] = []
        seen: dict[tuple, int] = {}
        by_budget: list[int | None] = [None] * len(budgets)
        order = sorted(range(len(budgets)), key=lambda i: -budgets[i])
        done: set[int] = set()
        for i in order:
            if i in done:
                continue
            done.add(i)
            budget = budgets[i]
            try:
                squad_df, _xi, _cap, objective = build_optimal_squad(rankings, budget, value_col=value_col)
            except Exception as exc:  # noqa: BLE001 - one bad solve must not fail the export
                print(f"optimal-squad: {key} at {budget} failed: {exc}")
                squad_df = None
            if squad_df is None:
                continue
            entry = _squad_entry(squad_df, objective, value_col)
            sig = (tuple(entry["xi"]), tuple(entry["bench"]), entry["cap"])
            if sig not in seen:
                seen[sig] = len(squads)
                squads.append(entry)
            by_budget[i] = seen[sig]
            for j in order:  # the budgets this squad also answers
                if j not in done and entry["spend"] - 1e-6 <= budgets[j] <= budget:
                    by_budget[j] = seen[sig]
                    done.add(j)
        out[key] = {"squads": squads, "by_budget": by_budget}
    return out


def _players(rankings: pd.DataFrame, solved: dict) -> dict:
    codes = {c for h in solved.values() for s in h["squads"] for c in s["xi"] + s["bench"]}
    by_code = rankings.drop_duplicates("player_code").set_index("player_code")

    def val(row, col):
        v = row[col]
        return None if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v) else _r(v)

    return {
        str(c): {
            "n": str(by_code.loc[c, "web_name"]), "team": str(by_code.loc[c, "team"]),
            "pos": str(by_code.loc[c, "position"]), "price": _r(by_code.loc[c, "price"]),
            "next": val(by_code.loc[c], "predicted_points"), "window": val(by_code.loc[c], "horizon_points"),
        }
        for c in sorted(codes)
    }


def build(rankings: pd.DataFrame, season: str, budgets: list[float] = BUDGETS) -> dict:
    solved = solve_all(rankings, budgets)
    top = budgets.index(max(budgets))
    for key, solution in solved.items():
        # The top budget always has an answer unless the solver itself is broken. Stop the deploy
        # here: solve_all contains errors, so without this an empty page would ship.
        if solution["by_budget"][top] is None:
            raise RuntimeError(f"optimal-squad: no squad for '{key}' at the top budget of {budgets[top]}.")
    predicted_gw = rankings["predicted_gw"].dropna().max() if not rankings.empty else None
    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "season": season,
        "predicted_gw": None if predicted_gw is None or pd.isna(predicted_gw) else int(predicted_gw),
        "horizon": DEFAULT_HORIZON,
        "budgets": budgets,
        "solutions": solved,
        "players": _players(rankings, solved),
    }


def payload(conn, season: str) -> dict:
    rankings = queries.load_player_rankings(conn, season)
    rankings = with_horizon(conn, rankings, season)
    return build(rankings, season)
