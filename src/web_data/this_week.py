"""This Week as data: everything the web site's This Week page shows, computed with the same
functions the Streamlit tab uses, as one JSON-ready dict.

The site is static - a GitHub Action runs export.py after each data refresh, writes this payload
next to the site's source, rebuilds and deploys. So nothing here runs per page view and the
database is never reachable from the browser.

The board and hairdryer builders are imported from dashboard/ (they are plain functions, no
Streamlit), so the site's calls are the Streamlit tab's calls exactly.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from dashboard.hairdryer import FULL_BLAST_PTS, build_calls
from dashboard.team_talk import board_calls, board_data
from src.models.monte_carlo import simulate_squad, summarize_simulation
from src.recommendations.gameweek_plan import with_horizon
from src.recommendations.horizon import DEFAULT_HORIZON, fixture_grade
from src.recommendations.plan_payloads import _iso, transfer_plan_payload
from src.recommendations.squad_optimizer import best_starting_xi
from src.recommendations.transfers import compute_free_transfers, suggest_transfer_plan
from src.web_data import queries

SIM_SAMPLES = 10000
HIST_BINS = 36


def captain_recommendation(starting: pd.DataFrame) -> dict | None:
    """Compares the model's top-predicted starter against the currently-armbanded captain -
    None if either side can't be determined (no predictions yet, or no captain flagged)."""
    if starting.empty or starting["predicted_points"].isna().all():
        return None
    current = starting[starting["is_captain"]]
    if current.empty:
        return None
    best_row = starting.loc[starting["predicted_points"].idxmax()]
    current_row = current.iloc[0]
    return {
        "matches": best_row["player_code"] == current_row["player_code"],
        "recommended_code": best_row["player_code"], "recommended_name": best_row["web_name"],
        "recommended_points": best_row["predicted_points"],
        "current_name": current_row["web_name"], "current_points": current_row["predicted_points"],
    }


def lineup_changes(squad: pd.DataFrame) -> dict:
    """Diffs the actual starting XI against best_starting_xi's recommendation by player_code -
    empty bring_in/bench_out means the fielded lineup is already optimal."""
    starting_codes = set(squad.loc[squad["multiplier"] > 0, "player_code"])
    optimal_xi, formation = best_starting_xi(squad)
    optimal_codes = set(optimal_xi["player_code"])
    return {
        "bring_in_codes": optimal_codes - starting_codes,
        "bench_out_codes": starting_codes - optimal_codes,
        "optimal_points": optimal_xi["predicted_points"].sum(),
        "formation": formation,
    }

def _num(v, digits: int = 1) -> float | None:
    if v is None:
        return None
    f = float(v)
    return None if math.isnan(f) else round(f, digits)


def _fixtures_by_team(grid: pd.DataFrame) -> dict[str, list[dict]]:
    """Each team's next fixtures, one entry per fixture (a double gameweek gives two)."""
    out: dict[str, list[dict]] = {}
    for _, r in grid.iterrows():
        out.setdefault(r["team"], []).append({
            "gw": int(r["event_id"]), "opp": r["opponent"], "home": bool(r["is_home"]),
            "grade": fixture_grade(r["fixture_difficulty_score"]),
        })
    return out


def _players(squad: pd.DataFrame, fixtures: dict[str, list[dict]]) -> dict[str, dict]:
    """The player sheet behind each magnet, keyed by player_code (as a string, for JSON)."""
    return {
        str(int(r["player_code"])): {
            "name": r["web_name"], "team": r["team"], "position": r["position"], "price": _num(r["price"]),
            "predicted": _num(r["predicted_points"]), "horizon": _num(r.get("horizon_points")),
            "floor": _num(r.get("floor_points")), "median": _num(r.get("median_points")),
            "ceiling": _num(r.get("ceiling_points")),
            "haul": _num(r.get("p_haul_10plus"), 3),
            "fixtures": fixtures.get(r["team"], []),
        }
        for _, r in squad.iterrows()
    }


def _outcomes(starting: pd.DataFrame) -> dict:
    totals = simulate_squad(starting, n_samples=SIM_SAMPLES)
    stats = summarize_simulation(totals)
    counts, edges = np.histogram(totals, bins=HIST_BINS)
    return {
        "samples": SIM_SAMPLES,
        "p10": _num(stats["p10"]), "p50": _num(stats["p50"]), "p90": _num(stats["p90"]),
        "mean": _num(stats["mean"]), "p_beat_60": _num(stats["p_beat_60"], 3), "p_beat_80": _num(stats["p_beat_80"], 3),
        "edges": [_num(e, 2) for e in edges], "counts": [int(c) for c in counts],
    }


def _after_board(squad: pd.DataFrame) -> dict:
    """The board once the lineup and captain calls are made: best XI from the 15, armband on its
    top predicted scorer, vice on the next. Bench as the approval bot sets it - goalkeeper
    first (FPL requires it), then outfielders by predicted points."""
    xi, _ = best_starting_xi(squad)
    after = squad.copy()
    in_xi = after["player_code"].isin(xi["player_code"])
    ranked = after[in_xi].sort_values("predicted_points", ascending=False)
    after["is_captain"] = after["player_code"] == ranked.iloc[0]["player_code"]
    after["is_vice_captain"] = after["player_code"] == ranked.iloc[1]["player_code"]
    bench = after[~in_xi].assign(_gk=lambda d: d["position"] != "GKP").sort_values(
        ["_gk", "predicted_points"], ascending=[True, False]
    )
    return board_data(after[in_xi], bench)


def this_week_payload(conn, season: str, entry_id: int | None) -> dict:
    """The page's data. squad is None when there's no manager squad to talk about (no
    FPL_ENTRY_ID, or nothing ingested yet) - the page shows that state rather than failing."""
    last_gw = queries.load_last_ingested_gw(conn, season)
    rankings = queries.load_player_rankings(conn, season)
    predicted_gw = rankings["predicted_gw"].dropna().max() if not rankings.empty else None
    predicted_gw = None if predicted_gw is None or pd.isna(predicted_gw) else int(predicted_gw)
    rankings = with_horizon(conn, rankings, season)
    deadline = queries.load_next_deadline(conn, season)

    payload: dict = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "season": season, "last_gw": last_gw, "predicted_gw": predicted_gw,
        "deadline": None if deadline is None else {
            "event_id": deadline["event_id"], "name": deadline["name"], "time": _iso(deadline["deadline_time"]),
        },
        "horizon": DEFAULT_HORIZON, "full_blast": FULL_BLAST_PTS,
        "squad": None,
    }
    if not entry_id:
        return payload
    squad, manager_gw = queries.load_squad(conn, season, entry_id)
    if squad.empty:
        return payload

    squad = squad.merge(
        rankings[["player_code", "horizon_points", "horizon_fixtures"]], on="player_code", how="left"
    )
    starting = squad[squad["multiplier"] > 0].copy()
    bench = squad[squad["multiplier"] == 0].sort_values("squad_position")

    free_transfers = compute_free_transfers(manager_gw)
    bank = float(manager_gw.iloc[0]["bank"]) / 10 if not manager_gw.empty else 0.0
    cap = captain_recommendation(starting)
    changes = lineup_changes(squad)
    # Both horizons, as the Streamlit tab's toggle and the Telegram approval offer them; the
    # calls (board and hairdryer) use the next-gameweek plan, as they always have.
    plan_1gw, squad_after_1gw, bank_after_1gw = suggest_transfer_plan(squad, rankings, bank, free_transfers)
    plan_5gw, squad_after_5gw, bank_after_5gw = suggest_transfer_plan(
        squad, rankings, bank, free_transfers, value_col="horizon_points"
    )
    rec = {
        "plan_1gw": plan_1gw, "plan_5gw": plan_5gw, "squad": squad, "free_transfers": free_transfers, "bank": bank,
        "squad_after_1gw": squad_after_1gw, "bank_after_1gw": bank_after_1gw,
        "squad_after_5gw": squad_after_5gw, "bank_after_5gw": bank_after_5gw,
    }
    start_gw = deadline["event_id"] if deadline is not None else (predicted_gw or (last_gw or 0) + 1)
    fixtures = _fixtures_by_team(queries.load_fixture_grid(conn, season, start_gw, DEFAULT_HORIZON))

    payload["squad"] = {
        "free_transfers": int(free_transfers), "bank": round(bank, 1),
        "board": board_data(starting, bench),
        "after_board": _after_board(squad),
        "board_calls": board_calls(cap, changes, squad, plan_1gw),
        "calls": build_calls(cap, changes, squad, starting, plan_1gw),
        "players": _players(squad, fixtures),
        "plans": {"1gw": transfer_plan_payload(rec, "1gw"), "5gw": transfer_plan_payload(rec, "5gw")},
        "outcomes": _outcomes(starting),
    }
    return payload
