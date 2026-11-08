"""Data for the public Players page. Public, so it takes no entry id and must not read a manager's squad.

The Streamlit Players tab's full table, for players the model rates as available (the tab's
"Available only" box is ticked by default). Numbers are rounded the way the tab shows them, which
keeps the page light: several hundred rows, each a handful of short numbers.
"""
from __future__ import annotations

import pandas as pd

from src.recommendations.gameweek_plan import with_horizon
from src.recommendations.horizon import DEFAULT_HORIZON
from src.web_data import queries


def _num(v, digits: int | None) -> float | int | None:
    """None for NaN/missing, else the value rounded (digits=None means a whole number)."""
    if v is None or pd.isna(v):
        return None
    return int(round(float(v))) if digits is None else round(float(v), digits)


def payload(conn, season: str) -> dict:
    rankings = queries.load_player_rankings(conn, season)
    predicted_gw = rankings["predicted_gw"].dropna().max() if not rankings.empty else None
    if not rankings.empty:
        rankings = with_horizon(conn, rankings, season)
        rankings = rankings[rankings["status"] == "a"]
        rankings = rankings.sort_values("predicted_points", ascending=False, na_position="last", kind="mergesort")
    players = []
    for r in rankings.itertuples(index=False):
        haul = None if pd.isna(r.p_haul_10plus) else r.p_haul_10plus * 100
        players.append({
            "name": r.web_name, "pos": r.position, "team": r.team,
            "price": _num(r.price, 1), "pred": _num(r.predicted_points, 1), "horizon": _num(r.horizon_points, 1),
            "floor": _num(r.floor_points, 1), "ceil": _num(r.ceiling_points, 1), "haul": _num(haul, None),
            "value": _num(r.predicted_value, 2), "pts": _num(r.total_points, None), "owned": _num(r.selected_by_percent, 1),
        })
    return {
        "predicted_gw": None if predicted_gw is None or pd.isna(predicted_gw) else int(predicted_gw),
        "horizon": DEFAULT_HORIZON,
        "players": players,
    }
