"""Data for the login-only Season page: the manager's gameweek history and how the model's
pre-deadline predictions did against what actually happened. Same numbers as the Streamlit
Season tab (dashboard/app.py); the two queries are copied here because that file imports
Streamlit.

Private exporter: it reads one manager's results, so the page is served only behind the login.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def load_manager_gameweeks(conn, season: str, entry_id: int) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT event_id, points, total_points, overall_rank, points_on_bench FROM manager_gameweeks "
        "WHERE entry_id = %(entry_id)s AND season = %(season)s ORDER BY event_id",
        conn, params={"entry_id": entry_id, "season": season},
    )


def load_gw_comparison(conn, season: str, entry_id: int) -> pd.DataFrame:
    """One row per elapsed gameweek from v_manager_gw_comparison (sql/analytics.sql). The model
    and FPL predictions are NaN for any gameweek with no pre-deadline prediction (always GW1):
    unavailable, not zero."""
    return pd.read_sql_query(
        "SELECT * FROM v_manager_gw_comparison WHERE entry_id = %(entry_id)s AND season = %(season)s "
        "ORDER BY event_id",
        conn, params={"entry_id": entry_id, "season": season},
    )


def _num(v, digits: int | None = None):
    """A plain number, or None for NaN/missing (JSON has no NaN)."""
    if v is None or pd.isna(v):
        return None
    return round(float(v), digits) if digits is not None else float(v)


def _int(v):
    return None if v is None or pd.isna(v) else int(v)


def _empty(season: str) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "season": season,
        "summary": None, "gameweeks": [], "accuracy": None,
    }


def summary(manager_gw: pd.DataFrame) -> dict:
    latest = manager_gw.sort_values("event_id").iloc[-1]
    best = manager_gw.loc[manager_gw["points"].idxmax()]
    return {
        "total_points": _int(latest["total_points"]),
        "overall_rank": _int(latest["overall_rank"]),
        "average_points": round(float(manager_gw["points"].mean()), 1),
        "best_gw": {"gw": int(best["event_id"]), "points": int(best["points"])},
        "bench_points": int(manager_gw["points_on_bench"].sum()),
        "gameweeks_played": int(len(manager_gw)),
    }


def gameweek_rows(comparison: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in comparison.iterrows():
        actual = _num(r["actual_points"])
        model = _num(r["model_predicted_points"], 1)
        rows.append({
            "gw": int(r["event_id"]),
            "actual": actual,
            "model": model,
            "fpl": _num(r["epl_predicted_points"], 1),
            "delta": None if model is None or actual is None else round(actual - model, 1),
            "average": _num(r["gw_average"]),
            "highest": _num(r["gw_highest"]),
            "percentile": _num(r["percentile"]),
        })
    return rows


def accuracy(comparison: pd.DataFrame) -> dict | None:
    """Weeks where both predictions existed: how often the model beat FPL's own, mean errors, and
    the biggest underestimate. None when no gameweek has both."""
    valid = comparison.dropna(subset=["model_predicted_points", "epl_predicted_points"])
    if valid.empty:
        return None
    model_err = (valid["actual_points"] - valid["model_predicted_points"]).abs()
    fpl_err = (valid["actual_points"] - valid["epl_predicted_points"]).abs()
    gap = valid["actual_points"] - valid["model_predicted_points"]
    miss = valid.loc[gap.idxmax()]
    return {
        "weeks": int(len(valid)),
        "beats": int((model_err < fpl_err).sum()),
        "model_error": round(float(model_err.mean()), 1),
        "fpl_error": round(float(fpl_err.mean()), 1),
        "biggest_miss": {
            "gw": int(miss["event_id"]), "predicted": round(float(miss["model_predicted_points"])),
            "actual": round(float(miss["actual_points"])), "gap": round(float(gap.max())),
        },
    }


def payload(conn, season: str, entry_id: int | None) -> dict:
    if not entry_id:
        return _empty(season)
    manager_gw = load_manager_gameweeks(conn, season, entry_id)
    if manager_gw.empty:
        return _empty(season)
    comparison = load_gw_comparison(conn, season, entry_id)
    out = _empty(season)
    out["summary"] = summary(manager_gw)
    out["gameweeks"] = gameweek_rows(comparison)
    out["accuracy"] = accuracy(comparison)
    return out
