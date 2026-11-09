"""Data for the public Fixtures page. Public, so it takes no entry id and must not read a manager's squad.

The Streamlit Fixtures tab's grid. The tab also warned when the manager owned players from the
hardest-run team; that reads a squad, so it is not ported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.recommendations.horizon import DEFAULT_HORIZON, fixture_grade
from src.web_data import queries

# The tab's own bands for a team's average difficulty over the window.
EASY_AT_OR_BELOW = -0.2
BRUTAL_AT_OR_ABOVE = 0.25


def run_rating(avg: float) -> str:
    return "Easy" if avg <= EASY_AT_OR_BELOW else "Brutal" if avg >= BRUTAL_AT_OR_ABOVE else "Mixed"


def payload(conn, season: str) -> dict:
    deadline = queries.load_next_deadline(conn, season)
    start_gw = deadline["event_id"] if deadline is not None else 2
    grid = queries.load_fixture_grid(conn, season, start_gw, DEFAULT_HORIZON)
    out: dict = {"start_gw": start_gw, "horizon": DEFAULT_HORIZON, "gws": [], "teams": []}
    if grid.empty:
        return out
    gws = sorted(int(g) for g in grid["event_id"].unique())
    out["gws"] = gws
    teams = []
    for team, team_rows in grid.groupby("team"):
        cells, scores = [], []
        for gw in gws:
            rows = team_rows[team_rows["event_id"] == gw]
            if rows.empty:
                cells.append(None)  # a blank gameweek
                continue
            mean = float(rows["fixture_difficulty_score"].mean())
            scores.append(mean)
            cells.append({
                "grade": fixture_grade(mean),  # the cell's colour, from the mean score
                "fx": [
                    {"opp": r.opponent, "home": bool(r.is_home), "grade": fixture_grade(r.fixture_difficulty_score)}
                    for r in rows.itertuples(index=False)
                ],
            })
        avg = float(np.mean(scores)) if scores else 0.0
        teams.append({"team": team, "avg": round(avg, 3), "run": run_rating(avg), "cells": cells})
    # Easiest run first, as the tab sorts it; stable so ties keep alphabetical order.
    teams.sort(key=lambda t: t["avg"])
    out["teams"] = teams
    return out
