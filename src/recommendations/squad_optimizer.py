"""Optimal starting XI from an existing 15-man squad, respecting FPL's actual formation rules
(from bootstrap-static's element_types: GKP exactly 1, DEF 3-5, MID 2-5, FWD 1-3, 11 total).

Within a fixed formation, the best XI is always "top-N predicted points per position" - there's
no cross-position substitution value beyond the headcount itself, so for a fixed (def, mid, fwd)
split this is trivially optimal. What's not trivial is which of the 8 valid formations to use,
so this brute-forces all 8 (a real constant - these are FPL's own starting-XI limits) and picks
the best.
"""
from __future__ import annotations

import pandas as pd

VALID_FORMATIONS = [  # (DEF, MID, FWD), GKP is always exactly 1 and always starts
    (3, 4, 3), (3, 5, 2),
    (4, 3, 3), (4, 4, 2), (4, 5, 1),
    (5, 2, 3), (5, 3, 2), (5, 4, 1),
]


def best_starting_xi(squad: pd.DataFrame) -> tuple[pd.DataFrame, tuple[int, int, int]]:
    """squad: 15 rows with position ('GKP'/'DEF'/'MID'/'FWD') and predicted_points.
    Returns (starting_xi_df, (def_count, mid_count, fwd_count))."""
    by_pos = {
        pos: squad[squad["position"] == pos].sort_values("predicted_points", ascending=False)
        for pos in ("GKP", "DEF", "MID", "FWD")
    }
    gkp = by_pos["GKP"].head(1)

    best_total = -1.0
    best_formation = None
    best_outfield = None
    for def_n, mid_n, fwd_n in VALID_FORMATIONS:
        if len(by_pos["DEF"]) < def_n or len(by_pos["MID"]) < mid_n or len(by_pos["FWD"]) < fwd_n:
            continue  # squad doesn't have enough players in this position for this formation
        outfield = pd.concat([
            by_pos["DEF"].head(def_n), by_pos["MID"].head(mid_n), by_pos["FWD"].head(fwd_n),
        ])
        total = outfield["predicted_points"].sum() + gkp["predicted_points"].sum()
        if total > best_total:
            best_total = total
            best_formation = (def_n, mid_n, fwd_n)
            best_outfield = outfield

    starting_xi = pd.concat([gkp, best_outfield]).reset_index(drop=True)
    return starting_xi, best_formation
