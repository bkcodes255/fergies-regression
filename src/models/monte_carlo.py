"""Monte Carlo simulation on top of the quantile models (src.models.train_quantile_models):
sample each player's next-gameweek points from their floor/median/ceiling, to answer
squad-level questions no single-player metric can - "what's the range of outcomes for my whole
starting XI," not just "what's the range for one player."

Per-player sampling: piecewise-linear inverse-CDF through the three known quantile points
(10th=floor, 50th=median, 90th=ceiling) - not a parametric distribution assumption, just
interpolation between what the quantile models actually predicted, which is the more honest
choice given points_per_90/total_points are nothing like normally distributed. The lower tail
(below the 10th percentile) is flat at `floor` - there's no information below it, and clamping
avoids extrapolating into physically-implausible negative scores. The upper tail (above the
90th percentile) extrapolates the (median, ceiling) slope forward, deliberately NOT flat -
real FPL hauls (20+, even 25+ points) are rare-but-real events, and capping the top 10% at
`ceiling` would understate exactly the upside this whole exercise exists to capture.

Known simplification, stated plainly rather than hidden: players are sampled INDEPENDENTLY.
Real match outcomes correlate within a team (two Arsenal attackers both benefit from the same
big Arsenal win) and across a fixture (a striker's goal correlates with the opposing defense's
clean sheet failing) - this simulation doesn't model that. Independent draws partially cancel
out in a sum, so the squad-level spread reported here is somewhat TIGHTER than reality would
actually produce - a real correlated simulation would show wider tails on both ends.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sample_player_points(floor: float, median: float, ceiling: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(n_samples)
    # np.interp handles the well-defined middle piecewise-linear segments (0.1-0.5-0.9) and
    # flat-extrapolates the lower tail (u<0.1) at `floor` by default - exactly the behavior we want there.
    samples = np.interp(u, [0.0, 0.1, 0.5, 0.9], [floor, floor, median, ceiling])

    # Upper tail: override np.interp's flat-at-ceiling default with a real extrapolation, using
    # the same slope as the (median, ceiling) segment, so the simulation can produce genuine
    # haul-sized (20+) outcomes in the extreme tail rather than hard-capping at the 90th percentile.
    upper_slope = (ceiling - median) / 0.4  # per unit of u across the [0.5, 0.9] segment
    high_mask = u > 0.9
    samples[high_mask] = ceiling + upper_slope * (u[high_mask] - 0.9)

    return np.clip(samples, 0, None)


def simulate_squad(
    squad: pd.DataFrame, n_samples: int = 10000, seed: int = 42,
) -> np.ndarray:
    """squad: rows with floor_points, median_points, ceiling_points, and multiplier (0=bench,
    1=normal starter, 2=captain) - only multiplier>0 rows count toward the total, matching how
    FPL actually scores a gameweek. Returns an array of `n_samples` simulated squad totals."""
    rng = np.random.default_rng(seed)
    starters = squad[squad["multiplier"] > 0]

    total = np.zeros(n_samples)
    for _, row in starters.iterrows():
        floor = row["floor_points"] if pd.notna(row["floor_points"]) else 0.0
        median = row["median_points"] if pd.notna(row["median_points"]) else row.get("predicted_points", 0.0) or 0.0
        ceiling = row["ceiling_points"] if pd.notna(row["ceiling_points"]) else median
        player_samples = sample_player_points(floor, median, ceiling, n_samples, rng)
        total += player_samples * row["multiplier"]

    return total


def summarize_simulation(totals: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(totals)),
        "p10": float(np.percentile(totals, 10)),
        "p50": float(np.percentile(totals, 50)),
        "p90": float(np.percentile(totals, 90)),
        "p_beat_60": float(np.mean(totals >= 60)),
        "p_beat_80": float(np.mean(totals >= 80)),
    }
