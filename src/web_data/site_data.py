"""The header and footer data every page shows. Public, so it takes no entry id."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.recommendations.horizon import DEFAULT_HORIZON
from src.recommendations.plan_payloads import _iso
from src.web_data import queries


def payload(conn, season: str) -> dict:
    rankings = queries.load_player_rankings(conn, season)
    predicted_gw = rankings["predicted_gw"].dropna().max() if not rankings.empty else None
    deadline = queries.load_next_deadline(conn, season)
    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "season": season,
        "last_gw": queries.load_last_ingested_gw(conn, season),
        "predicted_gw": None if predicted_gw is None or pd.isna(predicted_gw) else int(predicted_gw),
        "deadline": None if deadline is None else {
            "event_id": deadline["event_id"], "name": deadline["name"], "time": _iso(deadline["deadline_time"]),
        },
        "horizon": DEFAULT_HORIZON,
    }
