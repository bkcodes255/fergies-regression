"""Thin wrapper over the public FPL API. No auth required — it's a public read API."""
from __future__ import annotations

import time
from typing import Any, Optional

import requests

BASE_URL = "https://fantasy.premierleague.com/api"
USER_AGENT = "Mozilla/5.0 (fergies-regression data client; personal analytics project)"
TIMEOUT_SECONDS = 15
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


class FPLClient:
    """Thin, retrying wrapper over the endpoints Phase 1 needs.

    Deliberately dumb: returns raw parsed JSON, no field mapping or validation here.
    That happens in load.py against the schema in sql/schema.sql — see
    data/data_dictionary.md for the field-by-field mapping.
    """

    def __init__(self, base_url: str = BASE_URL, session: Optional[requests.Session] = None) -> None:
        self.base_url = base_url
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _get(self, path: str) -> Any:
        url = f"{self.base_url}/{path}"
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.session.get(url, timeout=TIMEOUT_SECONDS)
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        raise RuntimeError(f"FPL API request failed after {MAX_RETRIES} attempts: {url}") from last_error

    def get_bootstrap_static(self) -> dict:
        """Players, teams, gameweeks (events), season totals. The main reference pull."""
        return self._get("bootstrap-static/")

    def get_fixtures(self) -> list:
        """All fixtures for the season, with results/difficulty as known so far."""
        return self._get("fixtures/")

    def get_event_live(self, event_id: int) -> dict:
        """Per-player stats + point breakdown ('explain') for one gameweek."""
        return self._get(f"event/{event_id}/live/")

    def get_element_summary(self, player_id: int) -> dict:
        """One player's full gameweek history + upcoming fixtures. Not used in Phase 1's
        bulk pull (bootstrap-static + event/live cover the same ground for all players at
        once) — kept here for later on-demand lookups, e.g. the Player Explorer view."""
        return self._get(f"element-summary/{player_id}/")
