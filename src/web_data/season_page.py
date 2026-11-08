"""Data for the login-only Season page."""
from __future__ import annotations


def payload(conn, season: str, entry_id: int | None) -> dict:
    raise NotImplementedError
