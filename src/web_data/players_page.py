"""Data for the public Players page. Public, so it takes no entry id and must not read a manager's squad."""
from __future__ import annotations


def payload(conn, season: str) -> dict:
    raise NotImplementedError
