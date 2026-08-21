"""Postgres connection helper."""
from __future__ import annotations

import psycopg2

from config import settings


def get_connection():
    if not settings.DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env in the repo root and fill it in."
        )
    return psycopg2.connect(settings.DATABASE_URL)
