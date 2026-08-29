"""Postgres connection helpers - the one shared place every part of this project gets a DB
handle from. Used to be duplicated by hand in dashboard/app.py and src/notify/deadline_scheduler.py
too, which is how a fix applied to one silently missed the others - consolidated here instead.

Two entry points for two genuinely different process shapes in this project:
- get_connection(): a plain psycopg2 connection, for one-shot scripts that connect, do their
  work, and exit (ingestion, src/notify/deadline_scheduler.py, Model Lab's one-off writes).
- get_engine(): a pooled SQLAlchemy engine, for the long-running Streamlit dashboard that stays
  up for hours across many reruns - see its own docstring for why that distinction matters.
"""
from __future__ import annotations

import time

import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config import settings

# Supabase's Supavisor pooler tears down our per-tenant connection pool after a short idle
# window. The very first connection after that hits a real Supavisor-side bug: its per-tenant
# secret cache is cold, falls back to a broken one-off auth_query lookup, and spuriously reports
# "password authentication failed" even with a fully correct password. Confirmed directly
# against Supavisor's own logs (source='supavisor_logs') - every attempt immediately after the
# first failing one succeeds. This retry absorbs that transient failure instead of surfacing it.
_MAX_CONNECT_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 1.5


def get_connection():
    if not settings.DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env in the repo root and fill it in."
        )
    last_exc: psycopg2.OperationalError | None = None
    for _ in range(_MAX_CONNECT_ATTEMPTS):
        try:
            return psycopg2.connect(settings.DATABASE_URL)
        except psycopg2.OperationalError as exc:
            last_exc = exc
            time.sleep(_RETRY_DELAY_SECONDS)
    raise last_exc


_engine: Engine | None = None


def get_engine() -> Engine:
    """A pooled SQLAlchemy engine, cached as a module-level singleton (safe to share - engines
    are thread-safe by design, that's what the pool is for; module-level state also survives
    fine across Streamlit reruns since the module is only imported once per process).

    This exists specifically to fix a real bug, not as a style preference: the dashboard used to
    cache one raw psycopg2 connection via @st.cache_resource and reuse it for the app's whole
    lifetime. But Supavisor can silently drop/recycle the backend connection server-side at any
    point, and a raw cached connection has no way to notice - it just fails on the next query.
    That's exactly why some dashboard tables loaded and others didn't: whichever queries ran
    before the pooler recycled the connection worked, everything after silently used a dead one.

    pool_pre_ping issues a cheap liveness check before handing out any pooled connection and
    transparently reconnects if it's been dropped - this is the actual fix. pool_recycle
    proactively retires connections before Supavisor's own idle teardown would, for the same
    reason. This also fixes a real pandas UserWarning seen in production logs ("pandas only
    supports SQLAlchemy connectable... other DBAPI2 objects are not tested") - pass this engine
    directly to pd.read_sql_query, not a raw connection.
    """
    global _engine
    if _engine is None:
        if not settings.DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is not set. Copy .env.example to .env in the repo root and fill it in."
            )
        _engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=280,
            pool_size=5,
            max_overflow=2,
        )
    return _engine
