"""The ten read-only tools of contract v0.1 - what Muse (custom connector) and Claude (MCP) can
ask the app. Every tool is a curated query, never SQL passthrough, and runs inside one
READ ONLY transaction, so even a mistake here can't write to the database.

The plan tools (get_transfer_plan, get_lineup_plan) go through
src.recommendations.gameweek_plan.compute_recommendation - the same function the Telegram
approval flow proposes from and auto-submits - so an agent explains exactly the plan that would
run, not a parallel re-implementation of it.

Conventions (from the contract): money in decimal £m, times ISO-8601 UTC, players identified by
player_code (stable across seasons) with web_name for display. FPL's season-scoped element id is
never returned - it's a write-path detail.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import Field
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine

from config import settings
from src.recommendations.gameweek_plan import compute_recommendation, load_next_deadline, load_rankings, with_horizon
from src.recommendations.horizon import DEFAULT_HORIZON, fixture_grade
from src.recommendations.plan_payloads import (
    _hours_until, _int, _iso, _num, _player, _text, lineup_plan_payload, transfer_plan_payload,
)
from src.recommendations.transfers import compute_free_transfers

POSITION = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
MAX_LIMIT = 100

# deadline_scheduler.TIERS opens the "T-30m" tier 2h out because GitHub Actions' cron runs late
# (see the comment there) - so an undecided approval can execute anywhere in this window.
AUTO_SUBMIT_WINDOW = (
    "Anything still 'proposed' is auto-submitted between about 2 hours and 30 minutes before the "
    "deadline (GitHub Actions runs the scheduler late, so the exact time varies). An untouched "
    "transfer approval submits the next-gameweek plan, not the 5-gameweek one."
)


class ToolError(Exception):
    """A failure the caller should see as {code, message} rather than a crash."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


# --- database -------------------------------------------------------------------------------

_engine: Engine | None = None


def _get_engine() -> Engine:
    """TOOLS_DATABASE_URL is meant for a read-only role (sql/tools_server_role.sql); falls back
    to DATABASE_URL for local runs. Read-only is enforced per transaction either way."""
    global _engine
    if _engine is None:
        url = os.environ.get("TOOLS_DATABASE_URL") or settings.DATABASE_URL
        if not url:
            raise ToolError("not_configured", "No database configured: set TOOLS_DATABASE_URL.")
        _engine = create_engine(url, pool_pre_ping=True, pool_recycle=280, pool_size=3, max_overflow=2)
    return _engine


@contextmanager
def _read_only():
    """One READ ONLY transaction per tool call. Transaction-scoped on purpose: Supabase's pooler
    (transaction mode) would not keep a session-level setting between calls."""
    with _get_engine().begin() as conn:
        conn.exec_driver_sql("SET TRANSACTION READ ONLY")
        yield conn


def _query(conn: Connection, sql: str, **params: Any) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


# --- formatting helpers ---------------------------------------------------------------------

def _limit(limit: int) -> int:
    return max(1, min(int(limit), MAX_LIMIT))


# --- shared loaders -------------------------------------------------------------------------

def _season() -> str:
    return settings.SEASON


def _next_gameweek(conn: Connection) -> dict | None:
    return load_next_deadline(conn)


def _recommendation(conn: Connection) -> tuple[dict, dict]:
    gw = _next_gameweek(conn)
    if gw is None:
        raise ToolError("no_upcoming_gameweek", "There's no upcoming deadline in the ingested data.")
    if not settings.ENTRY_ID:
        raise ToolError("not_configured", "FPL_ENTRY_ID isn't set, so there's no squad to plan for.")
    rec = compute_recommendation(conn, gw)
    if rec is None:
        raise ToolError("no_squad", "No squad has been ingested for this entry yet.")
    return gw, rec


def _player_pool(conn: Connection) -> pd.DataFrame:
    """Every player this season with next-GW prediction, 5-GW horizon points, the quantile
    models, and the latest availability news - the same rankings the plans are built from,
    plus the descriptive columns the plans don't need."""
    season = _season()
    pool = with_horizon(conn, load_rankings(conn, season), season)
    extra = _query(
        conn,
        """
        SELECT ps.player_code, p.first_name, p.second_name,
               pr.event_id AS prediction_gw, pr.predicted_points AS raw_predicted_points,
               pr.floor_points, pr.median_points, pr.ceiling_points, pr.p_haul_10plus,
               snap.chance_of_playing_next_round, snap.news, snap.selected_by_percent
        FROM player_seasons ps
        JOIN players p ON p.player_code = ps.player_code
        LEFT JOIN predictions pr ON pr.player_code = ps.player_code AND pr.season = ps.season
            AND pr.event_id = (SELECT MAX(event_id) FROM predictions WHERE season = ps.season)
        LEFT JOIN LATERAL (
            SELECT chance_of_playing_next_round, news, selected_by_percent
            FROM player_price_snapshots pps
            WHERE pps.player_code = ps.player_code AND pps.season = ps.season
            ORDER BY snapshot_date DESC LIMIT 1
        ) snap ON true
        WHERE ps.season = %(season)s
        """,
        season=season,
    )
    return pool.merge(extra, on="player_code", how="left")


def _pool_entry(row: pd.Series) -> dict:
    return _player(row, {
        "status": _text(row.get("status")),
        "chance_of_playing": _int(row.get("chance_of_playing_next_round")),
        "news": _text(row.get("news")),
        "predicted_points": _num(row.get("raw_predicted_points")),
        "horizon_points": _num(row.get("horizon_points")) if pd.notna(row.get("raw_predicted_points")) else None,
        "horizon_fixtures": _int(row.get("horizon_fixtures")),
        "floor": _num(row.get("floor_points")),
        "median": _num(row.get("median_points")),
        "ceiling": _num(row.get("ceiling_points")),
        "haul_chance": _num(row.get("p_haul_10plus"), 3),
        "selected_by_percent": _num(row.get("selected_by_percent"), 1),
    })


def _filter_pool(pool: pd.DataFrame, position: str | None, team: str | None, max_price: float | None) -> pd.DataFrame:
    if position:
        pool = pool[pool["position"] == position]
    if team:
        pool = pool[pool["team"].str.upper() == team.upper()]
    if max_price is not None:
        pool = pool[pool["price"] <= max_price + 1e-9]
    return pool


# --- tools ----------------------------------------------------------------------------------

Position = Literal["GKP", "DEF", "MID", "FWD"]
Horizon = Literal["1gw", "5gw"]


def get_status() -> dict:
    """How fresh the data is: when it was last ingested, which gameweek the predictions and the
    synced squad are for, and the next deadline. Check this before answering anything
    time-sensitive; read `warnings` first - it says when numbers are stale."""
    season = _season()
    with _read_only() as conn:
        row = _query(
            conn,
            """
            SELECT
              (SELECT MAX(pulled_at) FROM player_price_snapshots WHERE season = %(s)s) AS prices_at,
              (SELECT MAX(pulled_at) FROM gameweeks WHERE season = %(s)s) AS gameweeks_at,
              (SELECT MAX(pulled_at) FROM player_gameweek_stats WHERE season = %(s)s) AS stats_at,
              (SELECT MAX(event_id) FROM predictions WHERE season = %(s)s) AS prediction_gw,
              (SELECT MAX(predicted_at) FROM predictions WHERE season = %(s)s) AS predicted_at,
              (SELECT MAX(event_id) FROM squad_picks WHERE season = %(s)s AND entry_id = %(e)s) AS squad_gw
            """,
            s=season, e=settings.ENTRY_ID or -1,
        ).iloc[0]
        gw = _next_gameweek(conn)

    pulled = [row[c] for c in ("prices_at", "gameweeks_at", "stats_at") if pd.notna(row[c])]
    last_ingest = max(pd.Timestamp(t) for t in pulled) if pulled else None
    prediction_gw, squad_gw = _int(row["prediction_gw"]), _int(row["squad_gw"])
    warnings = []
    if gw is not None:
        if prediction_gw is None:
            warnings.append("No predictions are stored yet.")
        elif prediction_gw != gw["event_id"]:
            warnings.append(
                f"Predictions are for GW{prediction_gw}, but the next deadline is GW{gw['event_id']}: "
                "anything about this gameweek uses stale predictions."
            )
        if squad_gw is not None and squad_gw < gw["event_id"] - 1:
            warnings.append(
                f"Your squad was last synced for GW{squad_gw}; changes since then aren't visible yet."
            )
    if last_ingest is not None and (pd.Timestamp.now(tz="UTC") - last_ingest).total_seconds() > 24 * 3600:
        warnings.append(f"The last data refresh was over a day ago ({_iso(last_ingest)}).")
    return {
        "season": season,
        "last_ingest_at": _iso(last_ingest),
        "latest_prediction_gameweek": prediction_gw,
        "predictions_made_at": _iso(row["predicted_at"]),
        "latest_squad_sync_gameweek": squad_gw,
        "entry_configured": bool(settings.ENTRY_ID),
        "next_deadline": None if gw is None else {
            "gameweek": gw["event_id"], "name": gw["name"],
            "deadline_at": _iso(gw["deadline_time"]), "hours_remaining": _hours_until(gw["deadline_time"]),
        },
        "warnings": warnings,
    }


def get_current_gameweek() -> dict:
    """The next gameweek whose deadline hasn't passed: id, name, deadline and hours remaining."""
    with _read_only() as conn:
        gw = _next_gameweek(conn)
    if gw is None:
        return {"upcoming": False, "message": "No upcoming deadline in the ingested data (season over, or not ingested)."}
    return {
        "upcoming": True, "event_id": gw["event_id"], "name": gw["name"],
        "deadline_at": _iso(gw["deadline_time"]), "hours_remaining": _hours_until(gw["deadline_time"]),
        "finished": False,
    }


def get_squad() -> dict:
    """Your current 15 as of the last synced gameweek: position, team, price, next-gameweek
    prediction, availability status and news, who starts and who's captain - plus bank, team
    value and free transfers for the coming gameweek."""
    if not settings.ENTRY_ID:
        raise ToolError("not_configured", "FPL_ENTRY_ID isn't set, so there's no squad.")
    season = _season()
    with _read_only() as conn:
        picks = _query(
            conn,
            """
            SELECT sp.player_code, p.web_name, ps.element_type, t.short_name AS team,
                   sp.squad_position, sp.multiplier, sp.is_captain, sp.is_vice_captain, sp.event_id,
                   snap.now_cost, snap.status, snap.chance_of_playing_next_round, snap.news,
                   pr.predicted_points, pr.event_id AS prediction_gw
            FROM squad_picks sp
            JOIN players p ON p.player_code = sp.player_code
            JOIN player_seasons ps ON ps.player_code = sp.player_code AND ps.season = sp.season
            JOIN teams t ON t.team_code = ps.team_code
            LEFT JOIN LATERAL (
                SELECT now_cost, status, chance_of_playing_next_round, news FROM player_price_snapshots pps
                WHERE pps.player_code = sp.player_code AND pps.season = sp.season
                ORDER BY snapshot_date DESC LIMIT 1
            ) snap ON true
            LEFT JOIN predictions pr ON pr.player_code = sp.player_code AND pr.season = sp.season
                AND pr.event_id = (SELECT MAX(event_id) FROM predictions WHERE season = sp.season)
            WHERE sp.entry_id = %(e)s AND sp.season = %(s)s
              AND sp.event_id = (SELECT MAX(event_id) FROM squad_picks WHERE entry_id = %(e)s AND season = %(s)s)
            ORDER BY sp.squad_position
            """,
            e=settings.ENTRY_ID, s=season,
        )
        history = _query(
            conn,
            "SELECT * FROM manager_gameweeks WHERE entry_id = %(e)s AND season = %(s)s ORDER BY event_id DESC",
            e=settings.ENTRY_ID, s=season,
        )
    if picks.empty:
        raise ToolError("no_squad", "No squad has been ingested for this entry yet.")
    picks["position"] = picks["element_type"].map(POSITION)
    picks["price"] = picks["now_cost"] / 10
    latest = history.iloc[0] if not history.empty else None
    as_of = int(picks["event_id"].iloc[0])
    return {
        "as_of_gameweek": as_of,
        "prediction_gameweek": _int(picks["prediction_gw"].dropna().max()) if picks["prediction_gw"].notna().any() else None,
        "bank": _num(latest["bank"] / 10, 1) if latest is not None and pd.notna(latest["bank"]) else None,
        "team_value": _num(latest["team_value"] / 10, 1) if latest is not None and pd.notna(latest["team_value"]) else None,
        "free_transfers": compute_free_transfers(history) if not history.empty else None,
        "players": [
            _player(r, {
                "predicted_points": _num(r["predicted_points"]),
                "status": _text(r["status"]), "chance_of_playing": _int(r["chance_of_playing_next_round"]),
                "news": _text(r["news"]), "starting": bool(r["multiplier"] > 0),
                "squad_position": _int(r["squad_position"]),
                "captain": bool(r["is_captain"]), "vice_captain": bool(r["is_vice_captain"]),
            })
            for _, r in picks.iterrows()
        ],
        "note": (
            f"This is the squad as picked for GW{as_of}. Transfers or lineup changes made in FPL since "
            "then show up after the next data refresh."
        ),
    }


def search_players(
    query: Annotated[str, Field(description="Part of the player's name, e.g. 'saka'. Empty matches everyone.")] = "",
    position: Annotated[Position | None, Field(description="Only this position.")] = None,
    team: Annotated[str | None, Field(description="Team short name, e.g. 'ARS'.")] = None,
    max_price: Annotated[float | None, Field(description="Maximum price in £m, e.g. 6.5.", gt=0)] = None,
    limit: Annotated[int, Field(description="How many to return (max 100).", ge=1, le=MAX_LIMIT)] = 25,
) -> dict:
    """Find players by name, position, team and price. Each comes with price, next-gameweek
    prediction, 5-gameweek fixture-weighted points, floor/ceiling, availability status and
    news. Sorted by 5-gameweek points."""
    with _read_only() as conn:
        pool = _player_pool(conn)
    pool = _filter_pool(pool, position, team, max_price)
    if query.strip():
        q = query.strip().lower()
        names = (pool["web_name"].fillna("") + " " + pool["first_name"].fillna("") + " " + pool["second_name"].fillna(""))
        pool = pool[names.str.lower().str.contains(q, regex=False)]
    pool = pool.sort_values("horizon_points", ascending=False, na_position="last")
    return {
        "count": int(len(pool)),
        "players": [_pool_entry(r) for _, r in pool.head(_limit(limit)).iterrows()],
        "horizon_gameweeks": DEFAULT_HORIZON,
    }


def get_top_predictions(
    horizon: Annotated[Horizon, Field(description="'1gw' ranks by next-gameweek points, '5gw' by fixture-weighted points over the next 5.")] = "5gw",
    position: Annotated[Position | None, Field(description="Only this position.")] = None,
    max_price: Annotated[float | None, Field(description="Maximum price in £m.", gt=0)] = None,
    available_only: Annotated[bool, Field(description="Leave out players flagged doubtful, injured, suspended or unavailable.")] = False,
    limit: Annotated[int, Field(description="How many to return (max 100).", ge=1, le=MAX_LIMIT)] = 25,
) -> dict:
    """The model's highest-predicted players, by next gameweek or the next 5 (fixture-weighted),
    with floor/median/ceiling and haul chance. For questions like 'best £6m midfielder for the
    next 5 gameweeks'. Predictions exist for the upcoming gameweek only."""
    with _read_only() as conn:
        pool = _player_pool(conn)
    pool = _filter_pool(pool, position, None, max_price)
    pool = pool[pool["raw_predicted_points"].notna()]
    if available_only:
        pool = pool[pool["status"] == "a"]
    sort_col = "horizon_points" if horizon == "5gw" else "raw_predicted_points"
    pool = pool.sort_values(sort_col, ascending=False)
    gw = _int(pool["prediction_gw"].max()) if not pool.empty else None
    return {
        "horizon": horizon, "prediction_gameweek": gw, "count": int(len(pool)),
        "players": [_pool_entry(r) for _, r in pool.head(_limit(limit)).iterrows()],
    }


def get_transfer_plan(
    horizon: Annotated[Horizon | None, Field(description="'1gw' or '5gw'. Omit to get both.")] = None,
) -> dict:
    """The model's suggested transfers for the coming gameweek - the same plans the Telegram
    approval offers. There are two: judged on next-gameweek points ('1gw') or fixture-weighted
    points over 5 gameweeks ('5gw'). `runs_if_silent` marks the one that auto-submits if
    nobody approves or rejects it (always '1gw'). xi_points_before/after are the best XI's
    points with the captain counted twice."""
    with _read_only() as conn:
        gw, rec = _recommendation(conn)
    horizons = [horizon] if horizon else ["1gw", "5gw"]
    return {
        "gameweek": gw["event_id"], "deadline_at": _iso(gw["deadline_time"]),
        "plans": [transfer_plan_payload(rec, h) for h in horizons],
    }


def get_lineup_plan(
    horizon: Annotated[Horizon, Field(description="Which transfer plan to build on: '1gw' (the one that runs if silent) or '5gw'.")] = "1gw",
) -> dict:
    """The best starting XI, bench order, captain and vice-captain for the coming gameweek,
    built from your squad AFTER the transfer plan - which is what the approval flow sets: when a
    transfer goes through, the lineup and captain are re-planned from the live squad by this
    same rule (best formation by next-gameweek points; captain and vice are the XI's top two)."""
    with _read_only() as conn:
        gw, rec = _recommendation(conn)
    return lineup_plan_payload(rec, horizon, gw["event_id"])


def get_fixtures(
    gameweeks_ahead: Annotated[int, Field(description="How many gameweeks from the next deadline (1-10).", ge=1, le=10)] = 5,
    team: Annotated[str | None, Field(description="Only fixtures involving this team short name, e.g. 'LIV'.")] = None,
) -> dict:
    """Upcoming fixtures with kickoff times and two difficulty measures per side: FPL's own
    rating (1-5) and this app's score (form-based; negative is easier), also as a 1-5 grade. The
    5-gameweek plans use the app's score, not FPL's. Also lists blanks and doubles."""
    season = _season()
    with _read_only() as conn:
        gw = _next_gameweek(conn)
        if gw is None:
            raise ToolError("no_upcoming_gameweek", "There's no upcoming deadline in the ingested data.")
        start, end = gw["event_id"], gw["event_id"] + gameweeks_ahead
        rows = _query(
            conn,
            """
            SELECT fd.fpl_fixture_id, fd.event_id, fd.kickoff_time, fd.is_home, fd.fpl_fdr,
                   fd.fixture_difficulty_score, t.short_name AS team, o.short_name AS opponent
            FROM v_fixture_difficulty fd
            JOIN teams t ON t.team_code = fd.team_code
            JOIN teams o ON o.team_code = fd.opponent_code
            WHERE fd.season = %(s)s AND fd.event_id >= %(a)s AND fd.event_id < %(b)s
            ORDER BY fd.event_id, fd.kickoff_time, fd.fpl_fixture_id
            """,
            s=season, a=start, b=end,
        )
        teams = _query(
            conn,
            "SELECT DISTINCT t.short_name FROM player_seasons ps JOIN teams t ON t.team_code = ps.team_code WHERE ps.season = %(s)s",
            s=season,
        )["short_name"].tolist()

    fixtures = []
    for fixture_id, g in rows.groupby("fpl_fixture_id", sort=False):
        home, away = g[g["is_home"]].iloc[0], g[~g["is_home"]].iloc[0]
        if team and team.upper() not in (home["team"].upper(), away["team"].upper()):
            continue
        side = lambda r: {  # noqa: E731
            "team": r["team"], "fpl_difficulty": _int(r["fpl_fdr"]),
            "app_difficulty_score": _num(r["fixture_difficulty_score"], 3),
            "app_difficulty_grade": fixture_grade(r["fixture_difficulty_score"]),
        }
        fixtures.append({"gameweek": _int(home["event_id"]), "kickoff_at": _iso(home["kickoff_time"]),
                         "home": side(home), "away": side(away)})

    counts = rows.groupby(["team", "event_id"]).size()
    blanks, doubles = [], []
    for t in sorted(teams):
        if team and t.upper() != team.upper():
            continue
        for e in range(start, end):
            n = int(counts.get((t, e), 0))
            if n == 0:
                blanks.append({"team": t, "gameweek": e})
            elif n > 1:
                doubles.append({"team": t, "gameweek": e, "fixtures": n})
    return {
        "from_gameweek": start, "to_gameweek": end - 1, "fixtures": fixtures,
        "blanks": blanks, "doubles": doubles,
        "difficulty_note": (
            "fpl_difficulty is FPL's published 1-5 rating. app_difficulty_score is this app's own "
            "measure from rolling team form (roughly zero-centred, negative = easier); "
            "app_difficulty_grade buckets it 1 (easiest) to 5 (hardest). The two often disagree."
        ),
    }


def _approval_summary(kind: str, plan: dict) -> dict:
    if kind == "transfer":
        def variant(key: str, horizon: str) -> dict:
            ts = plan.get(key, {}).get("transfers", [])
            return {
                "horizon": horizon, "runs_if_silent": horizon == "1gw",
                "transfers": [{"sell": t["sell"], "buy": t["buy"], "net": _num(t["net"]), "hit_cost": _int(t["hit"])} for t in ts],
                "total_net": _num(sum(t["net"] for t in ts)) if ts else 0.0,
            }
        return {"options": [variant("next_gw", "1gw"), variant("five_gw", "5gw")],
                "free_transfers": plan.get("free_transfers")}
    if kind == "captain":
        return {"captain": plan.get("captain_web_name"), "vice_captain": plan.get("vice_web_name")}
    return {"formation": "-".join(str(n) for n in plan.get("formation", [])),
            "starting_xi": plan.get("starting_names", []), "bench": plan.get("bench_names", [])}


def _approvals(conn: Connection, season: str, event_id: int) -> list[dict]:
    rows = _query(
        conn,
        "SELECT kind, plan_json, status, result_detail, created_at, decided_at FROM pending_approvals "
        "WHERE season = %(s)s AND event_id = %(e)s ORDER BY array_position(ARRAY['transfer','captain','squad'], kind)",
        s=season, e=event_id,
    )
    return [{
        "kind": r["kind"], "status": r["status"], "plan": _approval_summary(r["kind"], r["plan_json"] or {}),
        "result": _text(r["result_detail"]), "proposed_at": _iso(r["created_at"]), "decided_at": _iso(r["decided_at"]),
    } for _, r in rows.iterrows()]


def get_pending_approvals(
    gameweek: Annotated[int | None, Field(description="Gameweek id; defaults to the next deadline.", ge=1, le=38)] = None,
) -> dict:
    """The Telegram approvals for a gameweek (transfer, captain, lineup): what each proposes, its
    status (proposed / approved / rejected / executing / executed / failed), the result, and
    when anything undecided will auto-submit. Read-only: this can't approve or reject."""
    season = _season()
    with _read_only() as conn:
        gw = _next_gameweek(conn)
        event_id = gameweek or (gw["event_id"] if gw else None)
        if event_id is None:
            raise ToolError("no_upcoming_gameweek", "There's no upcoming deadline; pass a gameweek.")
        approvals = _approvals(conn, season, event_id)
        deadline = _query(conn, "SELECT deadline_time FROM gameweeks WHERE season = %(s)s AND event_id = %(e)s",
                          s=season, e=event_id)
    deadline_at = deadline["deadline_time"].iloc[0] if not deadline.empty else None
    return {
        "gameweek": event_id, "deadline_at": _iso(deadline_at), "hours_remaining": _hours_until(deadline_at),
        "approvals": approvals,
        "auto_submit": AUTO_SUBMIT_WINDOW,
        "note": None if approvals else (
            "No approvals sent for this gameweek yet. They go out in the T-3h tier, which opens up to "
            "6 hours before the deadline."
        ),
    }


def get_gameweek_review(
    gameweek: Annotated[int | None, Field(description="Gameweek id; defaults to the last finished gameweek.", ge=1, le=38)] = None,
) -> dict:
    """A finished gameweek side by side: what the model proposed (the approvals), what you
    fielded, and the actual points - including what the model's proposed XI and captain would
    have scored. Overrides are listed once they exist (planned; empty for now)."""
    if not settings.ENTRY_ID:
        raise ToolError("not_configured", "FPL_ENTRY_ID isn't set, so there's nothing to review.")
    season, entry = _season(), settings.ENTRY_ID
    with _read_only() as conn:
        if gameweek is None:
            last = _query(conn, "SELECT MAX(event_id) AS e FROM gameweeks WHERE season = %(s)s AND finished", s=season)
            gameweek = _int(last["e"].iloc[0])
            if gameweek is None:
                raise ToolError("no_finished_gameweek", "No gameweek has finished yet.")
        gw = _query(conn, "SELECT event_id, name, deadline_time, finished FROM gameweeks WHERE season = %(s)s AND event_id = %(e)s",
                    s=season, e=gameweek)
        if gw.empty:
            raise ToolError("unknown_gameweek", f"GW{gameweek} isn't in the ingested data.")
        result = _query(conn, "SELECT * FROM v_manager_gw_comparison WHERE entry_id = %(n)s AND season = %(s)s AND event_id = %(e)s",
                        n=entry, s=season, e=gameweek)
        history = _query(conn, "SELECT * FROM manager_gameweeks WHERE entry_id = %(n)s AND season = %(s)s AND event_id = %(e)s",
                         n=entry, s=season, e=gameweek)
        picks = _query(
            conn,
            """
            SELECT sp.player_code, p.web_name, sp.squad_position, sp.multiplier, sp.is_captain,
                   (SELECT SUM(total_points) FROM player_gameweek_stats s
                     WHERE s.season = sp.season AND s.event_id = sp.event_id AND s.player_code = sp.player_code) AS actual,
                   (SELECT predicted_points FROM predictions pr
                     WHERE pr.season = sp.season AND pr.event_id = sp.event_id AND pr.player_code = sp.player_code
                     ORDER BY predicted_at DESC LIMIT 1) AS predicted
            FROM squad_picks sp JOIN players p ON p.player_code = sp.player_code
            WHERE sp.entry_id = %(n)s AND sp.season = %(s)s AND sp.event_id = %(e)s
            ORDER BY sp.squad_position
            """,
            n=entry, s=season, e=gameweek,
        )
        approvals = _approvals(conn, season, gameweek)
        raw = _query(conn, "SELECT kind, plan_json FROM pending_approvals WHERE season = %(s)s AND event_id = %(e)s",
                     s=season, e=gameweek)
        plans = {r["kind"]: r["plan_json"] or {} for _, r in raw.iterrows()}
        model_codes = list(plans.get("squad", {}).get("starting_player_codes", []))
        captain_code = plans.get("captain", {}).get("captain_player_code")
        actual_by_code = {}
        codes = sorted(set(model_codes) | ({captain_code} if captain_code else set()))
        if codes:
            a = _query(
                conn,
                "SELECT player_code, SUM(total_points) AS pts FROM player_gameweek_stats "
                "WHERE season = %(s)s AND event_id = %(e)s AND player_code = ANY(%(c)s) GROUP BY player_code",
                s=season, e=gameweek, c=codes,
            )
            actual_by_code = dict(zip(a["player_code"], a["pts"]))

    fielded = picks[picks["multiplier"] > 0]
    fielded_points = float((fielded["actual"].fillna(0) * fielded["multiplier"]).sum()) if not fielded.empty else None
    model_points = None
    if model_codes:
        model_points = float(sum(actual_by_code.get(c, 0) or 0 for c in model_codes))
        if captain_code:
            model_points += float(actual_by_code.get(captain_code, 0) or 0)
    r = result.iloc[0] if not result.empty else None
    h = history.iloc[0] if not history.empty else None
    return {
        "gameweek": gameweek, "name": gw["name"].iloc[0], "finished": bool(gw["finished"].iloc[0]),
        "result": None if h is None else {
            "official_points": _int(h["points"]), "transfer_cost": _int(h["event_transfers_cost"]),
            "points_on_bench": _int(h["points_on_bench"]), "overall_rank": _int(h["overall_rank"]),
            "gameweek_rank": _int(h["event_rank"]), "chip": _text(h["active_chip"]),
            "gameweek_average": _int(r["gw_average"]) if r is not None else None,
            "gameweek_highest": _int(r["gw_highest"]) if r is not None else None,
            "model_predicted_points": _num(r["model_predicted_points"]) if r is not None else None,
            "fpl_predicted_points": _num(r["epl_predicted_points"]) if r is not None else None,
        },
        "fielded": [{
            "player_code": _int(p["player_code"]), "web_name": p["web_name"], "starting": bool(p["multiplier"] > 0),
            "captain": bool(p["is_captain"]), "multiplier": _int(p["multiplier"]),
            "predicted_points": _num(p["predicted"]), "actual_points": _int(p["actual"]),
        } for _, p in picks.iterrows()],
        "model_vs_you": {
            "your_xi_points": _num(fielded_points, 0),
            "model_proposal_xi_points": _num(model_points, 0),
            "difference": _num(fielded_points - model_points, 0) if fielded_points is not None and model_points is not None else None,
            "basis": "Actual points of each starting XI with the captain doubled, before auto-subs and transfer costs.",
        },
        "approvals": approvals,
        "overrides": [],
        "overrides_note": "Overrides aren't recorded yet; they arrive with gameweek plans.",
    }


TOOLS = [
    get_status, get_current_gameweek, get_squad, search_players, get_top_predictions,
    get_transfer_plan, get_lineup_plan, get_fixtures, get_pending_approvals, get_gameweek_review,
]
