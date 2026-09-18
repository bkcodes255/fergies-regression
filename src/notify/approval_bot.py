"""Telegram approval workflow - the piece that closes the gap deadline_scheduler.py's
recommendation-only reminders left open (see its own docstring): this is what actually calls
src/fpl_write/client.py's write functions, but only after Brian approves via Telegram, or a
silent timeout at T-30m (his explicit choice - see project memory).

Three independent approval items per gameweek - transfer plan, captain pick, starting XI/bench
- each with its own Approve/Reject button pair, tracked as its own row in `pending_approvals`
(sql/pending_approvals.sql). Independent by design: approving the captain pick shouldn't be
blocked on, or imply anything about, the transfer plan or lineup.

No persistent process (GitHub Actions runs are ephemeral, one-shot per cron tick - see
deadline_reminders.yml) - `poll_and_process()` uses Bot.get_updates(offset=...) to fetch
whatever callback-query button presses arrived since the last run, processes them, and
persists the new offset in `telegram_update_offset`. Called at the top of every
deadline_scheduler.py run, independent of which reminder tier is due, so a button tap is acted
on within ~15 minutes (the scheduler's own cron cadence) rather than only at the next tier.

Run directly for a one-off manual poll (needs the same env as deadline_scheduler.py):
    python -m src.notify.approval_bot
"""
from __future__ import annotations

import asyncio
import json

import pandas as pd
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

from config import settings
from src.fpl_write.client import FPLLoginError, get_my_team, login_via_cookie, set_captain, set_lineup, submit_transfers

KINDS = ("transfer", "captain", "squad")
_KIND_LABEL = {"transfer": "Transfer plan", "captain": "Captain pick", "squad": "Starting XI/bench"}


def _bot() -> Bot:
    if not settings.TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in .env - see .env.example.")
    return Bot(token=settings.TELEGRAM_BOT_TOKEN)


# --- plan building (snapshot for display + the record poll_and_process/auto_submit_expired act on) ---

def build_transfer_plan(plan_df: pd.DataFrame, free_transfers: int) -> dict:
    return {
        "transfers": [
            {"sell": row["sell"], "buy": row["buy"], "net": row["net"], "hit": row["hit"]}
            for _, row in plan_df.iterrows()
        ],
        "free_transfers": free_transfers,
    }


def build_captain_plan(captain_code: int, captain_name: str, vice_code: int, vice_name: str) -> dict:
    return {
        "captain_player_code": int(captain_code), "captain_web_name": captain_name,
        "vice_player_code": int(vice_code), "vice_web_name": vice_name,
    }


def build_squad_plan(starting_xi: pd.DataFrame, bench: pd.DataFrame, formation: tuple[int, int, int]) -> dict:
    return {
        "starting_player_codes": [int(c) for c in starting_xi["player_code"]],
        "starting_names": list(starting_xi["web_name"]),
        "bench_player_codes": [int(c) for c in bench["player_code"]],
        "bench_names": list(bench["web_name"]),
        "formation": list(formation),
    }


def _format_plan_text(kind: str, plan: dict) -> str:
    if kind == "transfer":
        if not plan["transfers"]:
            return "No transfers recommended this week."
        lines = [f"  {t['sell']} -> {t['buy']} (net {t['net']:+.1f})" for t in plan["transfers"]]
        return "\n".join(lines)
    if kind == "captain":
        return f"  Captain: {plan['captain_web_name']} / Vice: {plan['vice_web_name']}"
    return f"  {plan['formation'][0]}-{plan['formation'][1]}-{plan['formation'][2]}: {', '.join(plan['starting_names'])}\n  Bench: {', '.join(plan['bench_names'])}"


# --- DB helpers ---

def create_pending_approval(conn, season: str, event_id: int, kind: str, plan: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO pending_approvals (season, event_id, kind, plan_json) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (season, event_id, kind) DO NOTHING",
            (season, event_id, kind, json.dumps(plan)),
        )


def _set_message_id(conn, season: str, event_id: int, kind: str, message_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE pending_approvals SET telegram_message_id = %s "
            "WHERE season = %s AND event_id = %s AND kind = %s AND telegram_message_id IS NULL",
            (message_id, season, event_id, kind),
        )


def _claim(conn, season: str, event_id: int, kind: str) -> dict | None:
    """Atomically moves a row from 'proposed' to 'executing' and returns its plan, or None if
    it wasn't in 'proposed' (already decided/executed, or doesn't exist) - the guard that keeps
    a button-tap and the T-30m timeout from ever double-executing the same approval."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE pending_approvals SET status = 'executing' "
            "WHERE season = %s AND event_id = %s AND kind = %s AND status = 'proposed' "
            "RETURNING plan_json",
            (season, event_id, kind),
        )
        row = cur.fetchone()
    return row[0] if row else None


def _finish(conn, season: str, event_id: int, kind: str, status: str, detail: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE pending_approvals SET status = %s, result_detail = %s, decided_at = now() "
            "WHERE season = %s AND event_id = %s AND kind = %s",
            (status, detail[:2000], season, event_id, kind),
        )


def _reject(conn, season: str, event_id: int, kind: str) -> bool:
    """Like _claim but for a Reject tap - only succeeds from 'proposed', same double-fire guard."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE pending_approvals SET status = 'rejected', decided_at = now() "
            "WHERE season = %s AND event_id = %s AND kind = %s AND status = 'proposed'",
            (season, event_id, kind),
        )
        return cur.rowcount > 0


def _get_offset(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT last_update_id FROM telegram_update_offset WHERE id = 1")
        return cur.fetchone()[0]


def _save_offset(conn, update_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE telegram_update_offset SET last_update_id = %s WHERE id = 1", (update_id,))


def _fpl_id_map(conn, season: str, player_codes: list[int]) -> dict[int, int]:
    """player_code (stable) -> this season's numeric FPL element id - the id the write API
    actually uses. Never assume player_code IS the element id (see the season-scoped-id gotcha
    already documented elsewhere in this project)."""
    if not player_codes:
        return {}
    df = pd.read_sql_query(
        "SELECT player_code, fpl_id FROM player_seasons WHERE season = %(s)s AND player_code = ANY(%(codes)s)",
        conn, params={"s": season, "codes": player_codes},
    )
    return dict(zip(df["player_code"], df["fpl_id"]))


def _current_prices(conn, season: str, player_codes: list[int]) -> dict[int, int]:
    """Freshest ingested now_cost (FPL's *10 integer format) per player_code - used for a
    transfer target's purchase_price, since it isn't in my-team's picks (that only prices
    players you already own)."""
    if not player_codes:
        return {}
    df = pd.read_sql_query(
        """
        SELECT DISTINCT ON (player_code) player_code, now_cost
        FROM player_price_snapshots
        WHERE season = %(s)s AND player_code = ANY(%(codes)s)
        ORDER BY player_code, snapshot_date DESC
        """,
        conn, params={"s": season, "codes": player_codes},
    )
    return dict(zip(df["player_code"], df["now_cost"]))


# --- sending ---

async def send_approval_requests(
    conn, season: str, event_id: int, gw_name: str,
    transfer_plan: dict, captain_plan: dict, squad_plan: dict,
) -> None:
    plans = {"transfer": transfer_plan, "captain": captain_plan, "squad": squad_plan}
    for kind in KINDS:
        create_pending_approval(conn, season, event_id, kind, plans[kind])

    lines = [f"\U0001F4CB {gw_name} - review before the deadline:\n"]
    keyboard = []
    for kind in KINDS:
        lines.append(f"{_KIND_LABEL[kind]}:\n{_format_plan_text(kind, plans[kind])}\n")
        tag = f"{kind}:{season}:{event_id}"
        keyboard.append([
            InlineKeyboardButton(f"✅ Approve {_KIND_LABEL[kind]}", callback_data=f"appr:{tag}"),
            InlineKeyboardButton(f"❌ Skip", callback_data=f"rej:{tag}"),
        ])
    lines.append(
        "\n⚠️ Anything you don't respond to auto-submits within the last ~2h before "
        "the deadline (your standing default, timing depends on when GitHub's scheduler actually "
        "runs) - tap Skip now if you don't want that."
    )

    bot = _bot()
    message = await bot.send_message(
        chat_id=settings.TELEGRAM_CHAT_ID, text="\n".join(lines),
        reply_markup=InlineKeyboardMarkup(keyboard),
    )
    for kind in KINDS:
        _set_message_id(conn, season, event_id, kind, message.message_id)


# --- execution (the part that actually calls src/fpl_write/client.py) ---

def execute_kind(conn, season: str, event_id: int, kind: str, plan: dict) -> tuple[bool, str]:
    """Returns (ok, detail). Never raises past this point - a failure here must produce a clear
    Telegram message, not a silently swallowed exception, since the caller has no other way to
    find out a submission didn't happen."""
    try:
        session = login_via_cookie()
    except FPLLoginError as exc:
        return False, f"FPL auth failed - nothing was submitted. Re-extract the session cookie/header. ({exc})"

    entry_id = settings.ENTRY_ID
    try:
        if kind == "transfer":
            return _execute_transfer(conn, session, entry_id, season, event_id, plan)
        if kind == "captain":
            return _execute_captain(conn, session, entry_id, season, plan)
        return _execute_squad(conn, session, entry_id, season, plan)
    except Exception as exc:  # noqa: BLE001 - must always resolve to a reported result, never crash the poller
        return False, f"Unexpected error executing {kind} plan: {exc!r}"


def _execute_transfer(conn, session, entry_id: int, season: str, event_id: int, plan: dict) -> tuple[bool, str]:
    transfers = plan["transfers"]
    if not transfers:
        return True, "No transfers to submit this week."

    my_team = get_my_team(session, entry_id)
    selling_price = {p["element"]: p["selling_price"] for p in my_team["picks"]}
    sell_codes = [_lookup_code_by_name(conn, season, t["sell"]) for t in transfers]
    buy_codes = [_lookup_code_by_name(conn, season, t["buy"]) for t in transfers]
    id_map = _fpl_id_map(conn, season, sell_codes + buy_codes)
    buy_prices = _current_prices(conn, season, buy_codes)

    payload_transfers = []
    for t, sell_code, buy_code in zip(transfers, sell_codes, buy_codes):
        sell_element = id_map[sell_code]
        buy_element = id_map[buy_code]
        if sell_element not in selling_price:
            return False, f"{t['sell']} isn't currently in your squad - not submitting (plan is stale)."
        payload_transfers.append({
            "element_in": buy_element, "element_out": sell_element,
            "purchase_price": int(buy_prices[buy_code]), "selling_price": int(selling_price[sell_element]),
        })

    resp = submit_transfers(session, entry_id, event_id, payload_transfers)
    if resp.status_code not in (200, 202):
        return False, f"Transfer submission failed - HTTP {resp.status_code}: {resp.text[:500]}"
    return True, f"Submitted {len(payload_transfers)} transfer(s): " + ", ".join(f"{t['sell']}->{t['buy']}" for t in transfers)


def _execute_captain(conn, session, entry_id: int, season: str, plan: dict) -> tuple[bool, str]:
    id_map = _fpl_id_map(conn, season, [plan["captain_player_code"], plan["vice_player_code"]])
    captain_element = id_map[plan["captain_player_code"]]
    vice_element = id_map[plan["vice_player_code"]]
    my_team = get_my_team(session, entry_id)
    resp = set_captain(session, entry_id, my_team, captain_element, vice_element)
    if resp.status_code not in (200, 202):
        return False, f"Captain change failed - HTTP {resp.status_code}: {resp.text[:500]}"
    return True, f"Captain set to {plan['captain_web_name']} (vice {plan['vice_web_name']})."


def _execute_squad(conn, session, entry_id: int, season: str, plan: dict) -> tuple[bool, str]:
    codes = plan["starting_player_codes"] + plan["bench_player_codes"]
    id_map = _fpl_id_map(conn, season, codes)
    starting_elements = [id_map[c] for c in plan["starting_player_codes"]]
    bench_elements = [id_map[c] for c in plan["bench_player_codes"]]
    my_team = get_my_team(session, entry_id)
    try:
        resp = set_lineup(session, entry_id, my_team, starting_elements, bench_elements)
    except ValueError as exc:
        return False, f"Lineup not applied: {exc}"
    if resp.status_code not in (200, 202):
        return False, f"Lineup change failed - HTTP {resp.status_code}: {resp.text[:500]}"
    return True, f"Lineup set: {plan['formation'][0]}-{plan['formation'][1]}-{plan['formation'][2]}."


def _lookup_code_by_name(conn, season: str, web_name: str) -> int:
    """Transfer plans are stored/displayed by web_name (matches deadline_scheduler.py's
    existing convention) - resolve back to player_code against the current squad+rankings at
    execution time. Ambiguous/missing names fail loudly rather than guessing."""
    df = pd.read_sql_query(
        "SELECT player_code FROM players WHERE web_name = %(n)s", conn, params={"n": web_name},
    )
    if len(df) != 1:
        raise ValueError(f"Could not uniquely resolve player web_name={web_name!r} ({len(df)} matches)")
    return int(df.iloc[0]["player_code"])


async def _reply(chat_id: int, text: str) -> None:
    await _bot().send_message(chat_id=chat_id, text=text)


# --- polling (the entry point deadline_scheduler.py calls every run) ---

async def poll_and_process(conn) -> None:
    bot = _bot()
    offset = _get_offset(conn)
    updates = await bot.get_updates(offset=offset + 1, timeout=0)
    for update in updates:
        _save_offset(conn, update.update_id)
        cq = update.callback_query
        if cq is None:
            continue
        if cq.from_user is None or cq.from_user.id != settings.TELEGRAM_CHAT_ID:
            await bot.answer_callback_query(cq.id, text="Not authorized.")
            continue
        await bot.answer_callback_query(cq.id)

        action, kind, season, event_id_str = cq.data.split(":")
        event_id = int(event_id_str)
        if kind not in KINDS:
            continue

        if action == "rej":
            if _reject(conn, season, event_id, kind):
                await _reply(cq.from_user.id, f"Skipped: {_KIND_LABEL[kind]} for event {event_id}.")
            continue

        plan = _claim(conn, season, event_id, kind)
        if plan is None:
            continue  # already decided/executed elsewhere (e.g. the T-30m timeout beat this tap)
        ok, detail = execute_kind(conn, season, event_id, kind, plan)
        _finish(conn, season, event_id, kind, "executed" if ok else "failed", detail)
        prefix = "✅" if ok else "⚠️"
        await _reply(cq.from_user.id, f"{prefix} {_KIND_LABEL[kind]}: {detail}")


def auto_submit_expired(conn, season: str, event_id: int) -> None:
    """Called from deadline_scheduler's T-30m tier: anything still 'proposed' for this
    gameweek gets executed now, per Brian's explicit standing choice that a silent timeout
    should still act rather than do nothing."""
    bot = _bot()
    for kind in KINDS:
        plan = _claim(conn, season, event_id, kind)
        if plan is None:
            continue
        ok, detail = execute_kind(conn, season, event_id, kind, plan)
        _finish(conn, season, event_id, kind, "executed" if ok else "failed", detail)
        prefix = "✅" if ok else "⚠️"
        asyncio.run(bot.send_message(
            chat_id=settings.TELEGRAM_CHAT_ID,
            text=f"{prefix} (auto, no response received) {_KIND_LABEL[kind]}: {detail}",
        ))


if __name__ == "__main__":
    from src.ingestion.db import get_connection

    _conn = get_connection()
    _conn.autocommit = True
    asyncio.run(poll_and_process(_conn))
