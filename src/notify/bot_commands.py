"""Instruction-execution commands for the Telegram bot: /captain and /transfer let Brian tell
the bot to change his live FPL team, gated behind an explicit yes/no confirmation before any
write actually fires - see the project rule (src/notify/deadline_scheduler.py's docstring)
against ever submitting a transfer without a real, deliberate, user-approved confirmation.

Command syntax, not free-form natural language, is deliberate: this executes against a real
account, and a misread instruction has real cost (a wasted free transfer, a -4 hit, or a wrong
captain). A fixed grammar is far less likely to be misparsed than open NLP, and any resolution
ambiguity (multiple players matching a name) is surfaced back to Brian to pick from rather than
guessed at.

Pending-action state is a plain in-memory dict keyed by chat_id, with a short TTL - losing an
unconfirmed action on a process restart is the safe failure direction (Brian just re-sends the
instruction); a DB-backed table would be new complexity for no real safety gain on what's a
single-user bot.

This is a long-running process (unlike deadline_scheduler.py's one-shot script runs), so DB
reads go through src.ingestion.db.get_engine() - the pooled, pre-ping engine dashboard/app.py
uses for the same reason (see that module's docstring) - not get_connection(), which is only
safe for a connect-do-work-exit script.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from rapidfuzz import fuzz
from telegram import Update
from telegram.ext import ContextTypes

from config import settings
from src.fpl_write.client import FPLLoginError, get_my_team, login_via_cookie, set_captain, submit_transfer
from src.ingestion.db import get_engine
from src.notify.deadline_scheduler import load_next_deadline, load_rankings, load_squad
from src.recommendations.transfers import validate_transfer

PENDING_TTL_SECONDS = 5 * 60
FUZZY_THRESHOLD = 80

AUTH_EXPIRED_HINT = (
    "\n\nYour FPL_API_AUTHORIZATION has likely expired - re-extract it from a browser session "
    "(see .env.example) and update it."
)


@dataclass
class PendingAction:
    execute: Callable[[], dict]
    summary: str
    created_at: float


_pending: dict[int, PendingAction] = {}


def _resolve_player(name: str, pool: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    """Fuzzy-matches `name` against pool['web_name'] (a squad or rankings DataFrame). Returns
    (row, None) on a confident single match, or (None, message) on no match / ambiguous match -
    ambiguity is reported back to Brian to disambiguate, never guessed at."""
    if pool.empty:
        return None, "No player data loaded."
    name_norm = name.strip().lower()

    exact = pool[pool["web_name"].str.lower() == name_norm]
    if len(exact) == 1:
        return exact.iloc[0], None
    if len(exact) > 1:
        options = ", ".join(f"{r.web_name} ({r.team})" for r in exact.itertuples())
        return None, f"Multiple players named '{name}': {options}. Include the team to disambiguate."

    scores = pool["web_name"].map(lambda w: fuzz.token_sort_ratio(name_norm, w.lower()))
    best_score = scores.max()
    if best_score < FUZZY_THRESHOLD:
        return None, f"No player found matching '{name}'."
    matches = pool[scores == best_score]
    if len(matches) > 1:
        options = ", ".join(f"{r.web_name} ({r.team})" for r in matches.itertuples())
        return None, f"Multiple close matches for '{name}': {options}. Be more specific."
    return matches.iloc[0], None


def _load_context(engine) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Everything a command needs: this entry's next-deadline gameweek, live squad, and the
    full ranked player pool. Raises RuntimeError with a user-facing message on any gap."""
    if not settings.ENTRY_ID:
        raise RuntimeError("FPL_ENTRY_ID not set in .env.")
    gw = load_next_deadline(engine)
    if gw is None:
        raise RuntimeError("No upcoming gameweek found - nothing to act on.")
    squad, _ = load_squad(engine, gw["season"], settings.ENTRY_ID)
    if squad.empty:
        raise RuntimeError("No squad data ingested yet for this entry.")
    rankings = load_rankings(engine, gw["season"])
    return gw, squad, rankings


async def captain_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("Usage: /captain <player name>")
        return
    name = " ".join(context.args)

    try:
        gw, squad, _ = _load_context(get_engine())
    except RuntimeError as exc:
        await update.message.reply_text(str(exc))
        return

    player, error = _resolve_player(name, squad)
    if error:
        await update.message.reply_text(error)
        return

    try:
        session = login_via_cookie()
        my_team = get_my_team(session, settings.ENTRY_ID)
    except FPLLoginError as exc:
        await update.message.reply_text(f"FPL login failed: {exc}{AUTH_EXPIRED_HINT}")
        return

    current_element = next((p["element"] for p in my_team["picks"] if p["is_captain"]), None)
    current_name_rows = squad.loc[squad["fpl_id"] == current_element, "web_name"]
    current_name = current_name_rows.iloc[0] if not current_name_rows.empty else "unknown"

    captain_element = int(player["fpl_id"])

    def execute() -> dict:
        exec_session = login_via_cookie()
        fresh_team = get_my_team(exec_session, settings.ENTRY_ID)
        vice_element = next(
            (p["element"] for p in fresh_team["picks"] if p["is_vice_captain"]), captain_element
        )
        response = set_captain(exec_session, settings.ENTRY_ID, fresh_team, captain_element, vice_element)
        return {"ok": response.status_code == 200, "status": response.status_code, "body": response.text[:300]}

    _pending[chat_id] = PendingAction(
        execute=execute,
        summary=f"Set captain: {player['web_name']} (currently {current_name})",
        created_at=time.monotonic(),
    )
    await update.message.reply_text(
        f"Set captain to {player['web_name']} (currently {current_name})?\nReply yes/no."
    )


async def transfer_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    text = " ".join(context.args)
    if " for " not in text:
        await update.message.reply_text("Usage: /transfer <player to sell> for <player to buy>")
        return
    out_name, in_name = (part.strip() for part in text.split(" for ", 1))

    try:
        gw, squad, rankings = _load_context(get_engine())
    except RuntimeError as exc:
        await update.message.reply_text(str(exc))
        return

    out_player, error = _resolve_player(out_name, squad)
    if error:
        await update.message.reply_text(f"Selling '{out_name}': {error}")
        return
    in_player, error = _resolve_player(in_name, rankings)
    if error:
        await update.message.reply_text(f"Buying '{in_name}': {error}")
        return

    try:
        session = login_via_cookie()
        my_team = get_my_team(session, settings.ENTRY_ID)
    except FPLLoginError as exc:
        await update.message.reply_text(f"FPL login failed: {exc}{AUTH_EXPIRED_HINT}")
        return

    live_transfers = my_team.get("transfers", {})
    bank = (live_transfers.get("bank") or 0) / 10
    free_transfers = live_transfers.get("limit")
    free_transfers = 0 if free_transfers is None else free_transfers

    error = validate_transfer(squad, rankings, bank, out_player["player_code"], in_player["player_code"])
    if error:
        await update.message.reply_text(f"Can't make that transfer: {error}")
        return

    hit_note = "" if free_transfers > 0 else " (costs a -4 hit - no free transfers left)"
    out_element = int(out_player["fpl_id"])
    in_element = int(in_player["fpl_id"])
    event_id = gw["event_id"]
    # purchase_price falls back to the DB's ingested price if a fresh call's picks don't carry
    # selling_price - my-team's exact live shape isn't confirmed (see submit_transfer's
    # docstring), so this is a best-effort guess consistent with that same caveat.
    fallback_selling_price = int(round(out_player["price"] * 10))
    purchase_price = int(round(in_player["price"] * 10))

    def execute() -> dict:
        exec_session = login_via_cookie()
        fresh_team = get_my_team(exec_session, settings.ENTRY_ID)
        pick = next((p for p in fresh_team["picks"] if p["element"] == out_element), None)
        selling_price = (pick or {}).get("selling_price", fallback_selling_price)
        response = submit_transfer(
            exec_session, settings.ENTRY_ID, event_id,
            out_element, in_element, purchase_price, selling_price,
        )
        return {"ok": response.status_code == 200, "status": response.status_code, "body": response.text[:300]}

    _pending[chat_id] = PendingAction(
        execute=execute,
        summary=f"Transfer: {out_player['web_name']} -> {in_player['web_name']}{hit_note}",
        created_at=time.monotonic(),
    )
    await update.message.reply_text(
        f"Sell {out_player['web_name']}, buy {in_player['web_name']}{hit_note}?\nReply yes/no."
    )


async def handle_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Checked before the generic echo handler. Returns True if this message was consumed as a
    reply to a pending action, False if the caller should fall through to default echo
    handling (no pending action, or the text isn't a yes/no reply at all)."""
    chat_id = update.effective_chat.id
    text = (update.message.text or "").strip().lower()
    if text not in ("yes", "y", "confirm", "no", "n", "cancel"):
        return False

    pending = _pending.get(chat_id)
    if pending is None:
        return False

    if time.monotonic() - pending.created_at > PENDING_TTL_SECONDS:
        del _pending[chat_id]
        await update.message.reply_text("That confirmation expired - resend the instruction.")
        return True

    if text in ("no", "n", "cancel"):
        del _pending[chat_id]
        await update.message.reply_text("Cancelled.")
        return True

    del _pending[chat_id]
    await update.message.reply_text(f"Submitting: {pending.summary} ...")
    try:
        result = pending.execute()
    except FPLLoginError as exc:
        await update.message.reply_text(f"FPL login failed: {exc}{AUTH_EXPIRED_HINT}")
        return True

    if result["ok"]:
        await update.message.reply_text(f"Done: {pending.summary}")
    else:
        await update.message.reply_text(f"FPL rejected it (HTTP {result['status']}): {result['body']}")
    return True
