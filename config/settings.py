"""Environment-backed configuration. Copy .env.example to .env and fill it in."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

# model_versions.artifact_path is stored relative to this (just the filename) so a model
# trained on one machine can be loaded from any checkout - previously an absolute path baked
# in at train time, which only ever resolved on the machine that trained it.
MODELS_DIR = REPO_ROOT / "models"

DATABASE_URL = os.environ.get("DATABASE_URL")
SEASON = os.environ.get("FPL_SEASON", "2026-27")
_entry_id = os.environ.get("FPL_ENTRY_ID")
ENTRY_ID = int(_entry_id) if _entry_id else None

# Only needed for src/fpl_write - never log or print these.
FPL_EMAIL = os.environ.get("FPL_EMAIL")
FPL_PASSWORD = os.environ.get("FPL_PASSWORD")
# Manual cookie handoff - the programmatic email/password login flow no longer works (see
# src/fpl_write/client.py docstring). Extracted from a real browser session's /api/me/ request.
FPL_SESSION_COOKIE = os.environ.get("FPL_SESSION_COOKIE")
# The actual auth credential FPL's rebuilt frontend uses for API calls - a custom header, not
# the Cookie header. Extracted the same way, from the same /api/me/ request's Request Headers.
FPL_API_AUTHORIZATION = os.environ.get("FPL_API_AUTHORIZATION")

# Only needed for src/notify (Telegram bot). Token from @BotFather; chat ID captured by
# running src/notify/telegram_bot.py once and messaging the bot (it prints the chat_id).
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
_telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
TELEGRAM_CHAT_ID = int(_telegram_chat_id) if _telegram_chat_id else None

# Only needed for scripts/scrape_crests_to_supabase.py. SUPABASE_SERVICE_ROLE_KEY bypasses
# RLS - keep it out of .env.example's tracked defaults and never log or print it.
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_CRESTS_BUCKET = os.environ.get("SUPABASE_CRESTS_BUCKET", "team-crests")
