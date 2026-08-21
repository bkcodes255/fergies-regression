"""Environment-backed configuration. Copy .env.example to .env and fill it in."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL")
SEASON = os.environ.get("FPL_SEASON", "2026-27")
