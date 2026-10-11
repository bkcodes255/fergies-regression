"""Smoke check for the read-only tools server's tools against your real database - no server,
no network exposure: calls each tool function directly and prints a one-line summary, or the
error. Run this before hosting anything, ideally connected as the read-only role.

Run from the repo root (needs TOOLS_DATABASE_URL or DATABASE_URL, FPL_SEASON, FPL_ENTRY_ID in .env):
    python scripts/check_tools_server.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools_server.tools import TOOLS, ToolError  # noqa: E402


def summary(name: str, data: dict) -> str:
    if name == "get_status":
        return f"next deadline {data['next_deadline']}, warnings {data['warnings']}"
    if name == "get_squad":
        return f"GW{data['as_of_gameweek']} squad, {len(data['players'])} players, bank £{data['bank']}m, {data['free_transfers']} FT"
    if name == "get_transfer_plan":
        return "; ".join(f"{p['horizon']}: {[t['sell']['web_name'] + '->' + t['buy']['web_name'] for t in p['transfers']]}" for p in data["plans"])
    if name == "get_lineup_plan":
        return f"{data['formation']}, captain {data['captain']['web_name']}, built on {data['built_on_transfers']}"
    if name == "get_pending_approvals":
        return f"GW{data['gameweek']}: {[(a['kind'], a['status']) for a in data['approvals']]}"
    if name == "get_gameweek_review":
        return f"GW{data['gameweek']}: {data['model_vs_you']}"
    if "players" in data:
        return f"{data.get('count', len(data['players']))} players"
    if "fixtures" in data:
        return f"{len(data['fixtures'])} fixtures, blanks {data['blanks']}, doubles {data['doubles']}"
    return str(data)[:160]


failed = 0
for tool in TOOLS:
    try:
        print(f"OK   {tool.__name__}: {summary(tool.__name__, tool())}")
    except ToolError as exc:
        print(f"--   {tool.__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        failed += 1
        print(f"FAIL {tool.__name__}: {exc!r}")
sys.exit(1 if failed else 0)
