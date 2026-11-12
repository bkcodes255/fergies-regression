"""One-off manual test: fires the no-op my-team write (src/fpl_write/client.py) and prints the
result. Not part of the regular pipeline - delete once the write path is confirmed working and
folded into the real submit flow.

Run from the repo root:
    .\\.venv\\Scripts\\python.exe scripts\\test_noop_write.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ENTRY_ID
from src.fpl_write.client import get_my_team, login_via_cookie, save_my_team_noop

if ENTRY_ID is None:
    sys.exit("FPL_ENTRY_ID is not set in .env")

session = login_via_cookie()
entry_id = ENTRY_ID
my_team = get_my_team(session, entry_id)
resp = save_my_team_noop(session, entry_id, my_team)
print("STATUS:", resp.status_code)
print("BODY (first 500 chars):", resp.text[:500])
