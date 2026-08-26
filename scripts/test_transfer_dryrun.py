"""One-off manual test: a DRY-RUN transfer preview only (confirmed: false) - does NOT execute a
real transfer. Goal is just to see whether /api/transfers/ accepts this payload shape at all,
and if not, read the validation error to learn the real field names, before ever risking a
confirmed=true call. Delete once the transfer write path is confirmed and folded into the real
submit flow.

Run from the repo root:
    .\\.venv\\Scripts\\python.exe scripts\\test_transfer_dryrun.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fpl_write.client import login_via_cookie

TRANSFERS_URL = "https://fantasy.premierleague.com/api/transfers/"
ENTRY_ID = 8851945
NEXT_EVENT = 2  # GW1 is finished, GW2 hasn't started - this transfer would apply to GW2
ELEMENT_OUT = 329  # Rodon (DEF, LEE), bench, 4.5m
ELEMENT_IN = 473  # Aina (DEF, NFO), 4.5m - same price/position, zero squad overlap

session = login_via_cookie()

payload = {
    "confirmed": False,
    "entry": ENTRY_ID,
    "event": NEXT_EVENT,
    "transfers": [
        {
            "element_in": ELEMENT_IN,
            "element_out": ELEMENT_OUT,
            "purchase_price": 45,
            "selling_price": 45,
        }
    ],
    "wildcard": False,
    "freehit": False,
}

resp = session.post(TRANSFERS_URL, json=payload, timeout=15)
print("STATUS:", resp.status_code)
print("BODY:", resp.text[:1500])
