"""Emergency revert: undoes the accidental real transfer from test_transfer_dryrun.py
(confirmed: false did NOT prevent execution - Rodon was sold, Aina was bought for real).
Buys Rodon back, sells Aina, restoring the original squad. Safe pre-deadline: FPL only
finalizes/charges transfers at the deadline based on net change from the last confirmed state,
so this should net out to zero transfers used, same as if neither transfer ever happened.

Run from the repo root:
    .\\.venv\\Scripts\\python.exe scripts\\revert_transfer.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fpl_write.client import get_my_team, login_via_cookie

TRANSFERS_URL = "https://fantasy.premierleague.com/api/transfers/"
ENTRY_ID = 8851945
NEXT_EVENT = 2
ELEMENT_OUT = 473  # Aina (DEF, NFO) - sell back
ELEMENT_IN = 329  # Rodon (DEF, LEE) - buy back

session = login_via_cookie()

payload = {
    "confirmed": True,
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
print("BODY:", resp.text[:1500] or "(blank)")

print("\nVerifying...")
my_team = get_my_team(session, ENTRY_ID)
elements = sorted(p["element"] for p in my_team["picks"])
print("num picks:", len(my_team["picks"]))
print("Rodon(329) present:", 329 in elements, "(expected True)")
print("Aina(473) present:", 473 in elements, "(expected False)")
print("transfers:", my_team["transfers"], "(expect made back to 0)")
