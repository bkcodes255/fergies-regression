"""One-off manual test: a REAL captain/vice-captain swap (not a no-op), then swaps it straight
back - a full round trip in one run so nothing is left in a changed state. Safe pre-deadline:
GW1 is finished, GW2 hasn't started, so this only touches the still-editable upcoming team, not
anything already scored. Delete once the write path is folded into the real submit flow.

Run from the repo root:
    .\\.venv\\Scripts\\python.exe scripts\\test_captain_swap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fpl_write.client import get_my_team, login_via_cookie, set_captain

ENTRY_ID = 8851945
ORIGINAL_CAPTAIN = 12  # Saka
ORIGINAL_VICE = 40  # Rogers

session = login_via_cookie()

print(f"Step 1: swap captain -> {ORIGINAL_VICE} (Rogers), vice -> {ORIGINAL_CAPTAIN} (Saka)")
my_team = get_my_team(session, ENTRY_ID)
resp = set_captain(session, ENTRY_ID, my_team, captain_element=ORIGINAL_VICE, vice_captain_element=ORIGINAL_CAPTAIN)
print("  STATUS:", resp.status_code)

print("Step 2: verify the swap took effect")
my_team = get_my_team(session, ENTRY_ID)
captain = next(p["element"] for p in my_team["picks"] if p["is_captain"])
vice = next(p["element"] for p in my_team["picks"] if p["is_vice_captain"])
print(f"  captain={captain}, vice_captain={vice} (expected captain={ORIGINAL_VICE}, vice={ORIGINAL_CAPTAIN})")

print(f"Step 3: swap back -> captain -> {ORIGINAL_CAPTAIN} (Saka), vice -> {ORIGINAL_VICE} (Rogers)")
resp = set_captain(session, ENTRY_ID, my_team, captain_element=ORIGINAL_CAPTAIN, vice_captain_element=ORIGINAL_VICE)
print("  STATUS:", resp.status_code)

print("Step 4: verify reverted back to original")
my_team = get_my_team(session, ENTRY_ID)
captain = next(p["element"] for p in my_team["picks"] if p["is_captain"])
vice = next(p["element"] for p in my_team["picks"] if p["is_vice_captain"])
print(f"  captain={captain}, vice_captain={vice} (expected captain={ORIGINAL_CAPTAIN}, vice={ORIGINAL_VICE})")
