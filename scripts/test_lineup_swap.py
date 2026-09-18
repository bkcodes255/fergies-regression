"""One-off manual test: a REAL bench-order swap via set_lineup() (src/fpl_write/client.py),
then swaps it straight back - same safe round-trip pattern as test_captain_swap.py. This is the
one new write shape approval_bot.py's 'squad' kind depends on that has never been tested live
(only the narrower captain-only reassignment has). Unlike the transfers endpoint, this POSTs to
the same /api/my-team/{id}/ endpoint the captain swap already proved safe to write to repeatedly
- no resource gets consumed, so a test-then-revert round trip is fine here.

Reads your actual current squad live (no hardcoded element ids - your bench changes week to
week) and only reorders the two lowest-priority bench slots. Does NOT touch which 11 start or
who's captain/vice-captain.

Run from the repo root (needs a live FPL_API_AUTHORIZATION/FPL_SESSION_COOKIE in .env):
    .\\.venv\\Scripts\\python.exe scripts\\test_lineup_swap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from src.fpl_write.client import get_my_team, login_via_cookie, set_lineup

ENTRY_ID = settings.ENTRY_ID

session = login_via_cookie()

print("Reading current squad...")
my_team = get_my_team(session, ENTRY_ID)
picks_by_position = {p["position"]: p for p in my_team["picks"]}
starting_elements = [picks_by_position[i]["element"] for i in range(1, 12)]
bench_elements = [picks_by_position[i]["element"] for i in range(12, 16)]
print(f"  Starting XI (positions 1-11): {starting_elements}")
print(f"  Bench (positions 12-15): {bench_elements}")

if len(bench_elements) < 2:
    print("Fewer than 2 bench players found - can't safely test a swap. Aborting.")
    sys.exit(1)

swapped_bench = bench_elements.copy()
swapped_bench[0], swapped_bench[1] = swapped_bench[1], swapped_bench[0]

print(f"\nStep 1: swap bench order {bench_elements[:2]} -> {swapped_bench[:2]}")
resp = set_lineup(session, ENTRY_ID, my_team, starting_elements, swapped_bench)
print("  STATUS:", resp.status_code, "BODY:", (resp.text or "(blank)")[:300])

print("\nStep 2: verify the swap took effect")
my_team = get_my_team(session, ENTRY_ID)
new_bench = [p["element"] for p in sorted(
    (p for p in my_team["picks"] if p["position"] >= 12), key=lambda p: p["position"]
)]
print(f"  bench now: {new_bench} (expected {swapped_bench})")

print(f"\nStep 3: swap back -> {bench_elements[:2]}")
resp = set_lineup(session, ENTRY_ID, my_team, starting_elements, bench_elements)
print("  STATUS:", resp.status_code, "BODY:", (resp.text or "(blank)")[:300])

print("\nStep 4: verify reverted back to original")
my_team = get_my_team(session, ENTRY_ID)
final_bench = [p["element"] for p in sorted(
    (p for p in my_team["picks"] if p["position"] >= 12), key=lambda p: p["position"]
)]
print(f"  bench now: {final_bench} (expected {bench_elements})")
