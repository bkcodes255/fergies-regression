#!/usr/bin/env bash
# Downloads the historical FPL seasons src/features/historical_features.py trains on, from the
# vaastav/Fantasy-Premier-League archive (https://github.com/vaastav/Fantasy-Premier-League).
# data/historical/ is gitignored - re-run this after a fresh clone, or to add a season.
#
# Usage: scripts/download_historical_data.sh [season ...]
#   No arguments: downloads every season currently used for training/testing (see SEASONS below).
#   With arguments: downloads only the seasons named, e.g.
#     scripts/download_historical_data.sh 2019-20 2020-21
#
# Note on schema drift: FPL's own data has real gaps across seasons, not just missing files -
# see historical_features.py's module docstring. Seasons before 2022-23 lack the expected_*
# (xG-family) stats and `starts` entirely; seasons before 2020-21 also lack `position`/`team`
# in the gameweek file itself. historical_features.py handles the former (xg_data_available
# flag); it does NOT yet handle the latter, so adding a season before 2020-21 needs code changes
# there first, not just a download.
set -euo pipefail

SEASONS=(2020-21 2021-22 2022-23 2023-24 2024-25 2025-26)
if [ "$#" -gt 0 ]; then
    SEASONS=("$@")
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

for season in "${SEASONS[@]}"; do
    dir="$REPO_ROOT/data/historical/$season"
    mkdir -p "$dir"
    echo "=== $season ==="
    curl -sf -o "$dir/merged_gw.csv" "$BASE_URL/$season/gws/merged_gw.csv" \
        && echo "  merged_gw.csv: $(wc -l < "$dir/merged_gw.csv") lines"
    curl -sf -o "$dir/players_raw.csv" "$BASE_URL/$season/players_raw.csv" \
        && echo "  players_raw.csv: $(wc -l < "$dir/players_raw.csv") lines"
done
