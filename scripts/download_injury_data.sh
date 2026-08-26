#!/usr/bin/env bash
# Downloads the Kaggle injury-history dataset src/ingestion/injuries_kaggle.py backfills from:
# "European Football Injuries (2020-2025)" by Sanan Muzaffarov -
# https://www.kaggle.com/datasets/sananmuzaffarov/european-football-injuries-2020-2025
# (CC BY-SA 4.0). 15,603 real injury records across the Big-5 European leagues, 2020/21-2024/25 -
# spot-checked against public record (Van Dijk's 255-day ACL tear, Saka's 99-day hamstring
# injury, Maddison's Leicester->Tottenham transfer timing all matched real reporting).
#
# data/injuries/ is gitignored - re-run this after a fresh clone if the file is missing.
# No Kaggle account/API key needed - this specific dataset's zip is served anonymously.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$REPO_ROOT/data/injuries"
mkdir -p "$DEST_DIR"

echo "=== European Football Injuries (2020-2025), sananmuzaffarov ==="
curl -sfL -o "$DEST_DIR/kaggle_injuries.zip" \
    "https://www.kaggle.com/api/v1/datasets/download/sananmuzaffarov/european-football-injuries-2020-2025"
unzip -o -q "$DEST_DIR/kaggle_injuries.zip" -d "$DEST_DIR"
rm "$DEST_DIR/kaggle_injuries.zip"
echo "  $(wc -l < "$DEST_DIR/full_dataset_thesis - 1.csv") lines -> $DEST_DIR/full_dataset_thesis - 1.csv"
