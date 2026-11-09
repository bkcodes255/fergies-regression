"""Writes the web site's data: python -m src.web_data.export --out-dir web/src/data

Run by .github/workflows/web.yml after each data refresh, just before the site is built.
Needs DATABASE_URL, FPL_SEASON and FPL_ENTRY_ID, like the rest of the pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from config import settings
from src.ingestion.db import get_engine
from src.web_data import fixtures_page, optimal_squad_page, players_page, season_page, site_data
from src.web_data.this_week import this_week_payload

# Public pages get (conn, season) and so cannot read a manager's squad. Only PRIVATE exporters get
# the entry id, and their pages are served only behind the login (web/functions/_middleware.js).
# Model Lab isn't here: scripts/build_model_lab.py writes it offline and its file is committed.
PUBLIC = {
    "site": site_data.payload,
    "players": players_page.payload,
    "fixtures": fixtures_page.payload,
    "optimal-squad": optimal_squad_page.payload,
}
PRIVATE = {
    "this-week": this_week_payload,
    "season": season_page.payload,
}


def _plain(o):
    """numpy scalars that slipped through, as plain JSON numbers/bools."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"Not JSON serialisable: {type(o).__name__}")


def write(out_dir: Path, name: str, payload: dict) -> None:
    # allow_nan=False: a NaN would make the file invalid JSON for the browser; fail here instead.
    (out_dir / f"{name}.json").write_text(json.dumps(payload, default=_plain, allow_nan=False, indent=1))
    print(f"Wrote {name}.json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with get_engine().connect() as conn:
        for name, build in PUBLIC.items():
            write(args.out_dir, name, build(conn, settings.SEASON))
        for name, build in PRIVATE.items():
            write(args.out_dir, name, build(conn, settings.SEASON, settings.ENTRY_ID))
    return 0


if __name__ == "__main__":
    sys.exit(main())
