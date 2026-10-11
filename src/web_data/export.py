"""Writes the web site's data: python -m src.web_data.export --out web/src/data/this-week.json

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
from src.web_data.this_week import this_week_payload


def _plain(o):
    """numpy scalars that slipped through, as plain JSON numbers/bools."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"Not JSON serialisable: {type(o).__name__}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    with get_engine().connect() as conn:
        payload = this_week_payload(conn, settings.SEASON, settings.ENTRY_ID)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: a NaN would make the file invalid JSON for the browser; fail here instead.
    args.out.write_text(json.dumps(payload, default=_plain, allow_nan=False, indent=1))
    squad = payload["squad"]
    print(f"Wrote {args.out}: GW{payload['deadline']['event_id'] if payload['deadline'] else '-'}, "
          f"{'squad of ' + str(len(squad['players'])) if squad else 'no squad'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
