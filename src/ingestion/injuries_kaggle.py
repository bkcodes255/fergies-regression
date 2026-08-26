"""Backfills `player_injuries` from the Kaggle "European Football Injuries (2020-2025)" dataset
(sananmuzaffarov, CC BY-SA 4.0 - see scripts/download_injury_data.sh to fetch it). 15,603 real
injury records across the Big-5 European leagues, spot-checked against public record (Van
Dijk's 255-day ACL tear, Saka's 99-day hamstring injury, Maddison's Leicester->Tottenham
transfer timing all matched real reporting) before being trusted for this.

Matches each row's free-text player_name (+ club, for disambiguation) to our stable
player_code via injury_matching.match_names(), then upserts into player_injuries
(source='kaggle_thesis_2020_25'). Only rows that actually matched a player_code are inserted -
unmatched names are written to a report CSV for review, not silently dropped without a trace or
guessed at.

Run directly:
    python -m src.ingestion.injuries_kaggle
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import psycopg2.extras

from src.ingestion.db import get_connection
from src.ingestion.injury_matching import build_player_directory, match_names

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "injuries" / "full_dataset_thesis - 1.csv"
UNMATCHED_REPORT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "injury_unmatched_report.csv"
SOURCE = "kaggle_thesis_2020_25"


def _parse_days(value: str) -> int | None:
    """'43 days' -> 43. Some rows may have unexpected formats; return None rather than crash."""
    match = re.match(r"(\d+)", str(value))
    return int(match.group(1)) if match else None


def load_and_match() -> tuple[pd.DataFrame, list[str]]:
    """Returns (matched_rows_ready_to_insert, unmatched_names). matched_rows_ready_to_insert has
    one row per Kaggle injury record with player_code attached, dropping rows whose name never
    matched."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    directory = build_player_directory()
    report = match_names(df["player_name"], clubs=df["club"], directory=directory)

    df = df.merge(report.matched[["raw_name", "player_code"]], left_on="player_name", right_on="raw_name", how="left")
    matched = df.dropna(subset=["player_code"]).copy()
    matched["player_code"] = matched["player_code"].astype(int)
    matched["injury_from"] = pd.to_datetime(matched["injury_from_parsed"]).dt.date
    matched["injury_until"] = pd.to_datetime(matched["injury_until_parsed"]).dt.date
    matched["days_out"] = matched["Days"].map(_parse_days)
    return matched, report.unmatched


def upsert_injuries(cur, matched: pd.DataFrame) -> int:
    records = matched[["player_code", "league", "Injury", "injury_from", "injury_until",
                        "days_out", "Games missed", "player_name"]].to_dict("records")
    rows = [
        (
            int(r["player_code"]), SOURCE, r["league"], r["Injury"], r["injury_from"], r["injury_until"],
            r["days_out"], int(r["Games missed"]) if not pd.isna(r["Games missed"]) else None, r["player_name"],
        )
        for r in records
    ]
    if not rows:
        return 0
    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO player_injuries (
            player_code, source, league, injury_type, injury_from, injury_until,
            days_out, games_missed, raw_player_name
        ) VALUES %s
        ON CONFLICT (player_code, source, injury_from, injury_type) DO NOTHING
        """,
        rows,
    )
    return len(rows)


def run() -> None:
    print(f"Loading {DATA_PATH}...")
    matched, unmatched = load_and_match()
    total_unique_names = matched["player_name"].nunique() + len(unmatched)
    print(f"Matched {matched['player_name'].nunique()}/{total_unique_names} unique player names "
          f"- {len(matched)} injury rows will be inserted (unmatched rows are dropped, not guessed at).")

    if unmatched:
        pd.Series(sorted(unmatched), name="raw_name").to_csv(UNMATCHED_REPORT_PATH, index=False)
        print(f"{len(unmatched)} unmatched names written to {UNMATCHED_REPORT_PATH} - add real "
              f"ones to data/injury_name_overrides.csv (columns: raw_name,player_code) and rerun.")

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                inserted = upsert_injuries(cur, matched)
        print(f"Upserted {inserted} injury records (source={SOURCE!r}).")
    finally:
        conn.close()


if __name__ == "__main__":
    run()
