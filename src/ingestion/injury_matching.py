"""Multi-pass player-name matching for external football datasets (the Kaggle injury backfill
now, Transfermarkt/API-Football id resolution later) against our own stable `player_code`.

No existing precedent in this codebase for this: `historical_features._load_code_mapping` only
joins two already-compatible numeric id spaces (both sourced from FPL's own data). An external
provider's free-text player names need real name matching, not an id join.

Three passes, each applied only to names the previous pass left unmatched:
  1. Exact match on accent-stripped, lowercased "first_name second_name" OR `web_name`, against
     the UNION of every historical season's `players_raw.csv` - not just one season. Verified
     empirically: matching a 5-season external dataset against a single season's roster misses
     ~40% of names simply because a name from an early season is invisible in a later
     `players_raw.csv` and vice versa.
  2. Substring: does the external name contain `web_name` as a whole token? Catches cases like
     "Bruno Fernandes" vs. `web_name` "Fernandes", "Cristiano Ronaldo" vs. "Ronaldo" - real gaps
     confirmed left over after pass 1 on the actual Kaggle data. Only accepted when exactly one
     player_code matches (or club narrows it to one) - never resolved ambiguously.
  3. Fuzzy (rapidfuzz `token_sort_ratio`) as a last resort, disambiguated by club when the
     external source provides one, at a conservative similarity threshold.

Whatever's left after all three passes is reported unmatched, not guessed at. A committed (not
gitignored) override file at `data/injury_name_overrides.csv` (columns: raw_name, player_code)
is checked FIRST, before any automated pass, and always wins - the place to permanently fix a
name the matcher can't resolve on its own.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

HISTORICAL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "historical"
OVERRIDES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "injury_name_overrides.csv"
FUZZY_THRESHOLD = 90

# Club abbreviations vaastav's archive uses ("Man Utd") vs. the full names external sources like
# Kaggle use ("Manchester United") - fuzzy string similarity can't bridge these (confirmed:
# "manchester united" vs. "man utd" scores 60/100 on partial_ratio, well under any sane
# threshold, since "Utd" isn't a substring/typo of "United", it's a different word entirely).
# Normalizes both sides to the same canonical (long) form before comparison.
CLUB_ALIASES = {
    "man utd": "manchester united", "man city": "manchester city", "spurs": "tottenham hotspur",
    "wolves": "wolverhampton wanderers", "nott'm forest": "nottingham forest",
    "sheffield utd": "sheffield united", "newcastle": "newcastle united",
    "brighton": "brighton and hove albion", "west brom": "west bromwich albion",
    "west ham": "west ham united", "leeds": "leeds united",
}


def _normalize(s) -> str:
    if pd.isna(s):
        return ""
    stripped = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    return stripped.lower().strip()


def _normalize_club(s) -> str:
    norm = _normalize(s)
    return CLUB_ALIASES.get(norm, norm)


def build_player_directory(historical_dir: Path = HISTORICAL_DIR) -> pd.DataFrame:
    """One row per (player_code, season): normalized full_name/web_name/club. Spans every
    season we have players_raw.csv (+ teams.csv, where present) for - see module docstring for
    why the union of all seasons matters, not just one."""
    rows = []
    for season_dir in sorted(p for p in historical_dir.iterdir() if p.is_dir()):
        players_path = season_dir / "players_raw.csv"
        if not players_path.exists():
            continue
        players = pd.read_csv(players_path, usecols=["code", "first_name", "second_name", "web_name", "team"])
        teams_path = season_dir / "teams.csv"
        if teams_path.exists():
            teams = pd.read_csv(teams_path, usecols=["id", "name"])
            players = players.merge(
                teams.rename(columns={"id": "team", "name": "club"}), on="team", how="left"
            )
        else:
            players["club"] = None
        players["full_name_norm"] = (players["first_name"] + " " + players["second_name"]).map(_normalize)
        players["web_name_norm"] = players["web_name"].map(_normalize)
        players["club_norm"] = players["club"].map(_normalize_club)
        players["season"] = season_dir.name
        rows.append(players[["code", "full_name_norm", "web_name_norm", "club_norm", "season"]])
    directory = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["code", "full_name_norm", "web_name_norm", "club_norm", "season"]
    )
    return directory.rename(columns={"code": "player_code"})


def _load_overrides() -> dict[str, int]:
    if not OVERRIDES_PATH.exists():
        return {}
    overrides = pd.read_csv(OVERRIDES_PATH)
    return dict(zip(overrides["raw_name"].map(_normalize), overrides["player_code"]))


@dataclass
class MatchReport:
    matched: pd.DataFrame  # columns: raw_name, player_code, match_pass
    unmatched: list[str] = field(default_factory=list)


def match_names(raw_names: pd.Series, clubs: pd.Series | None = None,
                 directory: pd.DataFrame | None = None) -> MatchReport:
    """Matches a Series of free-text player names (optionally paired with a same-indexed
    Series of club names) to player_code. Operates on the UNIQUE set of names, not once per
    row - callers with many repeated names (e.g. one row per injury) should map the result back
    via a merge/dict, not call this per-row."""
    directory = directory if directory is not None else build_player_directory()
    unique_names = pd.Series(raw_names.unique())
    club_by_name = dict(zip(raw_names, clubs)) if clubs is not None else {}

    overrides = _load_overrides()
    results: list[dict] = []
    remaining: list[str] = []

    for name in unique_names:
        norm = _normalize(name)
        if norm in overrides:
            results.append({"raw_name": name, "player_code": overrides[norm], "match_pass": "override"})
        else:
            remaining.append(name)

    # Pass 1: exact match on full name or web_name.
    still_remaining = []
    exact_lookup: dict[str, set[int]] = {}
    for _, row in directory.iterrows():
        exact_lookup.setdefault(row["full_name_norm"], set()).add(row["player_code"])
        exact_lookup.setdefault(row["web_name_norm"], set()).add(row["player_code"])
    for name in remaining:
        norm = _normalize(name)
        codes = exact_lookup.get(norm, set())
        if len(codes) == 1:
            results.append({"raw_name": name, "player_code": next(iter(codes)), "match_pass": "exact"})
        else:
            still_remaining.append(name)
    remaining = still_remaining

    # Pass 2: does the external name contain a directory web_name as a whole token?
    web_name_to_codes: dict[str, set[int]] = {}
    for _, row in directory.iterrows():
        web_name_to_codes.setdefault(row["web_name_norm"], set()).add(row["player_code"])

    still_remaining = []
    for name in remaining:
        norm = _normalize(name)
        tokens = set(norm.split())
        candidates: set[int] = set()
        for web_name, codes in web_name_to_codes.items():
            if web_name and web_name in tokens:
                candidates |= codes
        club = _normalize_club(club_by_name.get(name)) if club_by_name.get(name) else None
        if len(candidates) == 1:
            results.append({"raw_name": name, "player_code": next(iter(candidates)), "match_pass": "substring"})
        elif len(candidates) > 1 and club:
            # Club names differ across sources ("Manchester United" vs. our archive's "Man
            # Utd") - fuzzy, not exact, match is needed to disambiguate; confirmed empirically
            # this exact mismatch was silently dropping real matches like Bruno Fernandes.
            candidate_rows = directory[directory["player_code"].isin(candidates)][
                ["player_code", "club_norm"]
            ].drop_duplicates()
            candidate_rows = candidate_rows[candidate_rows["club_norm"] != ""]
            if not candidate_rows.empty:
                scores = candidate_rows["club_norm"].map(lambda c: fuzz.partial_ratio(club, c))
                if scores.max() >= FUZZY_THRESHOLD:
                    narrowed = candidate_rows.loc[scores == scores.max(), "player_code"].unique()
                    if len(narrowed) == 1:
                        results.append({
                            "raw_name": name, "player_code": int(narrowed[0]), "match_pass": "substring+club",
                        })
                        continue
            still_remaining.append(name)
        else:
            still_remaining.append(name)
    remaining = still_remaining

    # Pass 3: fuzzy match against full names, disambiguated by club where available.
    unique_full_names = directory[["player_code", "full_name_norm", "club_norm"]].drop_duplicates()
    still_remaining = []
    for name in remaining:
        norm = _normalize(name)
        club = _normalize_club(club_by_name.get(name)) if club_by_name.get(name) else None
        pool = unique_full_names
        if club:
            club_scores = pool["club_norm"].map(lambda c: fuzz.partial_ratio(club, c) if c else 0)
            club_pool = pool[club_scores >= FUZZY_THRESHOLD]
            if not club_pool.empty:
                pool = club_pool
        scores = pool["full_name_norm"].map(lambda n: fuzz.token_sort_ratio(norm, n))
        if scores.empty or scores.max() < FUZZY_THRESHOLD:
            still_remaining.append(name)
            continue
        best = pool.loc[scores.idxmax()]
        ties = pool[scores == scores.max()]["player_code"].unique()
        if len(ties) == 1:
            results.append({"raw_name": name, "player_code": int(best["player_code"]), "match_pass": "fuzzy"})
        else:
            still_remaining.append(name)

    matched_df = pd.DataFrame(results, columns=["raw_name", "player_code", "match_pass"])
    return MatchReport(matched=matched_df, unmatched=sorted(still_remaining))
