"""Builds a leak-free training dataset from historical FPL seasons
(data/historical/*/merged_gw.csv, sourced from the vaastav/Fantasy-Premier-League archive).

Every rolling/lag feature uses ONLY prior gameweeks (shift(1) before any rolling window,
computed per (season, element) group) - a row's features never see that row's own outcome
or any future gameweek. Same no-leakage discipline as the rest of this project's backtesting
design (data/data_dictionary.md, sql/analytics.sql).

Double-gameweek fixtures are pre-aggregated to one row per (season, element, GW) by summing
the additive stats across fixtures, matching how the live FPL API's event/{gw}/live/ endpoint
already aggregates DGWs for the current season.

defensive_contribution and its raw-count inputs (clearances_blocks_interceptions, tackles,
recoveries) only exist in FPL data from the 2025-26 season onward - older seasons don't have
the columns at all, not just zeros. Rows from those seasons get dc_data_available=0 and the
dc_roll* features zeroed, rather than fabricating a value that never existed.

Same gap, same fix, for `starts` and the expected_* (xG-family) stats: FPL didn't track them at
all before the 2022-23 season, so 2020-21/2021-22 (added to broaden training data beyond the
original 3-season window) get xg_data_available=0 and their derived roll features zeroed rather
than a fabricated 0.0 that looks like a real "no expected-goal involvement" reading.

Fixture-strength features (was_home, own/opp_attack_form, own/opp_defense_form): the model was
otherwise blind to who a player is about to face - added after the Phase 6.5 error-analysis
notebook flagged "opponent defensive weakness" as an untested direction. Computed from each
team's own leak-free rolling goals-for/against (_compute_team_form, shift(1) before rolling,
same discipline as everything else here), then attached at the per-fixture level before the
DGW aggregation below so a double-gameweek's two fixtures average together like every other
per-fixture stat.

Injury-history features (days_since_last_injury, injuries_last_365d, is_returning_from_injury,
injury_data_available): backfilled from a verified external dataset into the player_injuries
DB table (src/ingestion/injuries_kaggle.py - the historical CSV archive itself has never
carried injury/availability data at all, confirmed by inspection, which is why this needs a
separate table rather than another *_data_available column derived from the archive like
dc_data_available/xg_data_available above). Leak-free the same way as everything else here:
only injury records with injury_from strictly before a row's own fixture kickoff_time are
visible to that row (compute_injury_features, shared with live_features.py so training and
serving use identical recency math).

Fixture-congestion features (days_since_last_match, matches_last_14d, is_congested) and the
muscle-injury-specific redesign (muscle_injuries_last_365d, injury_congestion_risk): the first
version of the injury features (above) were flat/standalone and, honestly, didn't help
(R2 went slightly DOWN, 0.3275->0.321 XGBoost) - the real mechanism per the sports-science
literature (Sports Medicine 2022 systematic review on fixture congestion and injury; the
underlying UEFA studies) is that injury risk rises specifically WHEN a congested run
(<=4 days between matches, the standard threshold in that literature) combines with a player's
own prior injury history - particularly muscle/tendon injuries (hamstring, calf, groin, etc. -
the fatigue-accumulation mechanism), not injuries generally (a concussion or bout of flu has no
plausible congestion-interaction). is_congested and muscle_injuries_last_365d are each
leak-free on their own; injury_congestion_risk = muscle_injuries_last_365d * is_congested is an
explicit product term so linear regression can see the interaction directly (tree models can
already discover it from the two raw ingredients via sequential splits, but the explicit term
keeps it auditable in Model Lab's OLS/permutation-importance diagnostics either way).
EPL-only limitation stated plainly: congestion here only counts EPL matches from our own
archive, not cup/European fixtures we don't have - a real, known undercount for continentally
active clubs, not silently assumed away.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HISTORICAL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "historical"
SEASONS_WITH_DC = {"2025-26"}
ROLLING_WINDOWS = (3, 5)
TEAM_FORM_WINDOW = 5  # matches v_team_form's window (sql/analytics.sql) - same design choice,
# reused here so the live dashboard's Fixture Difficulty Score and this model's opponent
# features are built on the same notion of "recent form", just computed independently in
# Python since training reads from the historical CSV archive, not our own live DB.

SUM_COLS = [
    "minutes", "goals_scored", "assists", "bps", "bonus", "total_points",
    "clean_sheets", "goals_conceded", "own_goals", "saves", "yellow_cards", "red_cards",
    "starts", "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "influence", "creativity", "threat", "ict_index",
]
DC_SUM_COLS = ["clearances_blocks_interceptions", "defensive_contribution", "recoveries", "tackles"]
XG_SUM_COLS = ["starts", "expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded"]
FIRST_COLS = ["name", "position", "value", "selected"]


INJURY_RECENCY_WINDOW_DAYS = 365
INJURY_RETURNING_WINDOW_DAYS = 14
INJURY_NO_HISTORY_SENTINEL = 9999  # "no record of this player ever being injured" - a large,
# clearly-out-of-range value a tree model can split around, same idea as the dc/xg
# data-availability flags above rather than fabricating a plausible-looking 0.

CONGESTION_WINDOW_DAYS = 14
CONGESTED_REST_THRESHOLD_DAYS = 4  # <=4 days between matches is the standard "congested"
# threshold in the sports-science literature (Sports Medicine 2022 systematic review on
# fixture congestion and injury in professional male soccer; some studies use <=5-6 days -
# this project uses the more conservative/common one). EPL-only limitation, stated plainly:
# this only counts EPL matches from our own archive - a team's TRUE match load also includes
# cup and European fixtures we don't have, so this undercounts true rest deficit for clubs in
# continental competition. A real, known gap, not silently assumed away.

# Muscle/tendon injuries are specifically the category the fatigue-accumulation mechanism in
# the literature implicates (repeated sprints/changes of direction -> fatigue -> muscle
# damage), distinct from ligament tears (more often traumatic/contact), fractures, or illness.
# Built from the actual distinct injury_type strings observed in player_injuries, not guessed
# blind - see the real value_counts pulled during development.
MUSCLE_TENDON_KEYWORDS = (
    "hamstring", "muscle", "muscular", "thigh", "calf", "groin", "adductor",
    "strain", "dead leg", "tendon", "achilles", "hip flexor", "pubalgia", "cramp", "contracture",
)


def load_player_injuries() -> pd.DataFrame:
    """All injury records from player_injuries, normalized for recency math. Public (no leading
    underscore) - live_features.py calls this directly so both sides read the exact same table
    the exact same way, not two independent queries that could quietly drift apart."""
    from src.ingestion.db import get_connection

    conn = get_connection()
    try:
        injuries = pd.read_sql_query(
            "SELECT player_code, injury_from, injury_until, injury_type FROM player_injuries", conn
        )
    finally:
        conn.close()
    injuries["injury_from"] = pd.to_datetime(injuries["injury_from"]).astype("datetime64[ns]")
    injuries["injury_until"] = pd.to_datetime(injuries["injury_until"]).astype("datetime64[ns]")
    # A NULL injury_until means "still out / unknown return as of when this was recorded" -
    # treating that as "already back" would be a dangerous assumption for a training feature,
    # so it's treated as an extremely long injury instead (matches INJURY_NO_HISTORY_SENTINEL's
    # spirit: an explicit out-of-range value, not a fabricated plausible one).
    injuries["injury_until"] = injuries["injury_until"].fillna(
        injuries["injury_from"] + pd.Timedelta(days=INJURY_NO_HISTORY_SENTINEL)
    )
    return injuries


_injuries_cache: pd.DataFrame | None = None


def _cached_player_injuries() -> pd.DataFrame:
    """_load_season() is called both from load_all_seasons() (once per season) and from
    _prior_season_summary() (once per season again, for the prior-season-carryover feature) -
    a per-process cache means the DB only gets hit once regardless of how many call sites end
    up needing it, rather than threading the same DataFrame through every layer by hand."""
    global _injuries_cache
    if _injuries_cache is None:
        _injuries_cache = load_player_injuries()
    return _injuries_cache


def compute_injury_features(codes: pd.Series, as_of: pd.Series, injuries: pd.DataFrame) -> pd.DataFrame:
    """For each (code, as_of) pair, leak-free injury-history features using only injury records
    with injury_from strictly before as_of. Shared by historical_features (training) and
    live_features (serving) so the same recency math runs both places.

    Vectorized via merge_asof per player group (C-level, not a per-row Python loop) - an
    earlier row-by-row implementation took ~4 minutes on the full training frame, too slow to
    re-pay on every retrain/Model Lab experiment/backward-elimination round. Only ~2000 player
    groups, each a fast asof join, not 150k+ individual lookups.

    running_max_until = the latest injury_until seen among all injuries up to and including
    this one (cummax, not just this row's own until) - correctly handles a player whose most
    RECENT injury by start date wasn't necessarily their LONGEST-running one. running_count is
    a cumulative count used to derive a windowed count via two asof lookups (as_of and
    as_of-365d) rather than a per-row filter+sum."""
    as_of_dt = pd.to_datetime(as_of.to_numpy(), utc=True).tz_localize(None).astype("datetime64[ns]")
    query = pd.DataFrame({"code": codes.to_numpy(), "as_of": as_of_dt})
    query["_orig_order"] = np.arange(len(query))
    has_history = set(injuries["player_code"].unique())

    parts = []
    for code, group in query.groupby("code", sort=False):
        g = group.sort_values("as_of").copy()
        hist = injuries.loc[injuries["player_code"] == code].sort_values("injury_from")
        if hist.empty:
            g["days_since_last_injury"] = float(INJURY_NO_HISTORY_SENTINEL)
            g["injuries_last_365d"] = 0
            parts.append(g)
            continue

        hist = hist.copy()
        hist["running_max_until"] = hist["injury_until"].cummax()
        hist["running_count"] = np.arange(1, len(hist) + 1)

        merged = pd.merge_asof(
            g, hist[["injury_from", "running_max_until", "running_count"]],
            left_on="as_of", right_on="injury_from", direction="backward", allow_exact_matches=False,
        )
        gap_days = (merged["as_of"] - merged["running_max_until"]) / np.timedelta64(1, "D")
        g["days_since_last_injury"] = gap_days.clip(lower=0).fillna(INJURY_NO_HISTORY_SENTINEL).to_numpy()
        count_as_of = merged["running_count"].fillna(0).to_numpy()

        window_query = pd.DataFrame({"as_of": g["as_of"] - pd.Timedelta(days=INJURY_RECENCY_WINDOW_DAYS)})
        merged_window = pd.merge_asof(
            window_query, hist[["injury_from", "running_count"]],
            left_on="as_of", right_on="injury_from", direction="backward", allow_exact_matches=False,
        )
        count_before_window = merged_window["running_count"].fillna(0).to_numpy()
        g["injuries_last_365d"] = (count_as_of - count_before_window).astype(int)
        parts.append(g)

    out = pd.concat(parts, ignore_index=True).sort_values("_orig_order").drop(columns="_orig_order")
    out = out.reset_index(drop=True)
    out["is_returning_from_injury"] = (out["days_since_last_injury"] <= INJURY_RETURNING_WINDOW_DAYS).astype(int)
    out["injury_data_available"] = out["code"].isin(has_history).astype(int)
    return out


def is_muscle_tendon_injury(injury_type) -> bool:
    if pd.isna(injury_type):
        return False
    text = str(injury_type).lower()
    return any(keyword in text for keyword in MUSCLE_TENDON_KEYWORDS)


def _attach_injury_features(df: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Adds injury-history features to per-player-fixture rows, before DGW aggregation, mirroring
    _attach_fixture_features's placement in the pipeline. Requires is_congested to already be
    present (from _attach_fixture_features, called first in _load_season) - the interaction
    term is the whole point of this redesign: injury proneness mainly matters WHEN combined
    with fixture congestion, not as a flat standalone signal (see the module docstring's
    discussion of the fatigue-accumulation mechanism)."""
    df = df.reset_index(drop=True)
    features = compute_injury_features(df["code"], df["kickoff_time"], injuries)
    for col in ("days_since_last_injury", "injuries_last_365d", "is_returning_from_injury", "injury_data_available"):
        df[col] = features[col]

    # Guard against a real pandas edge case (confirmed on pandas 3.0.5): boolean-mask filtering
    # an ALREADY-zero-row DataFrame drops every column, not just rows - `injuries` is zero-row
    # only on a fresh clone/CI run before the Kaggle injury backfill has ever been loaded, but
    # when it happens this silently strips player_code/injury_type and crashes
    # compute_injury_features below with a KeyError instead of the intended "nobody has any
    # injury history yet" no-op. injuries is already empty in that case, so reusing it directly
    # (same columns, zero rows) sidesteps the buggy filter rather than working around pandas.
    muscle_injuries = injuries if injuries.empty else injuries[injuries["injury_type"].map(is_muscle_tendon_injury)]
    muscle_features = compute_injury_features(df["code"], df["kickoff_time"], muscle_injuries)
    df["muscle_injuries_last_365d"] = muscle_features["injuries_last_365d"]

    df["injury_congestion_risk"] = df["muscle_injuries_last_365d"] * df["is_congested"]
    return df


def _load_team_name_mapping(season_dir: Path) -> dict:
    """merged_gw.csv's own `team` column is the full team NAME ('Man Utd'), but its
    `opponent_team` column is that season's numeric team id (1-20, from teams.csv) - not the
    same space, despite naming that suggests otherwise. teams.csv's `name` column matches
    merged_gw.csv's `team` strings exactly, so this maps name -> id to put both columns in
    the same space before any fixture-level join can work."""
    path = season_dir / "teams.csv"
    if not path.exists():
        return {}
    teams = pd.read_csv(path, usecols=["id", "name"])
    return dict(zip(teams["name"], teams["id"]))


def _load_code_mapping(season_dir: Path) -> dict:
    """element id (season-local, resets every season) -> code (stable across seasons),
    from that season's players_raw.csv. Same season-scoped-id gotcha as the live FPL API
    (data/data_dictionary.md) - vaastav's archive has the identical issue and the identical
    fix (a stable `code` field)."""
    path = season_dir / "players_raw.csv"
    if not path.exists():
        return {}
    raw = pd.read_csv(path, usecols=["id", "code"])
    return dict(zip(raw["id"], raw["code"]))


def _extract_fixture_results(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, fixture): that team's own goals for/against and home/away for the
    match. Deduped from the per-player rows - every player on a team shares an identical
    view of their own team's fixture-level facts, so this just picks out the distinct ones.

    Also pulls each team's fixture-level expected-goals-for/against, from the same per-player
    rows before dedup (unlike goals_for/against, xG isn't a shared fixture-level fact - it's a
    per-player stat that has to be aggregated up to the team). xg_for is the SUM of the team's
    own players' expected_goals for that fixture - standard practice: a team's aggregate xG is
    the sum of its shots' individual xG, and FPL's per-player expected_goals already IS each
    player's own shots' xG that match. xg_against is the MAX (not sum) of the team's players'
    expected_goals_conceded - FPL attributes the full match xG-conceded-while-on-pitch figure
    IDENTICALLY to every player who was on the pitch for it, not divided per player, so summing
    would multiply-count the same conceded chances by however many outfield players + GK shared
    the pitch for them. Whoever played closest to the full 90 (usually the ever-present keeper)
    carries the fullest, most representative reading, which max() picks out without needing to
    know who specifically played every minute."""
    cols = ["fixture", "team", "opponent_team", "was_home", "team_h_score", "team_a_score", "kickoff_time"]
    fx = df[cols].drop_duplicates(subset=["fixture", "team"]).copy()
    fx["kickoff_time"] = pd.to_datetime(fx["kickoff_time"])
    fx["goals_for"] = np.where(fx["was_home"], fx["team_h_score"], fx["team_a_score"])
    fx["goals_against"] = np.where(fx["was_home"], fx["team_a_score"], fx["team_h_score"])

    xg_agg = df.groupby(["fixture", "team"]).agg(
        xg_for=("expected_goals", "sum"), xg_against=("expected_goals_conceded", "max"),
    ).reset_index()
    fx = fx.merge(xg_agg, on=["fixture", "team"], how="left")
    return fx[["fixture", "team", "opponent_team", "was_home", "kickoff_time",
                "goals_for", "goals_against", "xg_for", "xg_against"]]


def _compute_team_form(fixture_results: pd.DataFrame, window: int = TEAM_FORM_WINDOW,
                        has_xg: bool = True) -> pd.DataFrame:
    """Leak-free rolling attack/defense form per team, ordered by kickoff time - shift(1)
    before rolling, same no-leakage discipline as every other feature in this module, so a
    team's form entering a fixture never includes that fixture's own result. A team's first
    fixture(s) of the season (no prior result to roll over) get the league-average goals
    figure instead of 0 - 0 would read as 'guaranteed to face the weakest possible
    attack/defense', a worse assumption than 'unknown, assume average'.

    Also computes the same rolling window over xg_for/xg_against (has_xg=False zeroes it
    instead - pre-2022-23 seasons never tracked the xG-family stats at all, same convention as
    every other xG-derived feature in this module) - chance quality created/conceded, which is
    less noisy than the actual score (finishing variance) for reading a team's true
    attacking/defensive level."""
    fixture_results = fixture_results.sort_values(["team", "kickoff_time"]).copy()
    league_avg_goals = fixture_results["goals_for"].mean()
    grouped = fixture_results.groupby("team", sort=False)
    fixture_results["attack_form"] = (
        grouped["goals_for"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .fillna(league_avg_goals)
    )
    fixture_results["defense_form"] = (
        grouped["goals_against"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .fillna(league_avg_goals)
    )
    if has_xg:
        league_avg_xg_for = fixture_results["xg_for"].mean()
        league_avg_xg_against = fixture_results["xg_against"].mean()
        fixture_results["xg_attack_form"] = (
            grouped["xg_for"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .fillna(league_avg_xg_for)
        )
        fixture_results["xg_defense_form"] = (
            grouped["xg_against"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .fillna(league_avg_xg_against)
        )
    else:
        fixture_results["xg_attack_form"] = 0.0
        fixture_results["xg_defense_form"] = 0.0
    return fixture_results


def _compute_fixture_congestion(fixture_results: pd.DataFrame) -> pd.DataFrame:
    """Leak-free team-level fixture congestion: days since THIS team's previous EPL match, and
    how many EPL matches they played in the trailing CONGESTION_WINDOW_DAYS - both computed
    strictly from matches before this fixture (shift(1), same discipline as _compute_team_form).
    Only ~20 teams x ~35-50 fixtures/season, so a plain per-team loop is fast - no need for the
    merge_asof machinery the much-larger injury lookup needed."""
    fixture_results = fixture_results.sort_values(["team", "kickoff_time"]).reset_index(drop=True)
    days_since = np.full(len(fixture_results), np.nan)
    matches_in_window = np.zeros(len(fixture_results), dtype=int)

    for _, idx in fixture_results.groupby("team").groups.items():
        idx = list(idx)
        kts = fixture_results.loc[idx, "kickoff_time"].to_numpy()
        for pos, row_idx in enumerate(idx):
            if pos == 0:
                continue
            days_since[row_idx] = (kts[pos] - kts[pos - 1]) / np.timedelta64(1, "D")
            window_start = kts[pos] - np.timedelta64(CONGESTION_WINDOW_DAYS, "D")
            matches_in_window[row_idx] = int((kts[:pos] >= window_start).sum())

    fixture_results["days_since_last_match"] = days_since
    fixture_results["matches_last_14d"] = matches_in_window
    return fixture_results


def _attach_fixture_features(df: pd.DataFrame, has_xg: bool) -> pd.DataFrame:
    """Adds was_home + each row's own team's and opponent's rolling attack/defense form
    (leak-free as of entering that specific fixture) to the per-player-fixture rows, before
    they get aggregated to one row per (season, element, GW). Unlike the rolling player-stat
    features elsewhere in this module, these are NOT shifted again at the player level - the
    upcoming fixture and the opponent's rolling form going into it are legitimately known
    before kickoff, so no additional lag is needed on top of _compute_team_form's own shift.

    Also adds each side's rolling expected-goals-for/against form (own_xg_attack_form,
    own_xg_defense_form, opp_xg_attack_form, opp_xg_defense_form) alongside the actual-goals
    form above - see _compute_team_form's docstring for why xG form is worth having
    separately from the goals-based version.

    Also adds the player's OWN team's fixture congestion (days_since_last_match,
    matches_last_14d, is_congested) - congestion is specifically about the player's own side's
    fatigue/rotation risk, not the opponent's."""
    fixture_results = _extract_fixture_results(df)
    form = _compute_team_form(fixture_results, has_xg=has_xg)
    form_cols = ["fixture", "team", "attack_form", "defense_form", "xg_attack_form", "xg_defense_form"]
    own = form[form_cols].rename(columns={
        "attack_form": "own_attack_form", "defense_form": "own_defense_form",
        "xg_attack_form": "own_xg_attack_form", "xg_defense_form": "own_xg_defense_form",
    })
    opp = form[form_cols].rename(columns={
        "team": "opponent_team", "attack_form": "opp_attack_form", "defense_form": "opp_defense_form",
        "xg_attack_form": "opp_xg_attack_form", "xg_defense_form": "opp_xg_defense_form",
    })
    df = df.merge(own, on=["fixture", "team"], how="left")
    df = df.merge(opp, on=["fixture", "opponent_team"], how="left")
    df["was_home"] = df["was_home"].astype(float)

    congestion = _compute_fixture_congestion(fixture_results)
    congestion_cols = congestion[["fixture", "team", "days_since_last_match", "matches_last_14d"]]
    df = df.merge(congestion_cols, on=["fixture", "team"], how="left")
    # No prior match on record (first fixture of our archive for that team) - treat as
    # well-rested rather than fabricating a specific number of days.
    df["days_since_last_match"] = df["days_since_last_match"].fillna(CONGESTION_WINDOW_DAYS)
    df["matches_last_14d"] = df["matches_last_14d"].fillna(0)
    df["is_congested"] = (df["days_since_last_match"] <= CONGESTED_REST_THRESHOLD_DAYS).astype(float)
    return df


INJURY_COLS = [
    "days_since_last_injury", "injuries_last_365d", "is_returning_from_injury", "injury_data_available",
    "muscle_injuries_last_365d", "injury_congestion_risk",
]


def _load_season(season_dir: Path, injuries: pd.DataFrame | None = None,
                  attach_extra_features: bool = True) -> pd.DataFrame:
    """attach_extra_features=False skips the fixture/injury feature attachment entirely (both
    the DB-backed injury lookup and the per-player recency computation, the expensive parts) -
    used by _prior_season_summary(), which only needs total_points/minutes/GW and would
    otherwise redundantly recompute the same fixture/injury features for a season that's also
    being (or will be) loaded in full elsewhere."""
    season = season_dir.name
    df = pd.read_csv(season_dir / "merged_gw.csv")
    df["season"] = season
    df["code"] = df["element"].map(_load_code_mapping(season_dir))

    has_dc = season in SEASONS_WITH_DC
    has_xg = "expected_goals" in df.columns  # FPL didn't track xG-family stats or `starts`
    # at all before 2022-23 - detected from the raw file rather than a hardcoded season list,
    # so adding another pre-2022-23 season later doesn't need a code change here too.
    for col in DC_SUM_COLS:
        if col not in df.columns:
            df[col] = np.nan
    for col in XG_SUM_COLS:
        if col not in df.columns:
            df[col] = np.nan

    agg = {col: "sum" for col in SUM_COLS + DC_SUM_COLS}
    agg.update({col: "first" for col in FIRST_COLS + ["code"]})

    if attach_extra_features:
        df["team"] = df["team"].map(_load_team_name_mapping(season_dir))
        df = _attach_fixture_features(df, has_xg)
        df = _attach_injury_features(df, injuries if injuries is not None else _cached_player_injuries())
        extra_cols = ["was_home", "own_attack_form", "own_defense_form", "opp_attack_form", "opp_defense_form",
                      "own_xg_attack_form", "own_xg_defense_form", "opp_xg_attack_form", "opp_xg_defense_form",
                      "days_since_last_match", "matches_last_14d", "is_congested"] + INJURY_COLS
        agg.update({col: "mean" for col in extra_cols})

    grouped = df.groupby(["season", "element", "GW"], as_index=False).agg(agg)
    grouped["dc_data_available"] = int(has_dc)
    grouped["xg_data_available"] = int(has_xg)
    return grouped


def _prior_season_str(season: str) -> str:
    start_year = int(season.split("-")[0])
    return f"{start_year - 1}-{str(start_year % 100).zfill(2)}"


def _prior_season_summary(season: str) -> pd.DataFrame:
    """Each returning player's FULL prior-season performance (by stable `code`), used as a
    'how did they do last year' prior - without this, a proven veteran and a total unknown
    look identical for however many gameweeks it takes the current season's own rolling
    history to build up. Matched by code so this works whether `season` is a historical
    training season or 2026/27 live data (same function, same join key, on purpose)."""
    empty = pd.DataFrame({
        "code": pd.Series(dtype="int64"),
        "prev_season_points_per90": pd.Series(dtype="float64"),
        "prev_season_minutes_avg": pd.Series(dtype="float64"),
        "prev_season_total_points": pd.Series(dtype="float64"),
    })
    prior_dir = HISTORICAL_DIR / _prior_season_str(season)
    if not prior_dir.exists() or not (prior_dir / "merged_gw.csv").exists():
        return empty

    prior = _load_season(prior_dir, attach_extra_features=False)
    summary = prior.groupby("code").agg(
        total_points=("total_points", "sum"), total_minutes=("minutes", "sum"), gameweeks=("GW", "nunique"),
    ).reset_index()
    summary["prev_season_points_per90"] = np.where(
        summary["total_minutes"] > 0, summary["total_points"] / summary["total_minutes"] * 90, 0.0
    )
    summary["prev_season_minutes_avg"] = summary["total_minutes"] / summary["gameweeks"].clip(lower=1)
    summary = summary.rename(columns={"total_points": "prev_season_total_points"})
    return summary[["code", "prev_season_points_per90", "prev_season_minutes_avg", "prev_season_total_points"]]


def load_all_seasons() -> pd.DataFrame:
    frames = [_load_season(p) for p in sorted(HISTORICAL_DIR.iterdir()) if p.is_dir()]
    df = pd.concat(frames, ignore_index=True)
    df["GW"] = df["GW"].astype(int)
    # "AM" = the experimental Manager position (real club managers were pickable in some
    # historical seasons) - not present in 2026/27's element_types (verified live: only
    # GKP/DEF/MID/FWD), so irrelevant to any squad we'd actually build this season.
    df = df[df["position"] != "AM"]
    return df.sort_values(["season", "element", "GW"]).reset_index(drop=True)


def _prior_rolling(series: pd.Series, window: int, agg: str) -> pd.Series:
    shifted = series.shift(1)
    if agg == "mean":
        return shifted.rolling(window, min_periods=1).mean()
    if agg == "sum":
        return shifted.rolling(window, min_periods=1).sum()
    if agg == "std":
        return shifted.rolling(window, min_periods=2).std()
    if agg == "max":
        return shifted.rolling(window, min_periods=1).max()
    raise ValueError(agg)


def _engineer_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("GW").copy()

    for window in ROLLING_WINDOWS:
        min_sum = _prior_rolling(g["minutes"], window, "sum")

        def per90(col: str) -> pd.Series:
            col_sum = _prior_rolling(g[col], window, "sum")
            return np.where(min_sum > 0, col_sum / min_sum * 90, np.nan)

        g[f"points_per90_roll{window}"] = per90("total_points")
        g[f"goals_per90_roll{window}"] = per90("goals_scored")
        g[f"assists_per90_roll{window}"] = per90("assists")
        g[f"xg_per90_roll{window}"] = per90("expected_goals")
        g[f"xa_per90_roll{window}"] = per90("expected_assists")
        g[f"xgc_per90_roll{window}"] = per90("expected_goals_conceded")
        g[f"minutes_roll{window}"] = _prior_rolling(g["minutes"], window, "mean")
        g[f"bps_roll{window}"] = _prior_rolling(g["bps"], window, "mean")
        g[f"ict_index_roll{window}"] = _prior_rolling(g["ict_index"], window, "mean")
        # threat/creativity are FPL's own attacking sub-indices, already summed into
        # ict_index but never exposed separately before - added specifically to probe the
        # "haul-blindness" finding (see notebooks/09_error_analysis.ipynb): a rising shot/
        # chance-creation trend is a plausible precursor to a big score that a blended index
        # or a pure past-points average wouldn't isolate on its own.
        g[f"threat_roll{window}"] = _prior_rolling(g["threat"], window, "mean")
        g[f"creativity_roll{window}"] = _prior_rolling(g["creativity"], window, "mean")
        g[f"dc_roll{window}"] = _prior_rolling(g["defensive_contribution"], window, "mean")
        g[f"starts_rate_roll{window}"] = _prior_rolling(g["starts"], window, "mean")

    # Volatility (window=5 only): a rolling MEAN can't distinguish "5.5 every week" from
    # "0, 0, 0, 0, 28" - both average the same, but only one has real haul potential. std/max
    # of the recent window surfaces that spread directly, for the quantile/haul models this
    # project already has built to consume a volatility signal a single point-estimate
    # regressor structurally can't use (same lesson as injury_congestion_risk's "volatility,
    # not mean-shift" finding - see historical_features.py's module docstring).
    g["points_std5"] = _prior_rolling(g["total_points"], 5, "std")
    g["goals_std5"] = _prior_rolling(g["goals_scored"], 5, "std")
    g["assists_std5"] = _prior_rolling(g["assists"], 5, "std")
    g["ict_std5"] = _prior_rolling(g["ict_index"], 5, "std")
    g["xg_std5"] = _prior_rolling(g["expected_goals"], 5, "std")
    g["max_points_last5"] = _prior_rolling(g["total_points"], 5, "max")

    # Regression-to-the-mean: actual output vs. the underlying chance quality that produced it,
    # over the same recent window - separates "playing well" (xG matches output) from
    # "running hot/cold" (output diverging from xG, likely to regress).
    g["goals_minus_xg_roll5"] = g["goals_per90_roll5"] - g["xg_per90_roll5"]
    g["assists_minus_xa_roll5"] = g["assists_per90_roll5"] - g["xa_per90_roll5"]

    # Trend/momentum: the recent 3-game window vs. the wider 5-game window, so the model gets
    # trajectory (improving/declining) as an explicit signal, not just two separately-windowed
    # levels it has to infer a difference between on its own.
    g["points_per90_trend_3v5"] = g["points_per90_roll3"] - g["points_per90_roll5"]
    g["xg_per90_trend_3v5"] = g["xg_per90_roll3"] - g["xg_per90_roll5"]
    g["xa_per90_trend_3v5"] = g["xa_per90_roll3"] - g["xa_per90_roll5"]
    g["minutes_trend_3v5"] = g["minutes_roll3"] - g["minutes_roll5"]
    g["ict_index_trend_3v5"] = g["ict_index_roll3"] - g["ict_index_roll5"]

    prior_minutes_sum = g["minutes"].shift(1).expanding().sum()
    prior_points_sum = g["total_points"].shift(1).expanding().sum()
    g["season_points_per90_avg"] = np.where(
        prior_minutes_sum.fillna(0) > 0, prior_points_sum / prior_minutes_sum * 90, np.nan
    )
    g["season_minutes_avg"] = g["minutes"].shift(1).expanding().mean()
    g["gw_number"] = g["GW"]
    return g


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    groups = [
        _engineer_group(g)
        for _, g in df.groupby(["season", "element"], sort=False)
    ]
    result = pd.concat(groups, ignore_index=True)

    dc_cols = [c for c in result.columns if c.startswith("dc_roll")]
    for col in dc_cols:
        result.loc[result["dc_data_available"] == 0, col] = 0.0
        result[col] = result[col].fillna(0.0)

    xg_cols = [c for c in result.columns if c.startswith(("xg_per90_roll", "xa_per90_roll", "xgc_per90_roll", "starts_rate_roll"))]
    xg_cols += [
        "xg_std5", "goals_minus_xg_roll5", "assists_minus_xa_roll5",
        "xg_per90_trend_3v5", "xa_per90_trend_3v5",
    ]
    for col in xg_cols:
        result.loc[result["xg_data_available"] == 0, col] = 0.0
        result[col] = result[col].fillna(0.0)

    # Player x opponent matchup interaction: a high-xG player facing a leaky defense is a
    # better opportunity than the same two facts read separately - this makes that product
    # explicit rather than leaving a tree model to rediscover it via sequential splits (same
    # rationale as injury_congestion_risk's explicit product term). Both operands are already
    # xg_data_available-gated to 0.0 above/via own_xg_attack_form's has_xg gating in
    # _compute_team_form, so this is correctly 0 for pre-2022-23 seasons without any extra
    # gating needed here.
    #
    # opp_xg_defense_form only exists on the historical training path - _attach_fixture_features
    # (called from _load_season, before engineer_features ever sees the frame) is what actually
    # populates it. live_features.py's serving path was never updated to attach it, since
    # matchup_xg_x_opp_xga isn't in FEATURE_COLS for the deployed model (this whole feature was a
    # Phase 7.6 experiment that came back a null result). Default to 0.0 rather than KeyError-ing
    # every live prediction run - matches the "not available yet" convention already used above.
    if "opp_xg_defense_form" in result.columns:
        result["matchup_xg_x_opp_xga"] = result["xg_per90_roll5"] * result["opp_xg_defense_form"]
    else:
        result["matchup_xg_x_opp_xga"] = 0.0

    prior_frames = []
    for season in result["season"].unique():
        summary = _prior_season_summary(season)
        summary = summary.copy()
        summary["season"] = season
        prior_frames.append(summary)
    prior_all = pd.concat(prior_frames, ignore_index=True) if prior_frames else pd.DataFrame(
        columns=["code", "season", "prev_season_points_per90", "prev_season_minutes_avg", "prev_season_total_points"]
    )
    result = result.merge(prior_all, on=["season", "code"], how="left")
    result["had_prior_season"] = result["prev_season_points_per90"].notna().astype(int)
    for col in ["prev_season_points_per90", "prev_season_minutes_avg", "prev_season_total_points"]:
        result[col] = result[col].fillna(0.0).astype(float)

    result["target_points_per90"] = np.where(
        result["minutes"] > 0, result["total_points"] / result["minutes"] * 90, np.nan
    )
    result = result.rename(columns={"value": "price", "total_points": "target_total_points",
                                     "minutes": "target_minutes"})
    return result


FEATURE_COLS = [
    # NOTE: "selected" (ownership) deliberately excluded - vaastav's historical data stores
    # it as a raw manager COUNT, not a percentage, so it's on a different scale per season
    # and would not transfer cleanly to our own live pipeline's selected_by_percent when this
    # model is later applied to 2026/27 data. Price and rolling performance are scale-stable.
    "price", "dc_data_available", "xg_data_available",
    "season_points_per90_avg", "season_minutes_avg",
    "had_prior_season", "prev_season_points_per90", "prev_season_minutes_avg", "prev_season_total_points",
    "was_home", "own_attack_form", "own_defense_form", "opp_attack_form", "opp_defense_form",
    "days_since_last_match", "matches_last_14d", "is_congested",
    "days_since_last_injury", "injuries_last_365d", "is_returning_from_injury", "injury_data_available",
    "muscle_injuries_last_365d", "injury_congestion_risk",
] + [f"{stat}_roll{w}" for w in ROLLING_WINDOWS for stat in (
    "points_per90", "goals_per90", "assists_per90", "xg_per90", "xa_per90",
    "xgc_per90", "minutes", "bps", "ict_index", "dc", "starts_rate", "threat", "creativity",
)]


def build_training_frame() -> pd.DataFrame:
    """Loads all historical seasons, engineers features, and one-hot encodes position.
    Returns one row per (season, element, GW) with FEATURE_COLS + position dummies +
    target_total_points, target_minutes, target_points_per90, plus season/gw_number/name
    for bookkeeping. Rows are NOT filtered here (e.g. GW1 rows with no prior history are
    still present, with NaN/0 features) - callers decide what to drop for training.
    """
    raw = load_all_seasons()
    engineered = engineer_features(raw)
    position_dummies = pd.get_dummies(engineered["position"], prefix="pos")
    return pd.concat([engineered, position_dummies], axis=1)
