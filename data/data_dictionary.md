# Fergie's Regression — Phase 1 Data Dictionary

Source: official FPL API (`https://fantasy.premierleague.com/api/`), verified live against the
2026/27 season on 2026-08-21. No third-party data source (Understat, FBref) is required — the
official API already exposes xG/xA and defensive-contribution stats directly.

## Endpoints used

| Endpoint | Purpose | Pull cadence |
|---|---|---|
| `GET /bootstrap-static/` | Full player/team/gameweek reference data + season totals | Daily (06:00) |
| `GET /fixtures/` | All fixtures, updated difficulty/results | Daily |
| `GET /event/{event_id}/live/` | Per-player stats + point breakdown for one gameweek | Every ~15 min on matchday, once post-match otherwise |
| `GET /element-summary/{player_id}/` | Per-player full gameweek history + upcoming fixtures | Weekly, or on demand |

**Important:** every numeric `id` in these payloads (`elements[].id`, `teams[].id`,
`events[].id`, `fixtures[].id`) is **scoped to the current season** and is reassigned each
season. The schema keys on the season-stable `code` field for players/teams instead — see
`schema.sql` comments. Never join across seasons on `id`.

---

## `teams` (dimension, keyed by `team_code`)

| Column | FPL source field | Notes |
|---|---|---|
| `team_code` | `teams[].code` | Stable across seasons (Opta-linked) |
| `name` | `teams[].name` | |
| `short_name` | `teams[].short_name` | e.g. "ARS" |

## `team_season_stats` (keyed by `team_code` + `season`)

| Column | FPL source field | Notes |
|---|---|---|
| `fpl_id` | `teams[].id` | This season's numeric id (1-20); API-lookup convenience only |
| `strength*` | `teams[].strength*` | FPL's own overall/attack/defence home/away ratings — recalculated each season, treat as one input signal, not gospel (see Fixture Analysis plan — we build our own FDS on top) |

## `players` (dimension, keyed by `player_code`)

| Column | FPL source field | Notes |
|---|---|---|
| `player_code` | `elements[].code` | Stable across seasons |
| `first_name` / `second_name` / `web_name` | `elements[].first_name` / `.second_name` / `.web_name` | `web_name` is what's shown on the game's UI |
| `birth_date` | `elements[].birth_date` | |

## `player_seasons` (keyed by `player_code` + `season`)

| Column | FPL source field | Notes |
|---|---|---|
| `fpl_id` | `elements[].id` | This season's numeric id |
| `team_code` | `elements[].team` (mapped via team `id`→`code`) | **Starting** team for the season only — mid-season transfers are captured via the denormalized `team_code` on `player_gameweek_stats`, not here |
| `element_type` | `elements[].element_type` | 1=GKP 2=DEF 3=MID 4=FWD |

## `gameweeks` (keyed by `season` + `event_id`)

| Column | FPL source field |
|---|---|
| `event_id` | `events[].id` |
| `name` | `events[].name` |
| `deadline_time` | `events[].deadline_time` |
| `finished` / `data_checked` / `is_current` | `events[].finished` / `.data_checked` / `.is_current` |
| `average_entry_score`, `highest_score` | `events[].average_entry_score`, `.highest_score` |
| `most_selected`, `most_captained`, `most_vice_captained`, `top_element` | same-named fields, mapped `id`→`code` |
| `chip_plays` | `events[].chip_plays` (raw JSON, e.g. `[{"chip_name": "bboost", "num_played": 123456}]`) |

## `fixtures` (keyed by `season` + `fpl_fixture_id`)

| Column | FPL source field |
|---|---|
| `fpl_fixture_id` | `fixtures[].id` |
| `event_id` | `fixtures[].event` |
| `team_h_code` / `team_a_code` | `fixtures[].team_h` / `.team_a` (mapped `id`→`code`) |
| `team_h_score` / `team_a_score` | `fixtures[].team_h_score` / `.team_a_score` |
| `kickoff_time` | `fixtures[].kickoff_time` |
| `team_h_difficulty` / `team_a_difficulty` | `fixtures[].team_h_difficulty` / `.team_a_difficulty` — this is FPL's own FDR (1-5); the plan's custom Fixture Difficulty Score is computed downstream, this column is just the raw input |
| `finished_provisional` | `fixtures[].finished_provisional` | **Read this carefully — two different "finished" signals exist and they are not interchangeable.** `finished` only flips to `true` once bonus points and stats are officially locked in, which can lag the actual final whistle by a day or more. `finished_provisional` flips to `true` at full-time, with the score already final (confirmed empirically: Arsenal's fixture 1 showed `team_h_score=3`, `finished_provisional=true`, `finished=false` several hours after full time). **Use `finished_provisional` for "has this team played yet" logic** (team form, fixture-difficulty views all use this) — using `finished` alone silently excludes every match that hasn't cleared official confirmation yet, which in practice is most of them at any given moment. Reserve `finished` for knowing when a gameweek's points are truly final and safe to treat as immutable for backtesting. |

## `player_gameweek_stats` (core fact table, keyed by `season` + `event_id` + `player_code`)

Sourced from `event/{event_id}/live/` → `elements[].stats` (one row per player per gameweek,
aggregated across fixtures for double-gameweeks). Most columns map 1:1 by name:
`minutes`, `starts`, `goals_scored`, `assists`, `clean_sheets`, `goals_conceded`,
`own_goals`, `penalties_saved`, `penalties_missed`, `yellow_cards`, `red_cards`, `saves`,
`bonus`, `bps`, `influence`, `creativity`, `threat`, `ict_index`,
`clearances_blocks_interceptions`, `tackles`, `recoveries`,
`expected_goals`, `expected_assists`, `expected_goal_involvements`, `expected_goals_conceded`,
`total_points`, `in_dreamteam`.

**`defensive_contribution` — read this carefully.** This column is the **raw combined action
count**, not points:
- Defenders: `clearances_blocks_interceptions + tackles` (recoveries excluded)
- Midfielders/Forwards: `clearances_blocks_interceptions + tackles + recoveries`

It converts to **2 flat points, capped once per match**, only once the count crosses a
position-dependent threshold: **10 for defenders (CBIT), 12 for midfielders/forwards
(CBIRT)**. GKPs are not eligible. The threshold constants are *not* exposed anywhere in the
API (confirmed by inspecting `bootstrap-static.game_config.scoring`, which only gives the flat
2-point value) — they're sourced from FPL's published rules and verified empirically against
live GW1 data (no player below the threshold received the bonus; the formula composition
matched exactly). If FPL ever changes these thresholds, the source of truth is
`game_config.scoring.defensive_contribution` for the point *value*, but the threshold itself
must be tracked manually in the feature-engineering layer — put it in one named constant
(`DC_THRESHOLD_DEF = 10`, `DC_THRESHOLD_MID_FWD = 12`), not scattered inline.

`explain` — stored verbatim as JSONB from `elements[].explain`: a list of
`{fixture, stats: [{identifier, points, value, points_modification}]}` objects. This is the
ground-truth, human-readable "why did this player score N points" breakdown, including
**live provisional bonus** (confirmed present mid-match, before full-time). It's the direct
input for the Phase 7 LLM-explanation feature — no need to reconstruct the scoring logic
yourself later.

## `player_price_snapshots` (keyed by `season` + `player_code` + `snapshot_date`)

Sourced from `bootstrap-static` → `elements[]`, pulled once daily (prices/ownership/status can
change any day, independent of gameweeks).

| Column | FPL source field | Notes |
|---|---|---|
| `now_cost` | `elements[].now_cost` | Tenths of £m — divide by 10 for display (65 → £6.5m) |
| `cost_change_event` / `cost_change_start` | same-named fields | |
| `selected_by_percent` | `elements[].selected_by_percent` | |
| `transfers_in_event` / `transfers_out_event` | same-named fields | Reset each gameweek by FPL |
| `status` | `elements[].status` | `a`=available `d`=doubtful `i`=injured `s`=suspended `u`=unavailable |
| `chance_of_playing_next_round` | `elements[].chance_of_playing_next_round` | Percent, nullable |
| `news` | `elements[].news` | Free-text injury/rotation note from FPL editors |

## `raw_snapshots`

Full unmodified JSON response body for every API pull, tagged with endpoint + season +
pull timestamp. This is the audit trail that makes "what did the model know at GW10" honest
during backtesting, and the fallback if a parsed column is ever found to be wrong.
