# Fergie's Regression

> "An FPL Analytics Report"

A data-driven Fantasy Premier League decision-support system: ingests live FPL data, engineers
player/fixture features, predicts expected player performance, optimizes squad decisions under
FPL's constraints, and evaluates its own decisions against actual outcomes each gameweek.

Built for the 2026/27 season, used to run a real FPL team every gameweek as its test case.

## Status

**Phase 1 — Foundation** (complete)

- [x] Database schema (`sql/schema.sql`)
- [x] Data dictionary (`data/data_dictionary.md`)
- [x] FPL API ingestion client (`src/ingestion/`)
- [x] First full data pull into local Postgres (2026/27 season, GW1)

**Phase 2 — Analytics** (in progress)

- [x] Player season-to-date totals + efficiency ratios (`sql/analytics.sql`, `v_player_season_totals`)
- [x] Player rolling/weighted form (`v_player_rolling_form`, `v_player_weighted_form`)
- [x] Team form + Fixture Difficulty Score v1 (`v_team_form`, `v_fixture_difficulty`)
- [ ] Ownership/transfer trend views
- [ ] Notebook-based exploration of the above against real data

See `data/data_dictionary.md` for exact field-level mapping from the FPL API to this schema,
including two verified gotchas worth reading before touching the data:
1. FPL's `id` fields are season-scoped and get reused next season — this schema keys on the
   season-stable `code` field instead.
2. `defensive_contribution` is a raw action count, not points — the position-dependent
   threshold (10 for DEF, 12 for MID/FWD) isn't in the API and must be applied downstream.

## Build phases

1. **Foundation** — FPL API client, raw snapshots, Postgres schema, manual ingestion
2. **Analytics** — player metrics, form, fixtures, value, ownership, rolling stats
3. **MVP dashboard** — player rankings, fixture planner, squad view, transfer candidates
4. **Prediction** — baseline → linear regression → Random Forest → XGBoost, plus a separate
   minutes model
5. **Decision engine** — transfer optimization, captaincy, bench selection, squad optimization
   under budget/formation constraints
6. **Validation** — historical backtesting with no data leakage, cross-validation, decision
   accuracy scoring
7. **Advanced** — Monte Carlo simulation, rank optimization, injury/news NLP, LLM-generated
   explanations, live match predictions

## Stack

Python, Pandas, PostgreSQL (local for now; Supabase-compatible schema for an easy later move),
scikit-learn / XGBoost, PuLP/OR-Tools for optimization, Streamlit for the first dashboard.
