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

**Phase 2 — Analytics** (complete)

- [x] Player season-to-date totals + efficiency ratios (`sql/analytics.sql`, `v_player_season_totals`)
- [x] Player rolling/weighted form (`v_player_rolling_form`, `v_player_weighted_form`)
- [x] Team form + Fixture Difficulty Score v1 (`v_team_form`, `v_fixture_difficulty`)
- [x] Ownership/transfer trend views (`v_latest_price_snapshot`, `v_price_trend`, `v_ownership_movers`)
- [x] Notebook exploring all of the above against real data (`notebooks/02_player_analysis.ipynb`)

See `data/data_dictionary.md` for exact field-level mapping from the FPL API to this schema,
including three verified gotchas worth reading before touching the data:
1. FPL's `id` fields are season-scoped and get reused next season — this schema keys on the
   season-stable `code` field instead.
2. `defensive_contribution` is a raw action count, not points — the position-dependent
   threshold (10 for DEF, 12 for MID/FWD) isn't in the API and must be applied downstream.
3. `fixtures.finished` lags the actual final whistle by 1+ days (waits for official bonus/stat
   confirmation) — `fixtures.finished_provisional` is what actually flips true at full-time.

To re-run a notebook after a fresh ingestion or retraining: the venv has a registered Jupyter
kernel (`fergies-regression`) — open the notebook in VS Code or Jupyter and select that kernel,
or re-execute headlessly with `nbclient`.

**Phase 4 — Prediction** (complete)

- [x] Historical training data: 3 seasons (2023-24, 2024-25, 2025-26) from the
      vaastav/Fantasy-Premier-League archive, leak-free rolling features
      (`src/features/historical_features.py`)
- [x] Minutes model, points-per-90 model, and a direct total-points model — each compared across
      baseline/linear regression/Random Forest/XGBoost (`src/models/train.py`), tracked in the
      `model_versions` table
- [x] Model comparison notebook with real backtest results (`notebooks/06_model_comparison.ipynb`)

**Key finding, worth reading before using these models**: the plan's original design predicts
`points_per_90` and `minutes` separately and combines them
(`expected_points = points_per_90_pred * minutes_pred / 90`). Backtested on the full held-out
2025-26 season, that decomposition actually *underperforms* a single model predicting total
points directly (R²=0.117 vs R²=0.317) — about 61% of player-gameweeks are unused/fringe
players with exactly 0 minutes, and two separately-noisy nonzero-biased predictions multiply
into a nonzero result more often than a direct model, which can learn "this profile → 0" as one
clean pattern. **The direct Random Forest model on `total_points` is the recommended
predictor.** The minutes model (R²=0.636) is still kept for rotation-risk flagging, just not as
a multiplicative input to points prediction. See `notebooks/06_model_comparison.ipynb` for the
full comparison.

**Phase 3 — MVP dashboard** (complete — built after Phase 4, so it has real predictions to
show, not just descriptive stats)

- [x] My Squad — your real GW picks (`FPL_ENTRY_ID` in `.env`), starting XI vs bench, projected
      points, and a model-vs-your-pick captain comparison
- [x] Player rankings, filterable by position, sorted by predicted next-gameweek points
- [x] Fixture planner (each team's next fixture, ranked by the v1 Fixture Difficulty Score)
- [x] Transfer targets (predicted points per £m, with an ownership-% filter for differentials)

Live inference (`src/features/live_features.py`, `src/models/predict_live.py`) applies the
Phase 4 model to our own current-season data — not the historical training data — to generate
real per-gameweek predictions, stored in a `predictions` table. Squad data comes from
`src/ingestion/load_manager.py` (FPL's `entry/{id}/` and `entry/{id}/event/{gw}/picks/`
endpoints). Run both after each ingestion to refresh:

```
python -m src.ingestion.load_manager
python -m src.models.predict_live
```

Run the dashboard: `streamlit run dashboard/app.py`

**Phase 5 — Decision engine** (partially complete — single-gameweek horizon only, see below)

- [x] Transfer suggestions (`src/recommendations/transfers.py`) — best available same-position
      replacement per squad player, respects budget (price + bank) and the max-3-per-team rule
- [x] Starting XI auto-substitution (`src/recommendations/squad_optimizer.py`) — brute-forces
      all 8 legal FPL formations against your actual 15, exact not heuristic (within a fixed
      formation, top-N predicted points per position is trivially optimal)
- [x] Captain check (built in Phase 3, still here) — flags when the model's top predicted
      scorer in your XI differs from your actual captain

**Known limitations, by design for now**: everything above is single-gameweek — our
`predictions` table only has next-gameweek projections, so there's no multi-week transfer
planning (the plan's "5-GW transfer horizon") yet. Transfer suggestions are independent 1-for-1
comparisons, not a coordinated multi-transfer plan — two rows recommending the same buy target
means "either of these is a good swap," not "buy them twice." No transfer-hit (-4) cost
modeling. A full squad rebuild from the entire player pool under budget (real integer
programming, per the plan's "Transfer Optimization" section) is not built — what exists only
optimizes within your *existing* 15.

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
