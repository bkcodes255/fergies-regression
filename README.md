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

- [x] Historical training data: 5 seasons (2020-21 through 2024-25) from the
      vaastav/Fantasy-Premier-League archive, test season 2025-26 held out entirely — leak-free
      rolling features (`src/features/historical_features.py`). Re-download with
      `scripts/download_historical_data.sh` (gitignored, not committed). FPL's own data has real
      schema gaps across seasons, not just missing files: 2020-21/2021-22 predate `starts` and
      the expected_* (xG-family) stats entirely, so those rows get `xg_data_available=0` and
      their derived rolling features zeroed rather than a fabricated value (same treatment
      `dc_data_available` already gets for `defensive_contribution`, which only exists from
      2025-26). Seasons before 2020-21 go a step further and drop `position`/`team` from the
      gameweek file itself — not yet supported, would need a per-season `players_raw.csv`
      lookup first.
- [x] Minutes model, points-per-90 model, and a direct total-points model — each compared across
      baseline/linear regression/Random Forest/XGBoost (`src/models/train.py`), tracked in the
      `model_versions` table
- [x] Model comparison notebook with real backtest results (`notebooks/06_model_comparison.ipynb`)

**Key finding, worth reading before using these models**: the plan's original design predicts
`points_per_90` and `minutes` separately and combines them
(`expected_points = points_per_90_pred * minutes_pred / 90`). Backtested on the full held-out
2025-26 season, that decomposition actually *underperforms* a single model predicting total
points directly (R²=0.141 vs R²=0.320) — about 61% of player-gameweeks are unused/fringe
players with exactly 0 minutes, and two separately-noisy nonzero-biased predictions multiply
into a nonzero result more often than a direct model, which can learn "this profile → 0" as one
clean pattern. **The direct Random Forest model on `total_points` is the recommended
predictor.** The minutes model (R²=0.632) is still kept for rotation-risk flagging, just not as
a multiplicative input to points prediction. See `notebooks/06_model_comparison.ipynb` for the
full comparison.

**Hyperparameters** (`RF_PARAMS` in `train.py`): a 5-fold walk-forward grid search (see Phase 6
below) found a shallower, more-regularized Random Forest (`max_depth=6, min_samples_leaf=10`,
down from the original `8`/`5`) wins on `total_points_direct` in every fold with lower
cross-fold variance, and trains faster. An unconstrained config (`max_depth=None,
min_samples_leaf=2`) tested clearly worse, confirming the original model was mildly overfit, not
underfit. Checked against the other two targets too — negligible effect either way (minutes
R² -0.003, points_per_90 R² +0.002) — so this is a net win with no real cost.

Adding the two older seasons (2020-21/2021-22) to training data alone, before the hyperparameter
change, left the deployed configuration's held-out R² completely unchanged (0.320 both ways) —
those seasons are missing the whole xG-family feature block, so the extra rows are lower-signal
per row than the newer seasons already in the training set. More historical data wasn't free
lunch here; the hyperparameter tuning is what actually moved the needle.

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

**Phase 5 — Decision engine** (complete)

- [x] Coordinated transfer plan (`src/recommendations/transfers.py`) — greedily builds a
      sequence of transfers (not independent 1-for-1 suggestions), so a 2nd transfer never
      recommends a target already used by the 1st. Applies `TRANSFER_HIT_COST` (-4) once free
      transfers run out, and stops as soon as the next transfer wouldn't survive its hit cost
- [x] Free-transfers-remaining tracking (`compute_free_transfers`) — computed entirely from
      `manager_gameweeks.event_transfers` history already ingested (no new endpoint needed),
      following the 2026/27 rule change to a 5-transfer bank (was 2)
- [x] Starting XI auto-substitution (`src/recommendations/squad_optimizer.py`) — brute-forces
      all 8 legal FPL formations against your actual 15, exact not heuristic (within a fixed
      formation, top-N predicted points per position is trivially optimal)
- [x] Captain check (built in Phase 3, still here) — flags when the model's top predicted
      scorer in your XI differs from your actual captain
- [x] Full squad-rebuild optimizer (`src/recommendations/squad_builder.py`) — a real integer
      program (PuLP/CBC), not a heuristic: picks the 15-player squad (exactly 2 GKP/5 DEF/5
      MID/3 FWD, max 3 per team, within budget) that maximizes what its *starting XI* can
      actually score. Squad and starting-XI selection are solved together in one MILP —
      captain doubling is in the objective too — rather than picking 15 players and hoping a
      good XI happens to fit inside them. Solves in under a second against the full player pool.

- [x] Multi-week transfer horizon (`src/recommendations/horizon.py`) — projects each player's
      existing next-GW prediction across the next `DEFAULT_HORIZON` (5) gameweeks, scaled
      week-by-week by their team's fixture difficulty (`v_fixture_difficulty`, reused from
      Phase 2). Deliberately not recursive re-prediction: the model only outputs total_points
      and minutes directly, so simulating the ~10 other rolling-feature inputs forward would
      mean crude approximation compounding over the window. One row per (player, fixture) means
      a blank gameweek naturally contributes 0 and a double gameweek is counted twice, with no
      special-casing needed. The transfer plan, squad-rebuild optimizer, and starting-XI
      auto-substitution all take a `value_col` parameter now — pass `"horizon_points"` instead
      of the default `"predicted_points"` to rank/optimize on the 5-GW window instead of just
      next gameweek. Wired into the dashboard as toggles on the transfer plan and Optimal Squad
      tabs, and as an extra column + sort option on Transfer Targets.

**Known limitations, still true**: the transfer plan is greedy (picks the best single transfer
at each step), not a global search over combinations — an early pick can block a better later
combination, so it's not guaranteed globally optimal, just internally consistent (unlike the
old independent-suggestions approach, which could recommend the same buy target multiple
times). The squad-rebuild optimizer is genuinely optimal for its stated objective, but builds
from scratch — it doesn't account for the cost/hits of actually transferring from your current
squad into it; that's still the transfer plan's job. The fixture-difficulty multiplier
(`DIFFICULTY_SENSITIVITY` in `horizon.py`) is a hand-picked coefficient, not fit to data — it
hasn't been backtested against how much fixture difficulty actually should move a points
projection.

**Phase 6 — Validation** (complete)

- [x] Decision-engine backtest (`src/validation/backtest.py`, `notebooks/07_decision_backtest.ipynb`)
      — simulates the full held-out 2025-26 season week by week using only leak-free,
      at-the-time predictions (never a gameweek's own result), and scores three policies
      against real historical outcomes:
      1. **Full decision engine** — MILP squad pick, then every week the greedy transfer plan +
         auto-substitution + captain choice, all prediction-driven.
      2. **Static squad** — the same starting squad, never transferred again, but
         auto-sub/captain still prediction-driven. Isolates the transfer plan's marginal value.
      3. **Static squad, oracle picks** — the same frozen squad, but auto-sub/captain chosen
         with actual (hindsight) results. A same-squad ceiling that isolates weekly-judgment
         quality from squad-composition quality.

      2025-26 result: full engine **1791 pts** (48.4/GW) vs. static squad **1445 pts**
      (39.1/GW) vs. static-squad-with-oracle-picks **1637 pts** (44.2/GW). The transfer plan
      contributed **+346 pts** over the season (engine vs. static) — more than even perfect
      weekly lineup/captain choices from a frozen squad would have (+192 pts, static oracle vs.
      static) — meaning the *decisions*, not just the predictions, hold up against a real
      season, and the transfer plan specifically is the dominant value driver. (Re-run after the
      historical-data expansion below; absolute totals moved because the GW2 MILP squad pick
      changed with the retrained model, but the relative story — transfer plan as the dominant
      value driver — held.)

      This validates the decision engine itself, on top of Phase 4's existing prediction-model
      backtest (R²=0.320 on the same held-out season, `notebooks/06_model_comparison.ipynb`).

- [x] Model cross-validation (`src/models/cross_validate.py`, `notebooks/08_cross_validation.ipynb`)
      — notebook 06 evaluates model selection with a single train/holdout split: one R²/RMSE
      reading per model type. This repeats that same expanding-window backtest at every season
      boundary the historical archive has (5 folds, e.g. fold 1: train=2020-21, test=2021-22,
      ... fold 5: train=2020-21 through 2024-25, test=2025-26 — notebook 06's own split), to
      check whether model selection is stable or a lucky split. It is: Random Forest wins
      `total_points_direct` in every fold (mean R²=0.297 ± 0.020 — the lowest cross-fold std of
      any model type), and `points_per_90` stays weak/unstable in every fold (R² near zero,
      sometimes negative), reinforcing Phase 4's "direct beats decomposed" finding across five
      independent test seasons, not just one. Purely an evaluation exercise — doesn't write to
      `model_versions` or save joblib artifacts, so it can't accidentally outrank the real
      deployed model in `predict_live.get_best_model`'s selection with a fold trained on less
      data that got lucky on an easier test season. This notebook's own results directly fed the
      hyperparameter tuning below.

- [x] Feature importance + error analysis (`src/models/error_analysis.py`,
      `notebooks/09_error_analysis.ipynb`) — a diagnostic pass on the deployed model, not just
      its headline R². Feature importance: `minutes_roll3` alone is ~66% of total importance,
      plus `ict_index_roll3`/`ict_index_roll5` (~23%) — three features explain ~89% of the
      model's decisions, meaning it's mostly answering "will this player play" rather than "how
      well will they play." **The real finding**: bucketing the held-out season by actual
      outcome shows the model's average prediction barely moves past ~3 points regardless of
      how big the real result was — for an actual 11+-point haul, mean predicted is 3.00 (less
      than a quarter of what happened). This is textbook regression-to-the-mean for an
      MSE-trained model on a right-skewed target: minimizing squared error rewards hedging
      toward a safe low prediction over confidently guessing big and sometimes being wrong. It's
      correct under the training objective, but it's specifically the failure mode that matters
      most for captaincy (doubling a haul is how a gameweek is won) and differential transfers —
      areas the decision-engine backtest above already flagged the transfer plan, not weekly
      captain judgment, as the dominant value driver. This finding held essentially unchanged
      through the historical-data expansion and hyperparameter retune below (haul mean predicted
      moved from 3.08 to 3.00) — it's a structural property of predicting a point estimate for a
      right-skewed target, not something more data or tuning fixes on its own. Two directions
      worth considering before Phase 7, not yet built: predicting a distribution/haul-probability
      instead of a point estimate, or a haul-specific feature set (shot volume trend, set-piece
      role, opponent defensive weakness) instead of general-form features.

- [x] Historical data expansion + hyperparameter retune (2026-08-22) — added 2020-21/2021-22 to
      training data (`scripts/download_historical_data.sh`, 5 training seasons total now; see
      Phase 4 above for the schema-gap handling those two seasons needed) and re-ran the Phase 6
      harnesses above to guide an actual parameter search rather than tuning blind. Adding the
      two older seasons alone, before any hyperparameter change, left the deployed
      configuration's held-out R² completely unchanged (0.320 both ways) — those seasons are
      missing the whole xG-family feature block, so more rows didn't mean more signal. A 5-fold
      grid search on `total_points_direct` *did* find a real improvement: a shallower,
      more-regularized Random Forest (`max_depth=6, min_samples_leaf=10`, down from `8`/`5`)
      wins every fold with lower cross-fold variance and trains faster; an unconstrained config
      tested clearly worse, confirming the original model was mildly overfit, not underfit.
      Adopted as the new default and retrained/redeployed (`model_id=43`); checked against the
      other two targets first — negligible effect either way. All of notebooks 06-09 were
      re-executed against the retrained model so their baked-in output reflects it.

      Phase 6 is now complete: decision-engine backtesting, model cross-validation, a
      feature-importance/error-analysis pass, and a data/hyperparameter iteration guided by
      that harness, on top of Phase 4's original no-leakage holdout backtest.

      **Known simplifications**, documented in `src/validation/backtest.py`'s module
      docstring: no historical injury/availability data survives in the archive (every player
      treated as always selectable); no reactive real-FPL auto-substitution for a starter who
      blanks; the multi-week fixture-difficulty horizon isn't used (it needs live-ingested
      fixture data only 2026/27 has) — this backtest is single-gameweek-horizon, like the live
      dashboard's default.

**Phase 6.5 — Haul-blindness fix** (2026-08-22, complete)

Brian picked this over jumping to Phase 7. Tested the "haul-specific feature set" direction
first: added `threat_roll{3,5}` / `creativity_roll{3,5}` (FPL's own attacking sub-indices,
already summed into `ict_index` but never exposed separately) to probe whether more granular
signal helped. Honest result — it didn't: R² unchanged (0.320 → 0.320), confirming the
regressor's problem is the squared-error objective on a skewed target, not missing features.

Built the other flagged direction instead: `src/models/train_haul_classifier.py` trains
classifiers for `P(points≥6)` and `P(points≥10)` — a different, answerable question — using
the same feature set and train/test split as the regressor. Compared baseline/logistic
regression/Random Forest/XGBoost; **XGBoost selected by Brier score (calibration), not
ROC-AUC** — RF/logistic used `class_weight="balanced"` for better class separation, but at a
real calibration cost (Brier 0.15–0.16) that would make a displayed "23% chance" not actually
mean 1-in-5; XGBoost trained on the natural class imbalance stayed well-calibrated (Brier
0.02–0.06) with equal-or-better ROC-AUC (~0.85–0.86 both thresholds) anyway. `predict_live.py`
now scores both classifiers alongside the point-estimate regressor, stored as
`p_return_6plus`/`p_haul_10plus` in `predictions`. Dashboard shows this as a **Ceiling %**
column and a differential-captain suggestion alongside the existing expected-points one — the
safe-pick-vs-upside-pick framing the plan's Risk Model section wanted, now real.

Two real bugs caught and fixed while wiring this in, not just the intended feature:
1. `predictions` had never cleaned up superseded rows from old `model_id`s (the table's PK
   includes `model_id`, so `ON CONFLICT` never touched them) — 1804 stale rows had
   accumulated, and Postgres's DESC-sorts-NULL-first default meant a naive query surfaced only
   pre-classifier NULL rows as the "top" results. `predict_live.py` now deletes existing rows
   for the gameweek before inserting fresh ones.
2. The dashboard's cached DB connection never called `autocommit = True`, so it sat "idle in
   transaction" for hours between reruns — this is what silently blocked an unrelated schema
   migration earlier the same session. Fixed at the source in `get_connection()`.

**Phase 6.6 — Fixture/opponent-strength features** (2026-08-22, complete)

Finished the other half of Phase 6.5's "haul-specific feature set" idea: `threat`/`creativity`
tested there were still player-only signal. The model had zero information about who a player
is actually about to face — `historical_features.py` never touched `opponent_team`/`was_home`/
`team_h_score`/`team_a_score`, even though they're present in every season of the source data.

Added `was_home` plus each row's own team's and opponent's rolling attack/defense form
(`own_attack_form`, `own_defense_form`, `opp_attack_form`, `opp_defense_form` — goals
for/against, leak-free `shift(1)` before a 5-fixture rolling window, same discipline as every
other feature here and the same window `v_team_form` already uses for the live dashboard's
Fixture Difficulty Score). Computed at the per-fixture level from each team's own match
results (`_compute_team_form`) before the double-gameweek aggregation, so a DGW's two fixtures
average together like any other per-fixture stat. One real gotcha: the archive's `team` column
on `merged_gw.csv` is the team's full *name* ("Man Utd"), but `opponent_team` is that season's
numeric id — not the same space despite the naming — so a `teams.csv` per season (now also
pulled by `download_historical_data.sh`) is needed to map name → id before the two can join.
Live serving (`live_features.py`) mirrors this from the `fixtures` table instead of the CSV
archive: rolling form as of the most recent finished result for the synthetic next-gameweek
row, and each team's next *unplayed* fixture (earliest by kickoff time) for the opponent/
home-away it hasn't faced yet — the same "form entering the next fixture" notion
`v_team_latest_form` already uses for the dashboard.

**Honest result**: a real but modest lift, not a breakthrough. The single held-out-season split
`train.py` uses moved from R²=0.320 (Random Forest, the prior deployed model) to R²=0.328
(XGBoost, which overtakes Random Forest as best on this fold now) — RF itself moved to 0.323.
Feature importance shows why: `opp_attack_form` is the most important of the five new features
but is still only ~0.6% of the deployed model's total decision weight (`minutes_roll3` alone is
still ~65%). Re-run across the full 5-fold walk-forward harness
(`notebooks/08_cross_validation.ipynb`'s split), the aggregate picture is flatter than the
single-split number suggests — mean R² across all folds barely moved (RF 0.297±0.020 →
0.298±0.020; XGBoost 0.295±0.026, close behind but not stably ahead) — the gain shows up mainly
on the fold with the most training data (the real deployed split), not uniformly across folds
with less history. Adopted anyway: it's a genuine, mechanistically sensible signal (a player in
front of a leaky defense should score more, and now the model can see that) with no
cross-validated downside, even though its practical size is small next to "will this player
play" and "how involved are they" — the same two features that already dominated before this
change and still do.

**Phase 7 — Quantile regression + Monte Carlo simulation** (2026-08-22, in progress — floor/
ceiling piece complete, injury-signal audit and LLM explanations not started)

`src/models/train_quantile_models.py` trains real floor (10th pct) / median (50th pct) /
ceiling (90th pct) models via `GradientBoostingRegressor(loss="quantile")` — not a simulated
spread from one point-estimate model's tree variance, which would reflect model *disagreement*
rather than true outcome variance. **Real finding while validating**: the naive calibration
check ("fraction of actual outcomes below the predicted quantile") looked badly wrong for
floor/median (63.3%/71.5% vs 10%/50% targets) — root-caused rather than blindly retrained
against: ~63% of all rows are exactly 0 points, so any near-zero floor prediction trivially
satisfies the check via ties on that whole majority. Conditioning on nailed-on starters only
(where blanking is genuinely rare) shows the real picture: floor calibrates to 13.0% (close),
median to 37.7% (a real, milder issue — runs a bit conservative for established starters).
Fixed the diagnostic itself so future retrains report the honest number.

`src/models/monte_carlo.py` samples each player's points via piecewise-linear inverse-CDF
through their floor/median/ceiling — flat lower tail (no information below the floor, avoids
implausible negative scores), but the upper tail deliberately extrapolates past ceiling rather
than capping there, since real FPL hauls (20+) are rare-but-real and capping would understate
exactly the upside this exists to surface. **Known simplification, stated plainly**: players
are sampled independently — real outcomes correlate within a team and across a fixture, so the
squad-level spread is somewhat tighter than reality. Verified: simulating Brian's actual squad
centers on ~23–24 points, matching his real GW1 score (24) almost exactly — good face validity
from one data point, to be re-checked as more gameweeks accumulate.

Dashboard: Player Rankings gets a Floor–Ceiling range column; My Squad gets a Monte Carlo
section with real percentile stats and a histogram of 10,000 simulated starting-XI outcomes.
See `notebooks/10_haul_classifier.ipynb` and `notebooks/11_monte_carlo.ipynb` for the full
writeups.

**Phase 7.5 — Injury/fixture-congestion features** (2026-08-26/27, complete — quantile-model
integration and LLM explanations still not started)

`status`/`chance_of_playing_next_round`/`news` are live-ingested (`player_price_snapshots`) but
confirmed absent from the historical training archive entirely, so this can't be a trained
feature from FPL's own live data alone. Backfilled real injury history instead from a verified
external dataset — "European Football Injuries (2020-2025)" (Kaggle, CC BY-SA 4.0, 15,603
records across the Big-5 European leagues) — into a new `player_injuries` table
(`src/ingestion/injuries_kaggle.py`), matched to our stable `player_code` via a new multi-pass
name matcher (`src/ingestion/injury_matching.py`: exact → substring-of-web_name → fuzzy,
disambiguated by club with a small alias table for abbreviations like "Man Utd"/"Manchester
United" that generic fuzzy matching can't bridge). 96.5% of EPL names matched; spot-checked
against real public record (Van Dijk's 255-day ACL tear, Saka's 99-day hamstring injury,
Maddison's Leicester→Tottenham transfer timing) before being trusted.

**First attempt, honestly reported as a miss**: four flat injury-history features
(`days_since_last_injury`, `injuries_last_365d`, `is_returning_from_injury`,
`injury_data_available`) slightly *hurt* accuracy (XGBoost R² 0.3275→0.321) — largely redundant
with what `minutes_roll3` already captures (an injured player already shows near-zero recent
minutes).

**Redesigned around the actual mechanism** (Sports Medicine 2022 systematic review on fixture
congestion and injury; the underlying UEFA studies): injury risk specifically rises when a
congested run (≤4 days between matches, the literature's standard threshold) combines with a
player's own prior injury history — particularly muscle/tendon injuries (hamstring, calf,
groin — the fatigue-accumulation mechanism), not injuries generally. Added
`days_since_last_match`/`matches_last_14d`/`is_congested` (own-team fixture congestion, EPL
matches only — a real, stated undercount for continentally active clubs, since cup/European
fixtures aren't in our archive), `muscle_injuries_last_365d` (keyword-classified from the real
`injury_type` strings observed), and `injury_congestion_risk` (their product, for linear
regression's benefit).

**Still didn't move the pooled R²** (0.321→0.322, noise-level) — but bucketing test predictions
by the interaction flag showed something real: the ~2% of rows where it fires are genuinely
harder to predict (R² drops to ~0.23 vs ~0.32 for everyone else) and score *more* on average,
not less (1.63 vs 1.16 actual mean points) — likely because surviving to "playing through a
congested run with a recent muscle injury on record" selects for first-team regulars managers
push through knocks, not a simple risk discount. **Conclusion: this is a volatility signal, not
a mean-shift signal** — a point-estimate regressor can only move its predicted mean, so it
structurally can't benefit from a "this one's noisier" signal even though the signal is real.

**Confirmed by wiring it into the quantile floor/ceiling models instead**
(`train_quantile_models.py` already trains on `FEATURE_COLS` directly, so no code change was
needed there beyond a retrain): the floor-ceiling spread for the ~2% of rows where the
interaction fires is **4.15 points vs. 2.83 for everyone else — 47% wider**, and the median is
higher too (0.80 vs. 0.60) — exactly the volatility this signal was supposed to represent,
showing up in the model type actually built to represent it. Real second bug caught while
re-running this: the quantile-crossing sanity check at the end of `train_quantile_models.py`
reloaded floor/median/ceiling from hardcoded static filenames that predated the run-unique-path
fix above, silently picking up stale months-old models and crashing on a feature-name mismatch
the first time it ran against the new feature set — fixed by keeping the fitted models in
memory instead of round-tripping through disk.

Mirrored into `live_features.py` for live serving (`_load_team_congestion`, reusing
`compute_injury_features`/`is_muscle_tendon_injury` from `historical_features.py` so training
and serving share identical recency math) whether or not the deployed model ends up using
these columns.

**Real bug found and fixed along the way, not just the intended feature**: `train.py` (and
`train_haul_classifier.py`/`train_quantile_models.py`) wrote every retrain to a *static*
filename (`models/{target}_{model_type}.joblib`), so re-running training silently overwrote
the file an OLDER `model_versions` row still pointed to — that row's recorded metrics/feature
list described a model that no longer existed at that path. This is a pre-existing bug (traced
back to `model_id=18` from 2026-08-21), not something introduced this session, but repeated
retraining today surfaced it as a live crash (`predict_live.py` tried to score 54 real features
through a stale artifact still recorded as the 45-feature Phase 6.6 model). Fixed by giving
every artifact a run-unique filename (a timestamp shared across one `run()`'s targets); cleaned
up the 60 already-stale rows by nulling their now-incorrect `artifact_path` (same convention
Model Lab's experiment rows already use to opt out of `predict_live.get_best_model`'s
selection) rather than deleting the historical record.

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
