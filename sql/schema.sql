-- Fergie's Regression — Phase 1 schema (local Postgres)
--
-- Design principles this schema enforces:
-- 1. FPL's `id` fields (players/teams/gameweeks/fixtures) are SEASON-SCOPED and get reused
--    next season. Every entity is keyed on its season-stable identity where one exists
--    (`code` for players/teams), with the season-local numeric `id` kept only as a
--    convenience column for hitting the live API, never as a primary key.
-- 2. Never overwrite. Time-varying attributes (price, ownership, status, team) are captured
--    as dated snapshots / per-gameweek facts, not mutated in place.
-- 3. Raw API payloads are preserved in full (raw_snapshots), so "what did we know at the
--    time" is always reconstructable — this is what makes backtesting leak-free later.

CREATE TABLE teams (
    team_code       INTEGER PRIMARY KEY,        -- FPL `code` — stable across seasons
    name            TEXT NOT NULL,
    short_name      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Season-scoped attributes of a team (strength ratings are recalculated every season).
CREATE TABLE team_season_stats (
    team_code                INTEGER NOT NULL REFERENCES teams(team_code),
    season                   TEXT NOT NULL,          -- e.g. '2026-27'
    fpl_id                   INTEGER NOT NULL,        -- this season's numeric id (1-20)
    strength                 INTEGER,
    strength_overall_home    INTEGER,
    strength_overall_away    INTEGER,
    strength_attack_home     INTEGER,
    strength_attack_away     INTEGER,
    strength_defence_home    INTEGER,
    strength_defence_away    INTEGER,
    pulled_at                TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (team_code, season)
);

CREATE TABLE players (
    player_code     INTEGER PRIMARY KEY,        -- FPL `code` — stable across seasons
    first_name      TEXT NOT NULL,
    second_name     TEXT NOT NULL,
    web_name        TEXT NOT NULL,
    birth_date      DATE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Season-scoped attributes of a player: starting position/team for that season.
-- Mid-season transfers and status changes are NOT modeled here (see player_price_snapshots
-- and the denormalized team_code on player_gameweek_stats for the time-accurate view).
CREATE TABLE player_seasons (
    player_code     INTEGER NOT NULL REFERENCES players(player_code),
    season          TEXT NOT NULL,
    fpl_id          INTEGER NOT NULL,           -- this season's numeric id
    team_code       INTEGER NOT NULL REFERENCES teams(team_code),
    element_type    SMALLINT NOT NULL,          -- 1=GKP 2=DEF 3=MID 4=FWD
    pulled_at       TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (player_code, season)
);

CREATE TABLE gameweeks (
    season                  TEXT NOT NULL,
    event_id                INTEGER NOT NULL,
    name                    TEXT NOT NULL,
    deadline_time           TIMESTAMPTZ NOT NULL,
    finished                BOOLEAN NOT NULL DEFAULT false,
    data_checked            BOOLEAN NOT NULL DEFAULT false,
    is_current               BOOLEAN NOT NULL DEFAULT false,
    average_entry_score     INTEGER,
    highest_score           INTEGER,
    most_selected           INTEGER,            -- player_code
    most_captained          INTEGER,            -- player_code
    most_vice_captained     INTEGER,            -- player_code
    top_element              INTEGER,            -- player_code
    chip_plays              JSONB,
    pulled_at               TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (season, event_id)
);

CREATE TABLE fixtures (
    season               TEXT NOT NULL,
    fpl_fixture_id       INTEGER NOT NULL,
    event_id             INTEGER,               -- nullable: blank-GW fixtures can be unscheduled
    team_h_code          INTEGER NOT NULL REFERENCES teams(team_code),
    team_a_code          INTEGER NOT NULL REFERENCES teams(team_code),
    team_h_score         INTEGER,
    team_a_score         INTEGER,
    kickoff_time         TIMESTAMPTZ,
    finished             BOOLEAN NOT NULL DEFAULT false,   -- official: bonus/stats locked, may lag full-time by 1+ days
    finished_provisional BOOLEAN NOT NULL DEFAULT false,   -- match is actually over (full-time whistle); use this for "has this team played" logic
    team_h_difficulty    SMALLINT,
    team_a_difficulty    SMALLINT,
    pulled_at            TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (season, fpl_fixture_id),
    FOREIGN KEY (season, event_id) REFERENCES gameweeks(season, event_id)
);

-- Core fact table: one row per player per gameweek (aggregated across fixtures if the
-- player's team had a double-gameweek — the fixture-level breakdown is preserved in
-- `explain`, sourced verbatim from the live API's per-fixture explain array).
CREATE TABLE player_gameweek_stats (
    season                              TEXT NOT NULL,
    event_id                            INTEGER NOT NULL,
    player_code                         INTEGER NOT NULL REFERENCES players(player_code),
    team_code                           INTEGER NOT NULL REFERENCES teams(team_code),  -- snapshot at the time
    minutes                             INTEGER NOT NULL DEFAULT 0,
    starts                              INTEGER NOT NULL DEFAULT 0,
    goals_scored                        INTEGER NOT NULL DEFAULT 0,
    assists                             INTEGER NOT NULL DEFAULT 0,
    clean_sheets                        INTEGER NOT NULL DEFAULT 0,
    goals_conceded                      INTEGER NOT NULL DEFAULT 0,
    own_goals                           INTEGER NOT NULL DEFAULT 0,
    penalties_saved                     INTEGER NOT NULL DEFAULT 0,
    penalties_missed                    INTEGER NOT NULL DEFAULT 0,
    yellow_cards                        INTEGER NOT NULL DEFAULT 0,
    red_cards                           INTEGER NOT NULL DEFAULT 0,
    saves                               INTEGER NOT NULL DEFAULT 0,
    bonus                               INTEGER NOT NULL DEFAULT 0,
    bps                                 INTEGER NOT NULL DEFAULT 0,
    influence                           NUMERIC(6,1),
    creativity                          NUMERIC(6,1),
    threat                              NUMERIC(6,1),
    ict_index                           NUMERIC(6,1),
    clearances_blocks_interceptions     INTEGER NOT NULL DEFAULT 0,
    tackles                             INTEGER NOT NULL DEFAULT 0,
    recoveries                          INTEGER NOT NULL DEFAULT 0,
    defensive_contribution              INTEGER NOT NULL DEFAULT 0,  -- raw CBIT/CBIRT count, NOT points; see data dictionary
    expected_goals                      NUMERIC(6,2),
    expected_assists                    NUMERIC(6,2),
    expected_goal_involvements          NUMERIC(6,2),
    expected_goals_conceded             NUMERIC(6,2),
    total_points                        INTEGER NOT NULL DEFAULT 0,
    in_dreamteam                        BOOLEAN NOT NULL DEFAULT false,
    explain                             JSONB,   -- raw per-fixture point breakdown from the live API
    pulled_at                           TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (season, event_id, player_code),
    FOREIGN KEY (season, event_id) REFERENCES gameweeks(season, event_id)
);

-- Daily time series: price/ownership/availability move independently of gameweeks
-- (prices can change any day; injury status can change mid-week).
CREATE TABLE player_price_snapshots (
    season                       TEXT NOT NULL,
    player_code                  INTEGER NOT NULL REFERENCES players(player_code),
    snapshot_date                DATE NOT NULL,
    now_cost                     INTEGER NOT NULL,   -- tenths of £m, e.g. 65 = £6.5m
    cost_change_event            INTEGER,
    cost_change_start            INTEGER,
    selected_by_percent          NUMERIC(5,1),
    transfers_in_event           INTEGER,
    transfers_out_event          INTEGER,
    status                       CHAR(1),             -- a=available d=doubtful i=injured s=suspended u=unavailable
    chance_of_playing_next_round SMALLINT,
    news                         TEXT,
    pulled_at                    TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (season, player_code, snapshot_date)
);

-- Full raw API response payloads, kept forever. This is the reproducibility backstop:
-- if a parsed column turns out wrong or FPL changes a field's meaning, the source of
-- truth is still here.
CREATE TABLE raw_snapshots (
    id          BIGSERIAL PRIMARY KEY,
    endpoint    TEXT NOT NULL,      -- e.g. 'bootstrap-static', 'fixtures', 'event/{n}/live'
    season      TEXT NOT NULL,
    event_id    INTEGER,            -- null for endpoints not tied to a specific gameweek
    pulled_at   TIMESTAMPTZ NOT NULL,
    payload     JSONB NOT NULL
);

-- Model provenance/tracking (Phase 4+). One row per trained model variant, so we can
-- answer "did the fancier model actually improve decision-making" later, not just assume it.
CREATE TABLE model_versions (
    model_id            BIGSERIAL PRIMARY KEY,
    model_type           TEXT NOT NULL,       -- 'baseline' | 'linear_regression' | 'random_forest' | 'xgboost'
    target               TEXT NOT NULL,       -- 'minutes' | 'points_per_90'
    training_seasons     TEXT[] NOT NULL,
    test_season          TEXT NOT NULL,
    features             JSONB NOT NULL,
    hyperparameters       JSONB,
    mae                  NUMERIC(8,4),
    rmse                 NUMERIC(8,4),
    r2                   NUMERIC(8,4),
    roc_auc              NUMERIC(6,4),        -- classifier rows only (e.g. target='haul_10plus')
    brier_score          NUMERIC(6,4),        -- classifier rows only - calibration, lower is better
    artifact_path        TEXT,                -- where the serialized model lives (models/, gitignored)
    trained_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_experiment         BOOLEAN NOT NULL DEFAULT false,  -- Model Lab dashboard tab run, not a
    -- real training-pipeline candidate. Already structurally excluded from live serving by
    -- artifact_path IS NULL (predict_live.get_best_model requires artifact_path IS NOT NULL),
    -- but this flag makes that intent explicit and queryable rather than implicit.
    diagnostics            JSONB               -- Model Lab only: train-set metrics, overfit gap,
    -- bootstrap R² CI, paired-bootstrap p-value vs. session baseline, and (model-type specific)
    -- OLS coefficient p-values or permutation importance. See src/models/experiment.py.
);

-- Your actual FPL team (Phase 3+). One manager (entry) tracked for now - this schema doesn't
-- assume single-manager, but only Brian's entry is ingested today.
CREATE TABLE managers (
    entry_id            INTEGER PRIMARY KEY,   -- FPL's own entry id - stable, not season-scoped
    player_first_name   TEXT,
    player_last_name    TEXT,
    favourite_team      INTEGER,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE manager_gameweeks (
    entry_id             INTEGER NOT NULL REFERENCES managers(entry_id),
    season               TEXT NOT NULL,
    event_id             INTEGER NOT NULL,
    points               INTEGER,
    total_points         INTEGER,
    overall_rank         INTEGER,
    bank                 INTEGER,               -- tenths of £m
    team_value           INTEGER,               -- tenths of £m
    event_transfers      INTEGER,
    event_transfers_cost INTEGER,
    points_on_bench      INTEGER,
    active_chip          TEXT,
    pulled_at            TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (entry_id, season, event_id)
);

CREATE TABLE squad_picks (
    entry_id        INTEGER NOT NULL REFERENCES managers(entry_id),
    season          TEXT NOT NULL,
    event_id        INTEGER NOT NULL,
    player_code     INTEGER NOT NULL REFERENCES players(player_code),
    squad_position  SMALLINT NOT NULL,    -- 1-11 starting XI (order = formation slot), 12-15 bench
    multiplier      SMALLINT NOT NULL,    -- 0=benched 1=normal 2=captain 3=triple captain
    is_captain      BOOLEAN NOT NULL DEFAULT false,
    is_vice_captain BOOLEAN NOT NULL DEFAULT false,
    pulled_at       TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (entry_id, season, event_id, player_code)
);

-- Live predictions for an upcoming gameweek, generated by applying a trained model_versions
-- entry to our own current-season data (not the historical training data).
CREATE TABLE predictions (
    season               TEXT NOT NULL,
    event_id             INTEGER NOT NULL,      -- the gameweek being predicted (not yet played)
    player_code          INTEGER NOT NULL REFERENCES players(player_code),
    model_id             BIGINT NOT NULL REFERENCES model_versions(model_id),
    predicted_points     NUMERIC(6,3) NOT NULL,
    p_return_6plus       NUMERIC(5,4),        -- P(actual points >= 6), from the haul classifier
    p_haul_10plus        NUMERIC(5,4),        -- P(actual points >= 10) - the "ceiling" signal
    floor_points         NUMERIC(6,3),        -- 10th percentile, quantile regression (Phase 7)
    median_points        NUMERIC(6,3),        -- 50th percentile - distinct from predicted_points,
                                               -- which is the regressor's mean-ish MSE estimate
    ceiling_points       NUMERIC(6,3),        -- 90th percentile
    predicted_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (season, event_id, player_code, model_id)
);

CREATE INDEX idx_pgs_player ON player_gameweek_stats(player_code);
CREATE INDEX idx_pgs_event ON player_gameweek_stats(season, event_id);
CREATE INDEX idx_price_player ON player_price_snapshots(player_code);
CREATE INDEX idx_fixtures_event ON fixtures(season, event_id);
CREATE INDEX idx_raw_endpoint ON raw_snapshots(endpoint, season, pulled_at);
