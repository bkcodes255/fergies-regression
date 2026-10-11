-- Plan snapshots: the Python job (src/recommendations/plan_snapshots.py, last step of
-- ingest_and_predict.yml) writes the current recommendation here, and the read-only Supabase
-- Edge Function (supabase/functions/tools) reads it back - so no planning logic ever runs in
-- TypeScript. Rows are kept per gameweek, which also gives get_gameweek_review a record of what
-- was planned before each deadline.
--
-- plan_json is built by src/recommendations/plan_payloads.py, the same code the Python tools
-- server returns live - the Edge Function hands it back verbatim.
--
-- Run as postgres. Idempotent. Run this before sql/tools_server_role.sql (which grants reads on
-- these tables) and before the snapshot step first runs.

CREATE TABLE IF NOT EXISTS transfer_plans (
    season               TEXT NOT NULL,
    event_id             INTEGER NOT NULL,
    horizon              TEXT NOT NULL CHECK (horizon IN ('1gw', '5gw')),
    plan_json            JSONB NOT NULL,     -- transfer_plan_payload(): transfers, hits, bank, free transfers, XI points
    prediction_gameweek  INTEGER,            -- the predictions this was computed from
    squad_sync_gameweek  INTEGER,            -- the synced squad it was computed from
    computed_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (season, event_id, horizon)
);

-- One lineup per transfer plan: built on the squad AFTER that plan's transfers, the same rule
-- the approval flow applies when a transfer executes (src/notify/approval_bot.py).
CREATE TABLE IF NOT EXISTS lineup_plans (
    season               TEXT NOT NULL,
    event_id             INTEGER NOT NULL,
    horizon              TEXT NOT NULL CHECK (horizon IN ('1gw', '5gw')),
    plan_json            JSONB NOT NULL,     -- lineup_plan_payload(): formation, XI, bench, captain, vice
    prediction_gameweek  INTEGER,
    squad_sync_gameweek  INTEGER,
    computed_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (season, event_id, horizon)
);

-- Each player's fixture-weighted points over the next DEFAULT_HORIZON gameweeks
-- (src/recommendations/horizon.py), so search and top-predictions can rank on it without
-- re-implementing the horizon maths outside Python.
CREATE TABLE IF NOT EXISTS player_horizons (
    season               TEXT NOT NULL,
    event_id             INTEGER NOT NULL,   -- the gameweek the window starts at
    player_code          INTEGER NOT NULL REFERENCES players(player_code),
    predicted_points     NUMERIC(6,3),
    horizon_points       NUMERIC(7,3),
    horizon_fixtures     SMALLINT,
    computed_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (season, event_id, player_code)
);

-- Supabase grants anon/authenticated access to new public tables by default, and a table
-- without RLS would be readable and writable through the Data API with the public anon key.
ALTER TABLE transfer_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE lineup_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE player_horizons ENABLE ROW LEVEL SECURITY;

DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['transfer_plans', 'lineup_plans', 'player_horizons'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
            EXECUTE format('REVOKE ALL ON public.%I FROM anon', t);
        END IF;
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
            EXECUTE format('REVOKE ALL ON public.%I FROM authenticated', t);
        END IF;
        -- The writer, if the ingest job connects as fergies_app (as postgres it bypasses RLS).
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fergies_app') THEN
            EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON public.%I TO fergies_app', t);
            EXECUTE format('DROP POLICY IF EXISTS fergies_app_all ON public.%I', t);
            EXECUTE format(
                'CREATE POLICY fergies_app_all ON public.%I FOR ALL TO fergies_app USING (true) WITH CHECK (true)', t
            );
        END IF;
    END LOOP;
END $$;
