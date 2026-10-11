-- Read-only database role for the tools server (src/tools_server) - what Muse and Claude read
-- through. SELECT on exactly the tables and views the tools query, nothing else: no write
-- grants anywhere, and no access to raw_snapshots, model_versions, reminder_log,
-- telegram_update_offset or the injury tables, which no tool needs.
--
-- Three layers, so one mistake isn't enough to write: no INSERT/UPDATE/DELETE grants, the role
-- defaults every transaction to read-only, and the server opens each call with
-- SET TRANSACTION READ ONLY as well.
--
-- Run as postgres (table owner). Set the password separately, never in this file:
--   ALTER ROLE fergies_tools PASSWORD '...';
-- Idempotent.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fergies_tools') THEN
        CREATE ROLE fergies_tools LOGIN;
    END IF;
END $$;

ALTER ROLE fergies_tools SET default_transaction_read_only = on;
ALTER ROLE fergies_tools SET statement_timeout = '15s';

GRANT USAGE ON SCHEMA public TO fergies_tools;

DO $$
DECLARE
    t text;
BEGIN
    -- Tables: SELECT grant plus an RLS read policy (RLS is enabled on every public table, and a
    -- role with no policy sees no rows - see sql/rls_fergies_app.sql).
    FOREACH t IN ARRAY ARRAY[
        'teams', 'players', 'player_seasons', 'gameweeks', 'fixtures', 'player_gameweek_stats',
        'player_price_snapshots', 'predictions', 'managers', 'manager_gameweeks', 'squad_picks',
        'pending_approvals'
    ] LOOP
        EXECUTE format('GRANT SELECT ON public.%I TO fergies_tools', t);
        EXECUTE format('DROP POLICY IF EXISTS fergies_tools_read ON public.%I', t);
        EXECUTE format(
            'CREATE POLICY fergies_tools_read ON public.%I FOR SELECT TO fergies_tools USING (true)', t
        );
    END LOOP;
    -- Views the tools read.
    FOREACH t IN ARRAY ARRAY[
        'v_player_season_totals', 'v_fixture_difficulty', 'v_manager_gw_comparison',
        'v_team_match_results', 'v_team_form', 'v_team_latest_form'
    ] LOOP
        EXECUTE format('GRANT SELECT ON public.%I TO fergies_tools', t);
    END LOOP;
END $$;
