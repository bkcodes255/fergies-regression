-- RLS was enabled on every public table (outside this repo, via Supabase) with no
-- policies, which silently hides all rows from fergies_app - the role the Streamlit
-- dashboard connects as. These policies restore fergies_app's pre-RLS access while
-- keeping RLS on, so Supabase's anon/authenticated Data API roles stay locked out.
-- Table grants still bound what fergies_app can actually do. Run as postgres
-- (the table owner); idempotent.

DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'fixtures', 'gameweeks', 'manager_gameweeks', 'managers', 'model_versions',
        'pending_approvals', 'player_external_ids', 'player_gameweek_stats',
        'player_injuries', 'player_price_snapshots', 'player_seasons', 'players',
        'predictions', 'raw_snapshots', 'reminder_log', 'squad_picks',
        'team_season_stats', 'teams', 'telegram_update_offset'
    ] LOOP
        EXECUTE format('DROP POLICY IF EXISTS fergies_app_all ON public.%I', t);
        EXECUTE format(
            'CREATE POLICY fergies_app_all ON public.%I FOR ALL TO fergies_app USING (true) WITH CHECK (true)',
            t
        );
    END LOOP;
END $$;
