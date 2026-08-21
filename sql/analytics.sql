-- Phase 2 — Analytics views. Built as SQL views (not a Python layer) because everything
-- here is naturally expressed as aggregation/window functions over data already in Postgres;
-- Python feature engineering starts in Phase 4 once we're building model inputs, not before.
--
-- All rolling-window views degrade gracefully with partial data (early season): a "5-GW
-- rolling average" with only 1 GW played just averages that 1 GW. Values will sharpen as
-- more gameweeks are ingested — they are not wrong now, just thin-sample.

-- =====================================================================================
-- 1. Player season-to-date totals + efficiency ratios
-- =====================================================================================
CREATE OR REPLACE VIEW v_player_season_totals AS
SELECT
    s.season,
    s.player_code,
    p.web_name,
    ps.element_type,
    ps.team_code,
    COUNT(*) FILTER (WHERE s.minutes > 0) AS games_played,
    SUM(s.minutes) AS total_minutes,
    SUM(s.total_points) AS total_points,
    SUM(s.goals_scored) AS goals,
    SUM(s.assists) AS assists,
    SUM(s.clean_sheets) AS clean_sheets,
    SUM(s.bonus) AS bonus,
    SUM(s.bps) AS bps,
    SUM(s.defensive_contribution) AS defensive_contribution,
    SUM(s.expected_goals) AS expected_goals,
    SUM(s.expected_assists) AS expected_assists,
    ROUND(SUM(s.total_points)::numeric / NULLIF(COUNT(*) FILTER (WHERE s.minutes > 0), 0), 2) AS points_per_game,
    ROUND(SUM(s.total_points)::numeric / NULLIF(SUM(s.minutes), 0) * 90, 2) AS points_per_90,
    ROUND(SUM(s.goals_scored)::numeric / NULLIF(SUM(s.minutes), 0) * 90, 2) AS goals_per_90,
    ROUND(SUM(s.assists)::numeric / NULLIF(SUM(s.minutes), 0) * 90, 2) AS assists_per_90,
    ROUND((SUM(s.goals_scored) + SUM(s.assists))::numeric / NULLIF(SUM(s.minutes), 0) * 90, 2) AS returns_per_90,
    latest_price.now_cost,
    ROUND(SUM(s.total_points)::numeric / NULLIF(latest_price.now_cost / 10.0, 0), 2) AS points_per_million,
    latest_price.selected_by_percent,
    latest_price.status
FROM player_gameweek_stats s
JOIN players p ON p.player_code = s.player_code
JOIN player_seasons ps ON ps.player_code = s.player_code AND ps.season = s.season
JOIN LATERAL (
    SELECT now_cost, selected_by_percent, status
    FROM player_price_snapshots pps
    WHERE pps.player_code = s.player_code AND pps.season = s.season
    ORDER BY snapshot_date DESC
    LIMIT 1
) latest_price ON true
GROUP BY s.season, s.player_code, p.web_name, ps.element_type, ps.team_code,
         latest_price.now_cost, latest_price.selected_by_percent, latest_price.status;

-- =====================================================================================
-- 2. Player rolling form: 3-GW / 5-GW rolling averages, plus a recency-weighted form
--    (weights per the plan's own example: most recent 5 GWs at 40/25/15/12/8%,
--    renormalized over however many gameweeks actually exist so far this season).
-- =====================================================================================
CREATE OR REPLACE VIEW v_player_rolling_form AS
SELECT
    season, event_id, player_code, total_points, minutes,
    ROUND(AVG(total_points::numeric) OVER w3, 2) AS points_avg_3gw,
    ROUND(AVG(total_points::numeric) OVER w5, 2) AS points_avg_5gw,
    ROUND(AVG(minutes::numeric) OVER w3, 1) AS minutes_avg_3gw,
    ROUND(AVG(minutes::numeric) OVER w5, 1) AS minutes_avg_5gw,
    ROUND(AVG(goals_scored::numeric) OVER w3, 2) AS goals_avg_3gw,
    ROUND(AVG(assists::numeric) OVER w3, 2) AS assists_avg_3gw,
    ROUND(AVG(bps::numeric) OVER w3, 1) AS bps_avg_3gw,
    ROUND(AVG(expected_goals) OVER w3, 3) AS xg_avg_3gw,
    ROUND(AVG(expected_assists) OVER w3, 3) AS xa_avg_3gw,
    ROUND(AVG(total_points::numeric) OVER (PARTITION BY season, player_code ORDER BY event_id), 2) AS points_avg_season
FROM player_gameweek_stats
WINDOW
    w3 AS (PARTITION BY season, player_code ORDER BY event_id ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),
    w5 AS (PARTITION BY season, player_code ORDER BY event_id ROWS BETWEEN 4 PRECEDING AND CURRENT ROW);

CREATE OR REPLACE VIEW v_player_weighted_form AS
SELECT
    pgs.season, pgs.event_id, pgs.player_code,
    ROUND(SUM(recent.total_points * recent.weight) / NULLIF(SUM(recent.weight), 0), 2) AS weighted_points_form
FROM player_gameweek_stats pgs
CROSS JOIN LATERAL (
    SELECT recent_row.total_points,
           CASE recent_row.rn WHEN 1 THEN 0.40 WHEN 2 THEN 0.25 WHEN 3 THEN 0.15
                               WHEN 4 THEN 0.12 WHEN 5 THEN 0.08 END AS weight
    FROM (
        SELECT r.total_points, ROW_NUMBER() OVER (ORDER BY r.event_id DESC) AS rn
        FROM player_gameweek_stats r
        WHERE r.season = pgs.season AND r.player_code = pgs.player_code AND r.event_id <= pgs.event_id
    ) recent_row
    WHERE recent_row.rn <= 5
) AS recent
GROUP BY pgs.season, pgs.event_id, pgs.player_code;

-- =====================================================================================
-- 3. Team form: actual match results unpivoted to one row per team per fixture, then
--    rolling last-5 attacking/defensive strength.
-- =====================================================================================
-- Uses finished_provisional, not finished: finished only flips once bonus/stats are
-- officially locked (can lag full-time by 1+ days), which would silently exclude most
-- recently-played matches from team-form calculations. See data_dictionary.md.
CREATE OR REPLACE VIEW v_team_match_results AS
SELECT season, fpl_fixture_id, event_id, kickoff_time,
       team_h_code AS team_code, team_a_code AS opponent_code,
       true AS is_home, team_h_score AS goals_for, team_a_score AS goals_against
FROM fixtures WHERE finished_provisional AND team_h_score IS NOT NULL
UNION ALL
SELECT season, fpl_fixture_id, event_id, kickoff_time,
       team_a_code AS team_code, team_h_code AS opponent_code,
       false AS is_home, team_a_score AS goals_for, team_h_score AS goals_against
FROM fixtures WHERE finished_provisional AND team_a_score IS NOT NULL;

CREATE OR REPLACE VIEW v_team_form AS
SELECT *,
    ROUND(AVG(goals_for::numeric) OVER w5, 2) AS attack_form_5,
    ROUND(AVG(goals_against::numeric) OVER w5, 2) AS defense_form_5
FROM v_team_match_results
WINDOW w5 AS (PARTITION BY season, team_code ORDER BY kickoff_time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW);

-- Most recent form snapshot per team — used to rate upcoming fixtures below.
CREATE OR REPLACE VIEW v_team_latest_form AS
SELECT DISTINCT ON (season, team_code)
    season, team_code, attack_form_5, defense_form_5
FROM v_team_form
ORDER BY season, team_code, kickoff_time DESC;

-- =====================================================================================
-- 4. Fixture Difficulty Score (v1) for upcoming (unplayed) fixtures, one row per team
--    per fixture (difficulty is asymmetric — it's rated from one team's perspective).
--
--    Formula, per the reviewed plan: 30% opponent defensive strength + 30% opponent
--    attacking strength + 20% home/away + 10% own recent form + 10% FPL's own FDR.
--    This is a v1 with real interpretive choices (z-score normalization, the exact
--    home/away and "recent form" treatment) that the plan itself treats as provisional
--    ("Fergie's Fixture Rating (TM)") — expect to revisit the weights once more
--    gameweeks give the underlying team-form numbers real signal instead of n=1 noise.
-- =====================================================================================
CREATE OR REPLACE VIEW v_fixture_difficulty AS
WITH perspectives AS (
    SELECT season, fpl_fixture_id, event_id, kickoff_time,
           team_h_code AS team_code, team_a_code AS opponent_code,
           true AS is_home, team_h_difficulty AS fpl_fdr
    FROM fixtures WHERE NOT finished_provisional
    UNION ALL
    SELECT season, fpl_fixture_id, event_id, kickoff_time,
           team_a_code AS team_code, team_h_code AS opponent_code,
           false AS is_home, team_a_difficulty AS fpl_fdr
    FROM fixtures WHERE NOT finished_provisional
),
league_norms AS (
    SELECT season,
           AVG(attack_form_5) AS league_attack_avg, STDDEV(attack_form_5) AS league_attack_sd,
           AVG(defense_form_5) AS league_defense_avg, STDDEV(defense_form_5) AS league_defense_sd
    FROM v_team_latest_form
    GROUP BY season
)
SELECT
    p.season, p.fpl_fixture_id, p.event_id, p.kickoff_time,
    p.team_code, p.opponent_code, p.is_home, p.fpl_fdr,
    opp.attack_form_5 AS opponent_attack_form,
    opp.defense_form_5 AS opponent_defense_form,
    own.attack_form_5 AS own_attack_form,
    ROUND((
        0.30 * COALESCE((opp.defense_form_5 - ln.league_defense_avg) / NULLIF(ln.league_defense_sd, 0), 0) * -1
      + 0.30 * COALESCE((opp.attack_form_5 - ln.league_attack_avg) / NULLIF(ln.league_attack_sd, 0), 0)
      + 0.20 * (CASE WHEN p.is_home THEN -0.5 ELSE 0.5 END)
      + 0.10 * COALESCE((ln.league_attack_avg - own.attack_form_5) / NULLIF(ln.league_attack_sd, 0), 0) * -1
      + 0.10 * (p.fpl_fdr::numeric - 3)
    )::numeric, 3) AS fixture_difficulty_score
FROM perspectives p
LEFT JOIN v_team_latest_form opp ON opp.season = p.season AND opp.team_code = p.opponent_code
LEFT JOIN v_team_latest_form own ON own.season = p.season AND own.team_code = p.team_code
LEFT JOIN league_norms ln ON ln.season = p.season;
