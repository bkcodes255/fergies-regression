// Fergie's Regression read-only tools (contract v0.1) as one Supabase Edge Function.
//
//   POST /functions/v1/tools/invoke    { "tool": "get_squad", "arguments": {} }
//   GET  /functions/v1/tools/catalog   every tool with its description and input schema
//   GET  /functions/v1/tools/healthz   no auth, no database
//
// No planning logic lives here. Transfer plans, post-transfer lineups and 5-gameweek horizon
// points are computed in Python (src/recommendations/plan_snapshots.py, the last step of the
// ingest job) and stored in transfer_plans / lineup_plans / player_horizons; this function only
// reads. The response shapes mirror the Python reference implementation in
// src/tools_server/tools.py field for field.
//
// Read-only by construction: TOOLS_DATABASE_URL connects as the fergies_tools role
// (sql/tools_server_role.sql: SELECT only, read-only by default), and every call runs in a
// READ ONLY transaction. Never the service-role key - that one can write everything.
//
// Secrets (supabase secrets set ...): TOOLS_DATABASE_URL, TOOLS_API_KEYS (comma-separated, one
// per client, 32+ chars each), FPL_SEASON, FPL_ENTRY_ID. Deploy with --no-verify-jwt: callers
// authenticate with their own API key, not a Supabase JWT, so the gateway's JWT check must be off.

import postgres from "npm:postgres@3.4.5";

const VERSION = "0.1.0";
const SEASON = Deno.env.get("FPL_SEASON") ?? "2026-27";
const ENTRY_ID = Number(Deno.env.get("FPL_ENTRY_ID") ?? "") || null;
const API_KEYS = (Deno.env.get("TOOLS_API_KEYS") ?? "").split(",").map((k) => k.trim()).filter(Boolean);
const MIN_KEY_LENGTH = 32;
const HORIZON_GAMEWEEKS = 5; // DEFAULT_HORIZON in src/recommendations/horizon.py
const MAX_LIMIT = 100;
const POSITION: Record<number, string> = { 1: "GKP", 2: "DEF", 3: "MID", 4: "FWD" };

const AUTO_SUBMIT_WINDOW =
  "Anything still 'proposed' is auto-submitted between about 2 hours and 30 minutes before the " +
  "deadline (GitHub Actions runs the scheduler late, so the exact time varies). An untouched " +
  "transfer approval submits the next-gameweek plan, not the 5-gameweek one.";

class ToolError extends Error {
  constructor(public code: string, message: string) {
    super(message);
  }
}

// --- database -------------------------------------------------------------------------------

type Sql = ReturnType<typeof postgres>;
// deno-lint-ignore no-explicit-any
type Tx = any;
let sql: Sql | null = null;

function db(): Sql {
  if (sql === null) {
    const url = Deno.env.get("TOOLS_DATABASE_URL");
    if (!url) throw new ToolError("not_configured", "No database configured: set TOOLS_DATABASE_URL.");
    // prepare: false - Supabase's transaction-mode pooler can't keep prepared statements.
    sql = postgres(url, { prepare: false, max: 2, idle_timeout: 20, connect_timeout: 10 });
  }
  return sql;
}

function readOnly<T>(fn: (tx: Tx) => Promise<T>): Promise<T> {
  return db().begin("read only", fn) as Promise<T>;
}

// --- formatting (same conventions as src/recommendations/plan_payloads.py) ------------------

type Row = Record<string, unknown>;

function num(v: unknown, digits = 2): number | null {
  if (v === null || v === undefined) return null;
  const f = Number(v);
  if (!Number.isFinite(f)) return null;
  const p = 10 ** digits;
  return Math.round(f * p) / p;
}

function int(v: unknown): number | null {
  const f = num(v, 0);
  return f === null ? null : Math.trunc(f);
}

function text(v: unknown): string | null {
  return v === null || v === undefined || v === "" ? null : String(v);
}

function iso(v: unknown): string | null {
  if (v === null || v === undefined) return null;
  const d = v instanceof Date ? v : new Date(String(v));
  return new Date(Math.floor(d.getTime() / 1000) * 1000).toISOString().replace(".000Z", "Z");
}

function hoursUntil(v: unknown): number | null {
  if (v === null || v === undefined) return null;
  const d = v instanceof Date ? v : new Date(String(v));
  return num((d.getTime() - Date.now()) / 3_600_000, 1);
}

// The 1 (easiest) to 5 (hardest) bands of fixture_grade in src/recommendations/horizon.py.
function fixtureGrade(score: unknown): number | null {
  if (score === null || score === undefined) return null;
  const s = Number(score);
  const bands = [-0.35, -0.1, 0.1, 0.35];
  for (let i = 0; i < bands.length; i++) if (s <= bands[i]) return i + 1;
  return 5;
}

// --- shared loaders -------------------------------------------------------------------------

async function nextGameweek(tx: Tx): Promise<Row | null> {
  const rows = await tx`
    SELECT season, event_id, name, deadline_time FROM gameweeks
    WHERE season = ${SEASON} AND NOT finished AND deadline_time > now()
    ORDER BY deadline_time ASC LIMIT 1`;
  return rows[0] ?? null;
}

async function requireNextGameweek(tx: Tx): Promise<Row> {
  const gw = await nextGameweek(tx);
  if (gw === null) throw new ToolError("no_upcoming_gameweek", "There's no upcoming deadline in the ingested data.");
  return gw;
}

// Every player this season - the same base as gameweek_plan.load_rankings (v_player_season_totals),
// with the prediction, quantiles, availability news, and the stored 5-gameweek horizon points.
async function playerPool(tx: Tx): Promise<Row[]> {
  const rows: Row[] = await tx`
    SELECT st.player_code, p.web_name, p.first_name, p.second_name, pos.element_type,
           t.short_name AS team, st.now_cost, st.status, st.selected_by_percent,
           pr.event_id AS prediction_gw, pr.predicted_points, pr.floor_points, pr.median_points,
           pr.ceiling_points, pr.p_haul_10plus,
           snap.chance_of_playing_next_round, snap.news,
           ph.horizon_points, ph.horizon_fixtures
    FROM v_player_season_totals st
    JOIN players p ON p.player_code = st.player_code
    JOIN player_seasons pos ON pos.player_code = st.player_code AND pos.season = st.season
    JOIN teams t ON t.team_code = st.team_code
    LEFT JOIN predictions pr ON pr.player_code = st.player_code AND pr.season = st.season
        AND pr.event_id = (SELECT MAX(event_id) FROM predictions WHERE season = st.season)
    LEFT JOIN LATERAL (
        SELECT chance_of_playing_next_round, news FROM player_price_snapshots pps
        WHERE pps.player_code = st.player_code AND pps.season = st.season
        ORDER BY snapshot_date DESC LIMIT 1
    ) snap ON true
    LEFT JOIN player_horizons ph ON ph.player_code = st.player_code AND ph.season = st.season
        AND ph.event_id = (SELECT MAX(event_id) FROM player_horizons WHERE season = st.season)
    WHERE st.season = ${SEASON}`;
  for (const r of rows) {
    r.position = POSITION[Number(r.element_type)];
    r.price = Number(r.now_cost) / 10;
  }
  return rows;
}

function poolEntry(r: Row) {
  const predicted = r.predicted_points;
  return {
    player_code: int(r.player_code), web_name: text(r.web_name), position: text(r.position),
    team: text(r.team), price: num(r.price, 1),
    status: text(r.status), chance_of_playing: int(r.chance_of_playing_next_round), news: text(r.news),
    predicted_points: num(predicted),
    horizon_points: predicted === null || predicted === undefined ? null : num(r.horizon_points),
    horizon_fixtures: int(r.horizon_fixtures),
    floor: num(r.floor_points), median: num(r.median_points), ceiling: num(r.ceiling_points),
    haul_chance: num(r.p_haul_10plus, 3), selected_by_percent: num(r.selected_by_percent, 1),
  };
}

function filterPool(pool: Row[], position: string | null, team: string | null, maxPrice: number | null): Row[] {
  return pool.filter((r) =>
    (!position || r.position === position) &&
    (!team || String(r.team).toUpperCase() === team.toUpperCase()) &&
    (maxPrice === null || Number(r.price) <= maxPrice + 1e-9)
  );
}

function byDesc(key: string) {
  return (a: Row, b: Row) => {
    const x = a[key] === null || a[key] === undefined ? -Infinity : Number(a[key]);
    const y = b[key] === null || b[key] === undefined ? -Infinity : Number(b[key]);
    return y - x || Number(a.player_code) - Number(b.player_code);
  };
}

const clampLimit = (n: number) => Math.max(1, Math.min(Math.trunc(n), MAX_LIMIT));

async function storedPlans(tx: Tx, table: "transfer_plans" | "lineup_plans", eventId: number): Promise<Row[]> {
  const rows = table === "transfer_plans"
    ? await tx`SELECT horizon, plan_json, computed_at, prediction_gameweek, squad_sync_gameweek
               FROM transfer_plans WHERE season = ${SEASON} AND event_id = ${eventId} ORDER BY horizon`
    : await tx`SELECT horizon, plan_json, computed_at, prediction_gameweek, squad_sync_gameweek
               FROM lineup_plans WHERE season = ${SEASON} AND event_id = ${eventId} ORDER BY horizon`;
  return rows;
}

function withProvenance(row: Row) {
  return {
    ...(row.plan_json as Row),
    computed_at: iso(row.computed_at),
    computed_from: { prediction_gameweek: int(row.prediction_gameweek), squad_sync_gameweek: int(row.squad_sync_gameweek) },
  };
}

const PLAN_NOT_READY = "No plan stored for this gameweek yet: the ingest job writes one after each data refresh.";

// --- tools ----------------------------------------------------------------------------------

async function getStatus() {
  return await readOnly(async (tx) => {
    const [r] = await tx`
      SELECT
        (SELECT MAX(pulled_at) FROM player_price_snapshots WHERE season = ${SEASON}) AS prices_at,
        (SELECT MAX(pulled_at) FROM gameweeks WHERE season = ${SEASON}) AS gameweeks_at,
        (SELECT MAX(pulled_at) FROM player_gameweek_stats WHERE season = ${SEASON}) AS stats_at,
        (SELECT MAX(event_id) FROM predictions WHERE season = ${SEASON}) AS prediction_gw,
        (SELECT MAX(predicted_at) FROM predictions WHERE season = ${SEASON}) AS predicted_at,
        (SELECT MAX(event_id) FROM squad_picks WHERE season = ${SEASON} AND entry_id = ${ENTRY_ID ?? -1}) AS squad_gw`;
    const gw = await nextGameweek(tx);
    const plans = gw ? await tx`
      SELECT MIN(computed_at) AS computed_at FROM transfer_plans
      WHERE season = ${SEASON} AND event_id = ${gw.event_id}` : [];
    const plansAt = plans[0]?.computed_at ?? null;

    const pulled = [r.prices_at, r.gameweeks_at, r.stats_at].filter((t) => t !== null).map((t) => new Date(t).getTime());
    const lastIngest = pulled.length ? new Date(Math.max(...pulled)) : null;
    const predictionGw = int(r.prediction_gw), squadGw = int(r.squad_gw);
    const warnings: string[] = [];
    if (gw) {
      if (predictionGw === null) warnings.push("No predictions are stored yet.");
      else if (predictionGw !== gw.event_id) {
        warnings.push(`Predictions are for GW${predictionGw}, but the next deadline is GW${gw.event_id}: anything about this gameweek uses stale predictions.`);
      }
      if (squadGw !== null && squadGw < Number(gw.event_id) - 1) {
        warnings.push(`Your squad was last synced for GW${squadGw}; changes since then aren't visible yet.`);
      }
      if (plansAt === null) warnings.push("No plans are stored for this gameweek yet; the next ingest run writes them.");
      else if (r.predicted_at && new Date(plansAt) < new Date(r.predicted_at)) {
        warnings.push("The stored plans predate the latest predictions; the next ingest run refreshes them.");
      }
    }
    if (lastIngest && Date.now() - lastIngest.getTime() > 24 * 3_600_000) {
      warnings.push(`The last data refresh was over a day ago (${iso(lastIngest)}).`);
    }
    return {
      season: SEASON,
      last_ingest_at: iso(lastIngest),
      latest_prediction_gameweek: predictionGw,
      predictions_made_at: iso(r.predicted_at),
      latest_squad_sync_gameweek: squadGw,
      plans_computed_at: iso(plansAt),
      entry_configured: ENTRY_ID !== null,
      next_deadline: gw === null ? null : {
        gameweek: gw.event_id, name: gw.name, deadline_at: iso(gw.deadline_time), hours_remaining: hoursUntil(gw.deadline_time),
      },
      warnings,
    };
  });
}

async function getCurrentGameweek() {
  const gw = await readOnly(nextGameweek);
  if (gw === null) {
    return { upcoming: false, message: "No upcoming deadline in the ingested data (season over, or not ingested)." };
  }
  return {
    upcoming: true, event_id: gw.event_id, name: gw.name, deadline_at: iso(gw.deadline_time),
    hours_remaining: hoursUntil(gw.deadline_time), finished: false,
  };
}

async function getSquad() {
  if (ENTRY_ID === null) throw new ToolError("not_configured", "FPL_ENTRY_ID isn't set, so there's no squad.");
  return await readOnly(async (tx) => {
    const picks: Row[] = await tx`
      SELECT sp.player_code, p.web_name, ps.element_type, t.short_name AS team,
             sp.squad_position, sp.multiplier, sp.is_captain, sp.is_vice_captain, sp.event_id,
             snap.now_cost, snap.status, snap.chance_of_playing_next_round, snap.news,
             pr.predicted_points, pr.event_id AS prediction_gw
      FROM squad_picks sp
      JOIN players p ON p.player_code = sp.player_code
      JOIN player_seasons ps ON ps.player_code = sp.player_code AND ps.season = sp.season
      JOIN teams t ON t.team_code = ps.team_code
      LEFT JOIN LATERAL (
          SELECT now_cost, status, chance_of_playing_next_round, news FROM player_price_snapshots pps
          WHERE pps.player_code = sp.player_code AND pps.season = sp.season
          ORDER BY snapshot_date DESC LIMIT 1
      ) snap ON true
      LEFT JOIN predictions pr ON pr.player_code = sp.player_code AND pr.season = sp.season
          AND pr.event_id = (SELECT MAX(event_id) FROM predictions WHERE season = sp.season)
      WHERE sp.entry_id = ${ENTRY_ID} AND sp.season = ${SEASON}
        AND sp.event_id = (SELECT MAX(event_id) FROM squad_picks WHERE entry_id = ${ENTRY_ID} AND season = ${SEASON})
      ORDER BY sp.squad_position`;
    if (picks.length === 0) throw new ToolError("no_squad", "No squad has been ingested for this entry yet.");
    const [latest] = await tx`
      SELECT bank, team_value FROM manager_gameweeks WHERE entry_id = ${ENTRY_ID} AND season = ${SEASON}
      ORDER BY event_id DESC LIMIT 1`;
    // Free transfers come from the stored plan (Python's compute_free_transfers), never recomputed here.
    const gw = await nextGameweek(tx);
    const plan = gw ? (await storedPlans(tx, "transfer_plans", Number(gw.event_id))).find((p) => p.horizon === "1gw") : undefined;
    const asOf = Number(picks[0].event_id);
    const predictionGws = picks.map((p) => p.prediction_gw).filter((g) => g !== null).map(Number);
    return {
      as_of_gameweek: asOf,
      prediction_gameweek: predictionGws.length ? Math.max(...predictionGws) : null,
      bank: latest && latest.bank !== null ? num(Number(latest.bank) / 10, 1) : null,
      team_value: latest && latest.team_value !== null ? num(Number(latest.team_value) / 10, 1) : null,
      free_transfers: plan ? int((plan.plan_json as Row).free_transfers) : null,
      players: picks.map((r) => ({
        player_code: int(r.player_code), web_name: text(r.web_name), position: POSITION[Number(r.element_type)],
        team: text(r.team), price: num(Number(r.now_cost) / 10, 1),
        predicted_points: num(r.predicted_points),
        status: text(r.status), chance_of_playing: int(r.chance_of_playing_next_round), news: text(r.news),
        starting: Number(r.multiplier) > 0, squad_position: int(r.squad_position),
        captain: Boolean(r.is_captain), vice_captain: Boolean(r.is_vice_captain),
      })),
      note: `This is the squad as picked for GW${asOf}. Transfers or lineup changes made in FPL since then show up after the next data refresh.` +
        (plan ? "" : " Free transfers show once the ingest job has stored this gameweek's plan."),
    };
  });
}

async function searchPlayers(a: Args) {
  const pool = filterPool(await readOnly(playerPool), a.position, a.team, a.max_price);
  const q = String(a.query ?? "").trim().toLowerCase();
  const hits = q
    ? pool.filter((r) => `${r.web_name ?? ""} ${r.first_name ?? ""} ${r.second_name ?? ""}`.toLowerCase().includes(q))
    : pool;
  hits.sort(byDesc("horizon_points"));
  return { count: hits.length, players: hits.slice(0, clampLimit(a.limit)).map(poolEntry), horizon_gameweeks: HORIZON_GAMEWEEKS };
}

async function getTopPredictions(a: Args) {
  let pool = filterPool(await readOnly(playerPool), a.position, null, a.max_price)
    .filter((r) => r.predicted_points !== null && r.predicted_points !== undefined);
  if (a.available_only) pool = pool.filter((r) => r.status === "a");
  pool.sort(byDesc(a.horizon === "5gw" ? "horizon_points" : "predicted_points"));
  const gws = pool.map((r) => Number(r.prediction_gw));
  return {
    horizon: a.horizon, prediction_gameweek: gws.length ? Math.max(...gws) : null, count: pool.length,
    players: pool.slice(0, clampLimit(a.limit)).map(poolEntry),
  };
}

async function getTransferPlan(a: Args) {
  return await readOnly(async (tx) => {
    const gw = await requireNextGameweek(tx);
    const rows = (await storedPlans(tx, "transfer_plans", Number(gw.event_id)))
      .filter((r) => !a.horizon || r.horizon === a.horizon);
    if (rows.length === 0) throw new ToolError("plan_not_ready", PLAN_NOT_READY);
    return { gameweek: gw.event_id, deadline_at: iso(gw.deadline_time), plans: rows.map(withProvenance) };
  });
}

async function getLineupPlan(a: Args) {
  return await readOnly(async (tx) => {
    const gw = await requireNextGameweek(tx);
    const row = (await storedPlans(tx, "lineup_plans", Number(gw.event_id))).find((r) => r.horizon === a.horizon);
    if (!row) throw new ToolError("plan_not_ready", PLAN_NOT_READY);
    return withProvenance(row);
  });
}

async function getFixtures(a: Args) {
  return await readOnly(async (tx) => {
    const gw = await requireNextGameweek(tx);
    const start = Number(gw.event_id), end = start + a.gameweeks_ahead;
    const rows: Row[] = await tx`
      SELECT fd.fpl_fixture_id, fd.event_id, fd.kickoff_time, fd.is_home, fd.fpl_fdr,
             fd.fixture_difficulty_score, t.short_name AS team, o.short_name AS opponent
      FROM v_fixture_difficulty fd
      JOIN teams t ON t.team_code = fd.team_code
      JOIN teams o ON o.team_code = fd.opponent_code
      WHERE fd.season = ${SEASON} AND fd.event_id >= ${start} AND fd.event_id < ${end}
      ORDER BY fd.event_id, fd.kickoff_time, fd.fpl_fixture_id`;
    const teams: string[] = (await tx`
      SELECT DISTINCT t.short_name FROM player_seasons ps JOIN teams t ON t.team_code = ps.team_code
      WHERE ps.season = ${SEASON}`).map((r: Row) => String(r.short_name));
    const team = a.team ? String(a.team).toUpperCase() : null;

    const byFixture = new Map<unknown, Row[]>();
    for (const r of rows) byFixture.set(r.fpl_fixture_id, [...(byFixture.get(r.fpl_fixture_id) ?? []), r]);
    const side = (r: Row) => ({
      team: r.team, fpl_difficulty: int(r.fpl_fdr),
      app_difficulty_score: num(r.fixture_difficulty_score, 3), app_difficulty_grade: fixtureGrade(r.fixture_difficulty_score),
    });
    const fixtures = [];
    for (const g of byFixture.values()) {
      const home = g.find((r) => r.is_home), away = g.find((r) => !r.is_home);
      if (!home || !away) continue;
      if (team && ![String(home.team).toUpperCase(), String(away.team).toUpperCase()].includes(team)) continue;
      fixtures.push({ gameweek: int(home.event_id), kickoff_at: iso(home.kickoff_time), home: side(home), away: side(away) });
    }
    const counts = new Map<string, number>();
    for (const r of rows) counts.set(`${r.team}|${r.event_id}`, (counts.get(`${r.team}|${r.event_id}`) ?? 0) + 1);
    const blanks = [], doubles = [];
    for (const t of [...teams].sort()) {
      if (team && t.toUpperCase() !== team) continue;
      for (let e = start; e < end; e++) {
        const n = counts.get(`${t}|${e}`) ?? 0;
        if (n === 0) blanks.push({ team: t, gameweek: e });
        else if (n > 1) doubles.push({ team: t, gameweek: e, fixtures: n });
      }
    }
    return {
      from_gameweek: start, to_gameweek: end - 1, fixtures, blanks, doubles,
      difficulty_note:
        "fpl_difficulty is FPL's published 1-5 rating. app_difficulty_score is this app's own " +
        "measure from rolling team form (roughly zero-centred, negative = easier); " +
        "app_difficulty_grade buckets it 1 (easiest) to 5 (hardest). The two often disagree.",
    };
  });
}

function approvalSummary(kind: string, plan: Row) {
  if (kind === "transfer") {
    const variant = (key: string, horizon: string) => {
      const ts = (((plan[key] as Row) ?? {}).transfers as Row[]) ?? [];
      return {
        horizon, runs_if_silent: horizon === "1gw",
        transfers: ts.map((t) => ({ sell: t.sell, buy: t.buy, net: num(t.net), hit_cost: int(t.hit) })),
        total_net: ts.length ? num(ts.reduce((s, t) => s + Number(t.net), 0)) : 0.0,
      };
    };
    return { options: [variant("next_gw", "1gw"), variant("five_gw", "5gw")], free_transfers: plan.free_transfers ?? null };
  }
  if (kind === "captain") return { captain: plan.captain_web_name ?? null, vice_captain: plan.vice_web_name ?? null };
  return {
    formation: ((plan.formation as number[]) ?? []).join("-"),
    starting_xi: plan.starting_names ?? [], bench: plan.bench_names ?? [],
  };
}

async function approvals(tx: Tx, eventId: number) {
  const rows: Row[] = await tx`
    SELECT kind, plan_json, status, result_detail, created_at, decided_at FROM pending_approvals
    WHERE season = ${SEASON} AND event_id = ${eventId}
    ORDER BY array_position(ARRAY['transfer','captain','squad'], kind)`;
  return rows.map((r) => ({
    kind: r.kind, status: r.status, plan: approvalSummary(String(r.kind), (r.plan_json as Row) ?? {}),
    result: text(r.result_detail), proposed_at: iso(r.created_at), decided_at: iso(r.decided_at),
  }));
}

async function getPendingApprovals(a: Args) {
  return await readOnly(async (tx) => {
    const gw = await nextGameweek(tx);
    const eventId = a.gameweek ?? (gw ? Number(gw.event_id) : null);
    if (eventId === null) throw new ToolError("no_upcoming_gameweek", "There's no upcoming deadline; pass a gameweek.");
    const list = await approvals(tx, eventId);
    const [d] = await tx`SELECT deadline_time FROM gameweeks WHERE season = ${SEASON} AND event_id = ${eventId}`;
    return {
      gameweek: eventId, deadline_at: iso(d?.deadline_time ?? null), hours_remaining: hoursUntil(d?.deadline_time ?? null),
      approvals: list, auto_submit: AUTO_SUBMIT_WINDOW,
      note: list.length ? null :
        "No approvals sent for this gameweek yet. They go out in the T-3h tier, which opens up to 6 hours before the deadline.",
    };
  });
}

async function getGameweekReview(a: Args) {
  if (ENTRY_ID === null) throw new ToolError("not_configured", "FPL_ENTRY_ID isn't set, so there's nothing to review.");
  return await readOnly(async (tx) => {
    let gameweek: number | null = a.gameweek;
    if (gameweek === null) {
      const [last] = await tx`SELECT MAX(event_id) AS e FROM gameweeks WHERE season = ${SEASON} AND finished`;
      gameweek = int(last.e);
      if (gameweek === null) throw new ToolError("no_finished_gameweek", "No gameweek has finished yet.");
    }
    const [gw] = await tx`SELECT event_id, name, finished FROM gameweeks WHERE season = ${SEASON} AND event_id = ${gameweek}`;
    if (!gw) throw new ToolError("unknown_gameweek", `GW${gameweek} isn't in the ingested data.`);
    const [r] = await tx`SELECT * FROM v_manager_gw_comparison WHERE entry_id = ${ENTRY_ID} AND season = ${SEASON} AND event_id = ${gameweek}`;
    const [h] = await tx`SELECT * FROM manager_gameweeks WHERE entry_id = ${ENTRY_ID} AND season = ${SEASON} AND event_id = ${gameweek}`;
    const picks: Row[] = await tx`
      SELECT sp.player_code, p.web_name, sp.squad_position, sp.multiplier, sp.is_captain,
             (SELECT SUM(total_points) FROM player_gameweek_stats s
               WHERE s.season = sp.season AND s.event_id = sp.event_id AND s.player_code = sp.player_code) AS actual,
             (SELECT predicted_points FROM predictions pr
               WHERE pr.season = sp.season AND pr.event_id = sp.event_id AND pr.player_code = sp.player_code
               ORDER BY predicted_at DESC LIMIT 1) AS predicted
      FROM squad_picks sp JOIN players p ON p.player_code = sp.player_code
      WHERE sp.entry_id = ${ENTRY_ID} AND sp.season = ${SEASON} AND sp.event_id = ${gameweek}
      ORDER BY sp.squad_position`;
    const list = await approvals(tx, gameweek);
    const raw: Row[] = await tx`SELECT kind, plan_json FROM pending_approvals WHERE season = ${SEASON} AND event_id = ${gameweek}`;
    const plans = Object.fromEntries(raw.map((x) => [x.kind, (x.plan_json as Row) ?? {}]));
    const modelCodes = ((plans.squad?.starting_player_codes as number[]) ?? []).map(Number);
    const captainCode = plans.captain?.captain_player_code ? Number(plans.captain.captain_player_code) : null;
    const codes = [...new Set([...modelCodes, ...(captainCode ? [captainCode] : [])])];
    const actual = new Map<number, number>();
    if (codes.length) {
      const pts: Row[] = await tx`
        SELECT player_code, SUM(total_points) AS pts FROM player_gameweek_stats
        WHERE season = ${SEASON} AND event_id = ${gameweek} AND player_code IN ${tx(codes)} GROUP BY player_code`;
      for (const p of pts) actual.set(Number(p.player_code), Number(p.pts ?? 0));
    }
    const stored = await tx`
      SELECT 'transfer' AS kind, horizon, plan_json FROM transfer_plans WHERE season = ${SEASON} AND event_id = ${gameweek}
      UNION ALL
      SELECT 'lineup' AS kind, horizon, plan_json FROM lineup_plans WHERE season = ${SEASON} AND event_id = ${gameweek}`;

    const fielded = picks.filter((p) => Number(p.multiplier) > 0);
    const fieldedPoints = fielded.length ? fielded.reduce((s, p) => s + Number(p.actual ?? 0) * Number(p.multiplier), 0) : null;
    let modelPoints: number | null = null;
    if (modelCodes.length) {
      modelPoints = modelCodes.reduce((s, c) => s + (actual.get(c) ?? 0), 0);
      if (captainCode) modelPoints += actual.get(captainCode) ?? 0;
    }
    return {
      gameweek, name: gw.name, finished: Boolean(gw.finished),
      result: !h ? null : {
        official_points: int(h.points), transfer_cost: int(h.event_transfers_cost), points_on_bench: int(h.points_on_bench),
        overall_rank: int(h.overall_rank), gameweek_rank: int(h.event_rank), chip: text(h.active_chip),
        gameweek_average: r ? int(r.gw_average) : null, gameweek_highest: r ? int(r.gw_highest) : null,
        model_predicted_points: r ? num(r.model_predicted_points) : null, fpl_predicted_points: r ? num(r.epl_predicted_points) : null,
      },
      fielded: picks.map((p) => ({
        player_code: int(p.player_code), web_name: p.web_name, starting: Number(p.multiplier) > 0,
        captain: Boolean(p.is_captain), multiplier: int(p.multiplier),
        predicted_points: num(p.predicted), actual_points: int(p.actual),
      })),
      model_vs_you: {
        your_xi_points: num(fieldedPoints, 0), model_proposal_xi_points: num(modelPoints, 0),
        difference: fieldedPoints !== null && modelPoints !== null ? num(fieldedPoints - modelPoints, 0) : null,
        basis: "Actual points of each starting XI with the captain doubled, before auto-subs and transfer costs.",
      },
      approvals: list,
      planned: stored.length
        ? stored.map((s: Row) => ({ kind: s.kind, horizon: s.horizon, plan: s.plan_json }))
        : null,
      planned_note: stored.length ? null : `No stored plans for GW${gameweek}: snapshots only exist from when the ingest step started writing them.`,
      overrides: [],
      overrides_note: "Overrides aren't recorded yet; they arrive with gameweek plans.",
    };
  });
}

// --- tool registry: descriptions and input schemas (mirror src/tools_server/tools.py) --------

type Param = {
  type: "string" | "integer" | "number" | "boolean"; description: string; enum?: string[];
  minimum?: number; maximum?: number; exclusiveMinimum?: number; nullable?: boolean; default: unknown;
};
// deno-lint-ignore no-explicit-any
type Args = Record<string, any>;
type Tool = { description: string; params: Record<string, Param>; run: (a: Args) => Promise<unknown> };

const position: Param = { type: "string", enum: ["GKP", "DEF", "MID", "FWD"], nullable: true, default: null, description: "Only this position." };
const maxPrice: Param = { type: "number", exclusiveMinimum: 0, nullable: true, default: null, description: "Maximum price in £m, e.g. 6.5." };
const limit: Param = { type: "integer", minimum: 1, maximum: MAX_LIMIT, default: 25, description: "How many to return (max 100)." };
const horizonOf = (def: string | null, description: string): Param =>
  ({ type: "string", enum: ["1gw", "5gw"], nullable: def === null, default: def, description });

const TOOLS: Record<string, Tool> = {
  get_status: {
    description: "How fresh the data is: when it was last ingested, which gameweek the predictions and the synced squad are for, when the stored plans were computed, and the next deadline. Check this before answering anything time-sensitive; read `warnings` first - it says when numbers are stale.",
    params: {}, run: getStatus,
  },
  get_current_gameweek: {
    description: "The next gameweek whose deadline hasn't passed: id, name, deadline and hours remaining.",
    params: {}, run: getCurrentGameweek,
  },
  get_squad: {
    description: "Your current 15 as of the last synced gameweek: position, team, price, next-gameweek prediction, availability status and news, who starts and who's captain - plus bank, team value and free transfers for the coming gameweek.",
    params: {}, run: getSquad,
  },
  search_players: {
    description: "Find players by name, position, team and price. Each comes with price, next-gameweek prediction, 5-gameweek fixture-weighted points, floor/ceiling, availability status and news. Sorted by 5-gameweek points.",
    params: {
      query: { type: "string", default: "", description: "Part of the player's name, e.g. 'saka'. Empty matches everyone." },
      position, team: { type: "string", nullable: true, default: null, description: "Team short name, e.g. 'ARS'." },
      max_price: maxPrice, limit,
    },
    run: searchPlayers,
  },
  get_top_predictions: {
    description: "The model's highest-predicted players, by next gameweek or the next 5 (fixture-weighted), with floor/median/ceiling and haul chance. For questions like 'best £6m midfielder for the next 5 gameweeks'. Predictions exist for the upcoming gameweek only.",
    params: {
      horizon: horizonOf("5gw", "'1gw' ranks by next-gameweek points, '5gw' by fixture-weighted points over the next 5."),
      position, max_price: maxPrice,
      available_only: { type: "boolean", default: false, description: "Leave out players flagged doubtful, injured, suspended or unavailable." },
      limit,
    },
    run: getTopPredictions,
  },
  get_transfer_plan: {
    description: "The model's suggested transfers for the coming gameweek - the same plans the Telegram approval offers. There are two: judged on next-gameweek points ('1gw') or fixture-weighted points over 5 gameweeks ('5gw'). `runs_if_silent` marks the one that auto-submits if nobody approves or rejects it (always '1gw'). xi_points_before/after are the best XI's points with the captain counted twice. `computed_at` says when the plan was stored.",
    params: { horizon: horizonOf(null, "'1gw' or '5gw'. Omit to get both.") },
    run: getTransferPlan,
  },
  get_lineup_plan: {
    description: "The best starting XI, bench order, captain and vice-captain for the coming gameweek, built from your squad AFTER the transfer plan - which is what the approval flow sets: when a transfer goes through, the lineup and captain are re-planned from the live squad by this same rule (best formation by next-gameweek points; captain and vice are the XI's top two).",
    params: { horizon: horizonOf("1gw", "Which transfer plan to build on: '1gw' (the one that runs if silent) or '5gw'.") },
    run: getLineupPlan,
  },
  get_fixtures: {
    description: "Upcoming fixtures with kickoff times and two difficulty measures per side: FPL's own rating (1-5) and this app's score (form-based; negative is easier), also as a 1-5 grade. The 5-gameweek plans use the app's score, not FPL's. Also lists blanks and doubles.",
    params: {
      gameweeks_ahead: { type: "integer", minimum: 1, maximum: 10, default: 5, description: "How many gameweeks from the next deadline (1-10)." },
      team: { type: "string", nullable: true, default: null, description: "Only fixtures involving this team short name, e.g. 'LIV'." },
    },
    run: getFixtures,
  },
  get_pending_approvals: {
    description: "The Telegram approvals for a gameweek (transfer, captain, lineup): what each proposes, its status (proposed / approved / rejected / executing / executed / failed), the result, and when anything undecided will auto-submit. Read-only: this can't approve or reject.",
    params: { gameweek: { type: "integer", minimum: 1, maximum: 38, nullable: true, default: null, description: "Gameweek id; defaults to the next deadline." } },
    run: getPendingApprovals,
  },
  get_gameweek_review: {
    description: "A finished gameweek side by side: what the model planned (stored plans) and proposed (the approvals), what you fielded, and the actual points - including what the model's proposed XI and captain would have scored. Overrides are listed once they exist (planned; empty for now).",
    params: { gameweek: { type: "integer", minimum: 1, maximum: 38, nullable: true, default: null, description: "Gameweek id; defaults to the last finished gameweek." } },
    run: getGameweekReview,
  },
};

function inputSchema(params: Record<string, Param>) {
  const properties: Record<string, unknown> = {};
  for (const [name, p] of Object.entries(params)) {
    const { nullable, ...rest } = p;
    properties[name] = nullable ? { ...rest, type: [p.type, "null"] } : rest;
  }
  return { type: "object", properties, additionalProperties: false };
}

function validate(params: Record<string, Param>, input: unknown): Args {
  if (input === null || input === undefined) input = {};
  if (typeof input !== "object" || Array.isArray(input)) throw new ToolError("invalid_arguments", "arguments must be a JSON object.");
  const args: Args = {};
  for (const key of Object.keys(input as object)) {
    if (!(key in params)) throw new ToolError("invalid_arguments", `${key}: unexpected argument.`);
  }
  for (const [name, p] of Object.entries(params)) {
    let v = (input as Args)[name];
    if (v === undefined) { args[name] = p.default; continue; }
    if (v === null) {
      if (!p.nullable) throw new ToolError("invalid_arguments", `${name}: may not be null.`);
      args[name] = null; continue;
    }
    const bad = (why: string) => new ToolError("invalid_arguments", `${name}: ${why}.`);
    if (p.type === "string" && typeof v !== "string") throw bad("must be a string");
    if (p.type === "boolean" && typeof v !== "boolean") throw bad("must be true or false");
    if (p.type === "number" && typeof v !== "number") throw bad("must be a number");
    if (p.type === "integer" && !(typeof v === "number" && Number.isInteger(v))) throw bad("must be an integer");
    if (p.enum && !p.enum.includes(v)) throw bad(`must be one of ${p.enum.join(", ")}`);
    if (p.minimum !== undefined && v < p.minimum) throw bad(`must be at least ${p.minimum}`);
    if (p.maximum !== undefined && v > p.maximum) throw bad(`must be at most ${p.maximum}`);
    if (p.exclusiveMinimum !== undefined && v <= p.exclusiveMinimum) throw bad(`must be greater than ${p.exclusiveMinimum}`);
    if (p.type === "string") v = v.trim();
    args[name] = v;
  }
  return args;
}

// --- HTTP -----------------------------------------------------------------------------------

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });
const ok = (data: unknown) => json({ ok: true, data });
const fail = (code: string, message: string, status: number, headers: Record<string, string> = {}) =>
  json({ ok: false, error: { code, message } }, status, headers);

function sameKey(a: string, b: string): boolean {
  const x = new TextEncoder().encode(a), y = new TextEncoder().encode(b);
  let diff = x.length ^ y.length;
  for (let i = 0; i < Math.max(x.length, y.length); i++) diff |= (x[i] ?? 0) ^ (y[i] ?? 0);
  return diff === 0;
}

function authorized(req: Request): boolean {
  const header = req.headers.get("authorization") ?? "";
  const token = header.toLowerCase().startsWith("bearer ") ? header.slice(7).trim() : "";
  return token !== "" && API_KEYS.some((k) => sameKey(token, k));
}

export async function handler(req: Request): Promise<Response> {
  const path = new URL(req.url).pathname.replace(/\/+$/, "");
  const route = path.slice(path.lastIndexOf("/") + 1);
  if (req.method === "GET" && route === "healthz") return json({ ok: true, version: VERSION });

  if (API_KEYS.length === 0 || API_KEYS.some((k) => k.length < MIN_KEY_LENGTH)) {
    return fail("not_configured", "The server has no valid API keys configured, so it refuses every request.", 503);
  }
  if (!authorized(req)) {
    return fail("unauthorized", "Missing or invalid API key (Authorization: Bearer <key>).", 401, { "www-authenticate": "Bearer" });
  }

  if (req.method === "GET" && route === "catalog") {
    return ok({
      version: VERSION,
      tools: Object.entries(TOOLS).map(([name, t]) => ({ name, description: t.description, input_schema: inputSchema(t.params) })),
    });
  }
  if (req.method !== "POST" || route !== "invoke") {
    return fail("not_found", "Use POST .../tools/invoke with {\"tool\", \"arguments\"}, or GET .../tools/catalog.", 404);
  }

  let body: Args;
  try {
    body = await req.json();
  } catch {
    return fail("invalid_json", "The request body must be JSON: {\"tool\": \"...\", \"arguments\": {...}}.", 400);
  }
  if (typeof body !== "object" || body === null || Array.isArray(body)) {
    return fail("invalid_arguments", "The request body must be a JSON object.", 400);
  }
  const tool = TOOLS[String(body.tool)];
  if (!tool) return fail("unknown_tool", `No tool named ${JSON.stringify(body.tool)}. GET .../tools/catalog lists them.`, 404);
  try {
    return ok(await tool.run(validate(tool.params, body.arguments)));
  } catch (e) {
    if (e instanceof ToolError) return fail(e.code, e.message, e.code === "invalid_arguments" ? 400 : 422);
    console.error(`Tool ${body.tool} failed:`, e); // full detail stays in the function logs
    return fail("internal_error", "The tool failed unexpectedly; the function logs have details.", 500);
  }
}

Deno.serve(handler);
