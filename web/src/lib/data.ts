// The page's data, baked in at build time. The deploy workflow writes src/data/this-week.json
// from the live database (python -m src.web_data.export); without it - local development - the
// committed sample is used, and the page says so. A deploy build sets REQUIRE_WEB_DATA=1 so a
// failed export stops the deploy instead of shipping the sample.

export interface Magnet { id: number; n: string; full: string; p: number | null; cap: boolean; vice: boolean }
export interface Board { xi: Magnet[][]; bench: Magnet[] }
export type BoardCall =
  | { kind: "captain"; n: number; from: number; to: number; label: string }
  | { kind: "lineup"; n: number; pairs: { on: number; off: number }[]; label: string }
  | { kind: "transfer"; n: number; sell: number | null; sell_name: string; buy_name: string; net: number; hit: number; label: string };
export interface Call {
  id: string; kind: string; title: string; detail: string; gain: number; done: boolean; shout?: string; mutter?: string;
}
export interface Fixture { gw: number; opp: string; home: boolean; grade: number | null }
export interface Player {
  name: string; team: string; position: string; price: number | null; predicted: number | null; horizon: number | null;
  floor: number | null; median: number | null; ceiling: number | null; haul: number | null; fixtures: Fixture[];
}
export interface PlanPlayer { player_code: number | null; web_name: string; position?: string; team?: string; price?: number; value?: number }
export interface Plan {
  horizon: string; runs_if_silent: boolean; value: string;
  transfers: { order: number; sell: PlanPlayer; buy: PlanPlayer; gain: number; hit_cost: number; net: number; uses_free_transfer: boolean }[];
  total_net_gain: number; total_hit_cost: number; free_transfers: number; bank_before: number; bank_after: number;
  xi_points_before: number; xi_points_after: number;
}
export interface Outcomes {
  samples: number; p10: number; p50: number; p90: number; mean: number; p_beat_60: number; p_beat_80: number;
  edges: number[]; counts: number[];
}
export interface ThisWeek {
  generated_at: string; season: string; last_gw: number | null; predicted_gw: number | null;
  deadline: { event_id: number; name: string; time: string } | null;
  horizon: number; full_blast: number;
  squad: null | {
    free_transfers: number; bank: number; board: Board; after_board: Board; board_calls: BoardCall[]; calls: Call[];
    players: Record<string, Player>; plans: { "1gw": Plan; "5gw": Plan }; outcomes: Outcomes;
  };
}

declare const process: { env: Record<string, string | undefined> }; // Node, at build time

const files = import.meta.glob<ThisWeek>("../data/*.json", { eager: true, import: "default" });
const live = files["../data/this-week.json"];
if (!live && process.env.REQUIRE_WEB_DATA) {
  throw new Error("REQUIRE_WEB_DATA is set but src/data/this-week.json is missing - the export step failed.");
}
export const isSample = !live;
export const data: ThisWeek = live ?? files["../data/sample-this-week.json"];

/** "Sat 17 Oct, 11:00 UK time" - FPL deadlines are set in UK time, so that's the clock shown. */
export function ukTime(iso: string): string {
  const d = new Date(iso);
  const part = (o: Intl.DateTimeFormatOptions) => new Intl.DateTimeFormat("en-GB", { timeZone: "Europe/London", ...o }).format(d);
  return `${part({ weekday: "short" })} ${part({ day: "numeric" })} ${part({ month: "short" })}, ${part({ hour: "2-digit", minute: "2-digit", hour12: false })} UK time`;
}

export const signed = (v: number, digits = 1) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(digits);
