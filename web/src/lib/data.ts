// The site's data, baked in at build time and never served from an API. The deploy workflow writes
// src/data/<name>.json from the live database (python -m src.web_data.export); without them - local
// development - the committed src/data/sample-<name>.json is used, and the page says so. A deploy
// build sets REQUIRE_WEB_DATA=1 so a failed export stops the deploy instead of shipping a sample.
//
// Never import this file from a browser script except with `import type`: the glob below holds
// every JSON file, the private squad included. Display helpers live in format.ts.

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

/** Shown in every page's header and footer, so it holds nothing about Brian's squad. */
export interface Site {
  generated_at: string; season: string; last_gw: number | null; predicted_gw: number | null;
  deadline: { event_id: number; name: string; time: string } | null;
  horizon: number;
}

declare const process: { env: Record<string, string | undefined> }; // Node, at build time

const files = import.meta.glob<unknown>("../data/*.json", { eager: true, import: "default" });

/** One page's data: src/data/<name>.json, else the committed sample. */
export function load<T>(name: string): { data: T; isSample: boolean } {
  const live = files[`../data/${name}.json`];
  if (live !== undefined) return { data: live as T, isSample: false };
  if (process.env.REQUIRE_WEB_DATA) {
    throw new Error(`REQUIRE_WEB_DATA is set but src/data/${name}.json is missing - the export step failed.`);
  }
  const sample = files[`../data/sample-${name}.json`];
  if (sample === undefined) throw new Error(`No src/data/${name}.json and no sample-${name}.json.`);
  return { data: sample as T, isSample: true };
}

export const site = load<Site>("site").data;
