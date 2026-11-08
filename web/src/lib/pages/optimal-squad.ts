// Optimal Squad: the exported shape and the lookup both the page (at build time) and its browser
// script share. Pure functions only, no data loading, so the script may import this file.

export type HorizonKey = "next" | "window";
export interface SquadEntry { xi: number[]; bench: number[]; cap: number | null; pts: number; spend: number }
export interface SquadPlayer { n: string; team: string; pos: string; price: number; next: number | null; window: number | null }
export interface OptimalSquad {
  generated_at: string; season: string; predicted_gw: number | null; horizon: number;
  budgets: number[];
  solutions: Record<HorizonKey, { squads: SquadEntry[]; by_budget: (number | null)[] }>;
  players: Record<string, SquadPlayer>;
}

export interface Row { name: string; team: string; pos: string; price: number; pts: number | null; cap: boolean }
export interface View {
  budget: number; basis: string;
  found: boolean;
  pts: number; spend: number; unspent: number;
  xi: Row[]; bench: Row[];
}

export const DEFAULT_BUDGET = 100;

/** Index of the budget shown first: 100.0 if it was solved, else the highest budget there is. */
export function defaultIndex(budgets: number[]): number {
  const i = budgets.indexOf(DEFAULT_BUDGET);
  return i >= 0 ? i : Math.max(0, budgets.length - 1);
}

export function basisLabel(key: HorizonKey, horizon: number): string {
  return key === "window" ? `next ${horizon} gameweeks` : "next gameweek";
}

/** The stored answer for one budget (by index into `budgets`) and horizon. found is false when no squad fits. */
export function view(data: OptimalSquad, key: HorizonKey, index: number): View {
  const budget = data.budgets[index];
  const basis = basisLabel(key, data.horizon);
  const sol = data.solutions[key];
  const at = sol.by_budget[index];
  const squad = at === null || at === undefined ? null : sol.squads[at];
  if (!squad) return { budget, basis, found: false, pts: 0, spend: 0, unspent: 0, xi: [], bench: [] };
  const row = (code: number): Row => {
    const p = data.players[String(code)];
    return { name: p.n, team: p.team, pos: p.pos, price: p.price, pts: p[key], cap: code === squad.cap };
  };
  return {
    budget, basis, found: true, pts: squad.pts, spend: squad.spend, unspent: Math.max(0, budget - squad.spend),
    xi: squad.xi.map(row), bench: squad.bench.map(row),
  };
}
