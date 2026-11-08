// The Players page's data (src/web_data/players_page.py). Types only: safe to import anywhere.
export interface PlayerRow {
  name: string; pos: string; team: string; price: number | null; pred: number | null; horizon: number | null;
  floor: number | null; ceil: number | null; haul: number | null; value: number | null; pts: number | null; owned: number | null;
}
export interface PlayersPage { predicted_gw: number | null; horizon: number; players: PlayerRow[] }

/** A number for a table cell, or a dash when there is none. */
export const cell = (v: number | null, digits = 1) => (v === null ? "—" : v.toFixed(digits));
export const range = (p: PlayerRow) => (p.floor === null || p.ceil === null ? "—" : `${p.floor.toFixed(1)}–${p.ceil.toFixed(1)}`);
