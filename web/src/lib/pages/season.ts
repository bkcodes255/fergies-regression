// Types and chart geometry for the Season page. Pure functions, run at build time.

export interface SeasonGw {
  gw: number; actual: number | null; model: number | null; fpl: number | null; delta: number | null;
  average: number | null; highest: number | null; percentile: number | null;
}
export interface Season {
  generated_at: string; season: string;
  summary: null | {
    total_points: number | null; overall_rank: number | null; average_points: number; bench_points: number;
    gameweeks_played: number; best_gw: { gw: number; points: number };
  };
  gameweeks: SeasonGw[];
  accuracy: null | {
    weeks: number; beats: number; model_error: number; fpl_error: number;
    biggest_miss: { gw: number; predicted: number; actual: number; gap: number };
  };
}

export const W = 640;
export const H = 260;
const PAD = { l: 38, r: 12, t: 12, b: 28 };

export interface Chart {
  yTicks: { y: number; label: string }[];
  xTicks: { x: number; label: string }[];
  /** One path per series; a gap in the data starts a new segment, never a line through zero. */
  paths: Record<"actual" | "model" | "fpl", string>;
  dots: Record<"actual" | "model" | "fpl", { x: number; y: number; gw: number; v: number }[]>;
}

export function buildChart(rows: SeasonGw[]): Chart {
  const keys = ["actual", "model", "fpl"] as const;
  const vals = rows.flatMap((r) => keys.map((k) => r[k]).filter((v): v is number => v !== null));
  const top = Math.max(10, Math.ceil(Math.max(0, ...vals) / 20) * 20);
  const n = rows.length;
  const x = (i: number) => PAD.l + (n <= 1 ? (W - PAD.l - PAD.r) / 2 : (i * (W - PAD.l - PAD.r)) / (n - 1));
  const y = (v: number) => PAD.t + (1 - v / top) * (H - PAD.t - PAD.b);
  const paths = { actual: "", model: "", fpl: "" };
  const dots: Chart["dots"] = { actual: [], model: [], fpl: [] };
  for (const k of keys) {
    let pen = false;
    rows.forEach((r, i) => {
      const v = r[k];
      if (v === null) { pen = false; return; }
      paths[k] += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)} `;
      pen = true;
      dots[k].push({ x: x(i), y: y(v), gw: r.gw, v });
    });
    paths[k] = paths[k].trim();
  }
  const step = top / 4;
  const yTicks = [0, 1, 2, 3, 4].map((i) => ({ y: y(i * step), label: String(Math.round(i * step)) }));
  const every = Math.max(1, Math.ceil(n / 12));
  const xTicks = rows.flatMap((r, i) => (i % every === 0 ? [{ x: x(i), label: `GW${r.gw}` }] : []));
  return { yTicks, xTicks, paths, dots };
}

/** A prediction that wasn't available shows as a dash, never 0. */
export const dash = (v: number | null, digits = 1) => (v === null ? "—" : v.toFixed(digits));
