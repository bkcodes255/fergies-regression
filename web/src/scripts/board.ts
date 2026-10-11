// The magnetic tactics board: your XI as magnets showing predicted points, this week's calls
// drawn on in the gaffer's marker. Ported from dashboard/team_talk.py (where it had to live in an
// iframe); here it also opens a player card on tap, and slides the magnets into the lineup the
// calls lead to ("After the talk").
import type { Board, BoardCall, Magnet } from "../lib/data";

type Pt = { x: number; y: number };
interface Layout {
  W: number; H: number; pitch: { W: number; H: number }; portrait: boolean; r: number; fName: number; fPts: number;
  at: (line: number, i: number, n: number) => Pt; bench: (i: number) => Pt; benchLabel: Pt;
}

const NS = "http://www.w3.org/2000/svg";
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, parent: Element): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, String(attrs[k]));
  parent.appendChild(e);
  return e;
}

function layout(portrait: boolean): Layout {
  if (!portrait) {
    const W = 160, H = 100, benchY = 110;
    return { W, H: 118, pitch: { W, H }, portrait, r: 4.4, fName: 2.5, fPts: 2.9,
      at: (l, i, n) => ({ x: [10, 44, 86, 128][l], y: ((i + 0.5) / n) * H }),
      bench: (i) => ({ x: 52 + i * 21, y: benchY }), benchLabel: { x: 34, y: benchY } };
  }
  const W = 100, H = 150, benchY = 162;
  return { W, H: 172, pitch: { W, H }, portrait, r: 5.6, fName: 3.4, fPts: 3.8,
    at: (l, i, n) => ({ x: ((i + 0.5) / n) * W, y: [140, 110, 72, 30][l] }),
    bench: (i) => ({ x: 14 + i * 24, y: benchY }), benchLabel: { x: 3, y: benchY - 7 } };
}

function positions(b: Board, L: Layout) {
  const where: Record<number, Pt & { bench: boolean; cap: boolean; vice: boolean }> = {};
  b.xi.forEach((line, li) => line.forEach((m, i) => { where[m.id] = { ...L.at(li, i, line.length), bench: false, cap: m.cap, vice: m.vice }; }));
  b.bench.forEach((m, i) => { where[m.id] = { ...L.bench(i), bench: true, cap: m.cap, vice: m.vice }; });
  return where;
}

const xiTotal = (b: Board) => {
  const xi = b.xi.flat();
  const cap = xi.find((m) => m.cap);
  return xi.reduce((s, m) => s + (m.p ?? 0), 0) + (cap?.p ?? 0);
};

export function mountBoard(root: HTMLElement, now: Board, after: Board, calls: BoardCall[]) {
  const svg = root.querySelector<SVGSVGElement>("svg")!;
  const frame = root.querySelector<HTMLElement>(".board")!;
  const summary = root.querySelector<HTMLElement>("[data-after-summary]");
  const all = new Map<number, Magnet>([...now.xi.flat(), ...now.bench].map((m) => [m.id, m]));
  let built: boolean | null = null;
  let mode: "now" | "after" = "now";
  let first = true;
  let L: Layout;
  let whereNow: ReturnType<typeof positions>, whereAfter: ReturnType<typeof positions>;
  const mags = new Map<number, SVGGElement>();
  let ink: SVGGElement;

  function stroke(d: string, parent: Element, delay: number, w: number) {
    const p = el("path", { d, fill: "none", stroke: "var(--marker)", "stroke-width": w, "stroke-linecap": "round",
      "stroke-linejoin": "round", class: reduce || !first ? "" : "ink" }, parent);
    p.style.setProperty("--len", String(p.getTotalLength()));
    p.style.animationDelay = `${delay}s`;
    return p;
  }

  function arrow(a: Pt, b: Pt, bend: number, parent: Element, delay: number, w: number, head: number, ctrl?: Pt): Pt {
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2, dx = b.x - a.x, dy = b.y - a.y;
    const cx = ctrl ? ctrl.x : mx - dy * bend, cy = ctrl ? ctrl.y : my + dx * bend;
    stroke(`M${a.x},${a.y} Q${cx},${cy} ${b.x},${b.y}`, parent, delay, w);
    const ang = Math.atan2(b.y - cy, b.x - cx);
    const p1 = { x: b.x - head * Math.cos(ang - 0.45), y: b.y - head * Math.sin(ang - 0.45) };
    const p2 = { x: b.x - head * Math.cos(ang + 0.45), y: b.y - head * Math.sin(ang + 0.45) };
    stroke(`M${p1.x},${p1.y} L${b.x},${b.y} L${p2.x},${p2.y}`, parent, delay + 0.55, w);
    return { x: (a.x + 2 * cx + b.x) / 4, y: (a.y + 2 * cy + b.y) / 4 };
  }

  function badge(n: string, c: Pt, parent: Element, delay: number) {
    const r = L.r * 0.72;
    stroke(`M${c.x + r},${c.y} A${r},${r} 0 1 1 ${c.x + r - 0.01},${c.y - 0.6}`, parent, delay, L.portrait ? 0.55 : 0.45);
    const t = el("text", { x: c.x, y: c.y + r * 0.5, "text-anchor": "middle", "font-family": "Permanent Marker, cursive",
      "font-size": r * 1.35, fill: "var(--marker)", class: reduce || !first ? "" : "fade", style: `animation-delay:${delay + 0.3}s` }, parent);
    t.textContent = n;
  }

  function drawPitch() {
    const { W, H } = L.pitch;
    const s = { fill: "none", stroke: "var(--print)", "stroke-width": L.portrait ? 0.5 : 0.4 };
    el("rect", { x: 1, y: 1, width: W - 2, height: H - 2, ...s }, svg);
    if (!L.portrait) {
      el("line", { x1: W / 2, y1: 1, x2: W / 2, y2: H - 1, ...s }, svg); el("circle", { cx: W / 2, cy: H / 2, r: 10, ...s }, svg);
      el("rect", { x: 1, y: H / 2 - 22, width: 17, height: 44, ...s }, svg); el("rect", { x: W - 18, y: H / 2 - 22, width: 17, height: 44, ...s }, svg);
      el("rect", { x: 1, y: H / 2 - 10, width: 6, height: 20, ...s }, svg); el("rect", { x: W - 7, y: H / 2 - 10, width: 6, height: 20, ...s }, svg);
    } else {
      el("line", { x1: 1, y1: H / 2, x2: W - 1, y2: H / 2, ...s }, svg); el("circle", { cx: W / 2, cy: H / 2, r: 10, ...s }, svg);
      el("rect", { x: W / 2 - 24, y: 1, width: 48, height: 17, ...s }, svg); el("rect", { x: W / 2 - 24, y: H - 18, width: 48, height: 17, ...s }, svg);
      el("rect", { x: W / 2 - 11, y: 1, width: 22, height: 6, ...s }, svg); el("rect", { x: W / 2 - 11, y: H - 7, width: 22, height: 6, ...s }, svg);
    }
    const bl = el("text", { x: L.benchLabel.x, y: L.benchLabel.y + 1, "font-family": "Big Shoulders Display, sans-serif",
      "font-weight": 800, "font-size": L.fName * 1.1, fill: "var(--muted)" }, svg);
    bl.textContent = "BENCH";
  }

  function magnet(m: Magnet, delay: number) {
    const r = L.r;
    const g = el("g", { class: "mag", tabindex: 0, role: "button", "data-id": m.id }, svg);
    const drop = el("g", { class: reduce || !first ? "" : "drop", style: `animation-delay:${delay}s` }, g);
    el("circle", { cx: 0, cy: 0.5, r, fill: "rgba(0,0,0,.35)" }, drop);
    el("circle", { class: "disc", cx: 0, cy: 0, r, stroke: "rgba(255,255,255,.55)", "stroke-width": 0.35 }, drop);
    const t = el("text", { x: 0, y: L.fPts * 0.36, "text-anchor": "middle", "font-family": "Big Shoulders Display, Arial Narrow, sans-serif",
      "font-weight": 900, "font-size": L.fPts, fill: "#fff" }, drop);
    t.textContent = m.p === null ? "–" : m.p.toFixed(1);
    const ty = r + L.fName * 1.05, tw = m.n.length * L.fName * 0.5 + 2.4;
    el("rect", { x: -tw / 2, y: ty - L.fName * 0.85, width: tw, height: L.fName * 1.2, fill: "#f2efe2", rx: 0.3 }, drop);
    const nm = el("text", { x: 0, y: ty, "text-anchor": "middle", "font-family": "Barlow, system-ui, sans-serif", "font-weight": 700,
      "font-size": L.fName * 0.85, fill: "#1a1a12" }, drop);
    nm.textContent = m.n;
    const cb = el("g", { class: "armband", transform: `translate(${r * 0.8} ${-r * 0.8})` }, drop);
    el("circle", { cx: 0, cy: 0, r: r * 0.42 }, cb);
    el("text", { x: 0, y: r * 0.16, "text-anchor": "middle", "font-family": "Barlow, sans-serif", "font-weight": 800, "font-size": r * 0.5, fill: "#14402a" }, cb);
    // A hit target bigger than the disc, covering the name tape too.
    el("rect", { x: -Math.max(r, tw / 2), y: -r, width: Math.max(2 * r, tw), height: r + ty + L.fName * 0.4, fill: "transparent" }, g);
    mags.set(m.id, g);
  }

  function place() {
    const where = mode === "now" ? whereNow : whereAfter;
    for (const [id, g] of mags) {
      const w = where[id], m = all.get(id)!;
      g.style.transform = `translate(${w.x}px, ${w.y}px)`;
      g.classList.toggle("on-bench", w.bench);
      const band = g.querySelector<SVGGElement>(".armband")!;
      band.style.display = w.cap || w.vice ? "" : "none";
      band.querySelector("circle")!.setAttribute("fill", w.cap ? "#f2f5f0" : "#a9b8ae");
      band.querySelector("text")!.textContent = w.cap ? "C" : "V";
      const role = w.bench ? "on the bench" : w.cap ? "captain" : w.vice ? "vice-captain" : "starting";
      g.setAttribute("aria-label", `${m.full}, ${m.p === null ? "no prediction" : m.p.toFixed(1) + " predicted points"}, ${role}. Open player card`);
    }
    ink.classList.toggle("gone", mode === "after");
    if (summary) summary.hidden = mode === "now";
  }

  function drawCalls() {
    const r = L.r, where = whereNow;
    const w = L.portrait ? 0.7 : 0.55, head = L.portrait ? 2.6 : 2.1;
    let t = reduce || !first ? 0 : 1.2;
    for (const c of calls) {
      const g = el("g", {}, ink);
      if (c.kind === "captain" && where[c.from] && where[c.to] && c.from !== c.to) {
        const a = where[c.from], b = where[c.to];
        const mid = arrow({ x: a.x + r * 0.9, y: a.y - r * 1.3 }, { x: b.x - r * 1.1, y: b.y - r * 0.6 }, L.portrait ? -0.25 : 0.25, g, t + 0.2, w, head);
        badge(String(c.n), { x: clamp(mid.x, r, L.W - r), y: clamp(mid.y, r, L.pitch.H - r) }, g, t + 0.4);
        t += 1.3;
      } else if (c.kind === "lineup") {
        c.pairs.forEach((p, k) => {
          const off = where[p.off], on = where[p.on];
          if (!off || !on) return;
          stroke(`M${off.x - r * 1.05},${off.y - r * 1.05} L${off.x + r * 1.05},${off.y + r * 1.05}`, g, t, w);
          stroke(`M${off.x + r * 1.05},${off.y - r * 1.05} L${off.x - r * 1.05},${off.y + r * 1.05}`, g, t + 0.15, w);
          if (L.portrait) arrow({ x: on.x + r * 0.9, y: on.y - r * 1.1 }, { x: off.x + r * 1.5, y: off.y + r * 2.6 }, 0, g, t + 0.3, w, head, { x: L.W + 4, y: (on.y + off.y) / 2 + 30 });
          else arrow({ x: on.x + r * 1.2, y: on.y - r * 1.2 }, { x: off.x - r * 1.3, y: off.y + r * 2.4 }, 0.1, g, t + 0.3, w, head);
          if (k === 0) badge(String(c.n), { x: clamp(off.x + r * 2.6, r, L.W - r), y: clamp(off.y + r * 1.6, r, L.pitch.H - r) }, g, t + 0.5);
          t += 0.9;
        });
        t += 0.4;
      } else if (c.kind === "transfer") {
        const lines = [`${c.n} · SELL ${c.sell_name.toUpperCase()}`, `BUY ${c.buy_name.toUpperCase()}`,
          `${c.net >= 0 ? "+" : ""}${c.net.toFixed(1)} · ${c.hit ? "-" + c.hit + " hit" : "no hit"}`];
        const fs = L.portrait ? 3.3 : 2.7, nh = fs * 1.35 * 3 + fs * 0.9;
        const nw = Math.max(26, Math.max(...lines.map((s) => s.length)) * fs * 0.62 + 4);
        const anchor = c.sell !== null && where[c.sell] ? where[c.sell] : { x: L.W - nw / 2 - 3, y: nh + 6 };
        // Scan the pitch for the spot nearest the player being sold that covers no magnet.
        const covers = (x: number, y: number) => Object.values(where).filter((m) =>
          m.x + r * 1.6 > x && m.x - r * 1.6 < x + nw && m.y + r * 2.2 > y && m.y - r < y + nh).length;
        let best: [number, number, number] | null = null;
        for (let y = 2; y <= L.pitch.H - nh - 2; y += 2) {
          for (let x = 2; x <= L.W - nw - 2; x += 2) {
            const score = covers(x, y) * 1e6 + Math.hypot(x + nw / 2 - anchor.x, y + nh / 2 - anchor.y);
            if (best === null || score < best[2]) best = [x, y, score];
          }
        }
        const [nx, ny] = best!;
        const note = el("g", { class: reduce || !first ? "" : "fade", style: `animation-delay:${t}s`,
          transform: `rotate(${L.portrait ? 3 : -3} ${nx + nw / 2} ${ny + nh / 2})` }, g);
        el("rect", { x: nx + 0.6, y: ny + 0.8, width: nw, height: nh, fill: "rgba(0,0,0,.35)" }, note);
        el("rect", { x: nx, y: ny, width: nw, height: nh, fill: "var(--marker)" }, note);
        el("rect", { x: nx + nw / 2 - 5, y: ny - 1.2, width: 10, height: 3, fill: "rgba(240,240,230,.75)" }, note);
        lines.forEach((s, i) => {
          const tt = el("text", { x: nx + 2.2, y: ny + fs * 1.5 + i * fs * 1.35, "font-family": "Permanent Marker, cursive",
            "font-size": fs, fill: "#1a1a12" }, note);
          tt.textContent = s;
        });
      }
    }
  }

  function build() {
    const portrait = frame.clientWidth < 620;
    if (built === portrait) return;
    built = portrait;
    svg.replaceChildren();
    mags.clear();
    L = layout(portrait);
    svg.setAttribute("viewBox", `0 0 ${L.W} ${L.H}`);
    whereNow = positions(now, L);
    whereAfter = positions(after, L);
    drawPitch();
    let d = reduce || !first ? 0 : 0.1;
    // Bench first, so a magnet sliding up from the bench passes under the XI, not over it.
    for (const m of [...now.bench, ...now.xi.flat()]) { magnet(m, d); d += 0.05; }
    ink = el("g", { class: "calls" }, svg);
    drawCalls();
    svg.classList.add("still");
    place();
    requestAnimationFrame(() => requestAnimationFrame(() => svg.classList.remove("still")));
    first = false;
  }

  svg.addEventListener("click", (e) => {
    const g = (e.target as Element).closest<SVGGElement>(".mag");
    if (g) root.dispatchEvent(new CustomEvent("player:open", { bubbles: true, detail: { id: Number(g.dataset.id) } }));
  });
  svg.addEventListener("keydown", (e) => {
    const g = (e.target as Element).closest<SVGGElement>(".mag");
    if (g && (e.key === "Enter" || e.key === " ")) {
      e.preventDefault();
      root.dispatchEvent(new CustomEvent("player:open", { bubbles: true, detail: { id: Number(g.dataset.id) } }));
    }
  });
  root.querySelectorAll<HTMLInputElement>('input[name="board-view"]').forEach((i) =>
    i.addEventListener("change", () => { mode = i.value === "after" ? "after" : "now"; place(); }));

  if (summary) {
    const before = xiTotal(now), next = xiTotal(after), diff = next - before;
    summary.textContent = `After the talk: ${next.toFixed(1)} predicted points from this XI, ` +
      `${diff >= 0 ? "+" : "−"}${Math.abs(diff).toFixed(1)} on the lineup you have now.` +
      (calls.some((c) => c.kind === "transfer") ? " The transfer comes on top." : "");
  }
  build();
  new ResizeObserver(() => build()).observe(frame);
}
