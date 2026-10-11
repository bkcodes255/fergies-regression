"""The Team Talk look, built into the app: the broadcast header bar that tops every tab, and the
magnetic tactics board that replaces the old pitch - your XI as magnets showing predicted
points, with this week's calls drawn on in the gaffer's marker.

Ported from the design-lab mockup (Fergie's Team Talk), generalised to any squad and formation.
Both render inside st.iframe tiles (see dashboard/app.py show_tile), so each carries its own
fonts and styles.
"""
from __future__ import annotations

import html
import json

import pandas as pd

# Team Talk palette - dashboard/app.py's tokens and .streamlit/config.toml use the same values.
STUDIO = "#0B0F0D"      # page ground
SURFACE = "#111A15"     # panels
LINE = "#24352B"
INK = "#F2F5F0"
MUTED = "#A9B8AE"
BOARD = "#14402A"
BOARD_EDGE = "#0E2D1E"
DISC = "#D7263D"        # your team's magnets
MARKER = "#FFE14D"      # the gaffer's marker
BAR = "#F4F4EF"         # broadcast lower-third
GOOD = "#7BE495"

FONTS = (
    '<link rel="preload" as="style" '
    'href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&'
    'family=Big+Shoulders+Display:wght@700;800;900&family=Permanent+Marker&display=swap" '
    'onload="this.onload=null;this.rel=\'stylesheet\'">'
)
LINE_OF = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}


def _short(name: str, limit: int = 11) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "…"


def render_broadcast_bar_html(
    season: str, deadline: dict | None, last_gw: int | None, predicted_gw: int | None, deadline_label: str | None,
) -> str:
    """The TV-style bar at the top of every tab: gameweek badge, title, data freshness, and a
    deadline clock that keeps counting down while the page is open."""
    gw = f"GW{deadline['event_id']}" if deadline else "—"
    meta = [season]
    if predicted_gw:
        meta.append(f"predictions for GW{predicted_gw}")
    if last_gw:
        meta.append(f"data to GW{last_gw}")
    deadline_ms = int(pd.Timestamp(deadline["deadline_time"]).timestamp() * 1000) if deadline else None
    clock_label = f"Deadline · {deadline_label}" if deadline_label else "No upcoming deadline"
    # Rendered straight into the page with st.html, not an iframe: an iframe can't size itself
    # below 150px, which left a gap under a 60px bar. So the CSS is scoped under .ttbar, and the
    # theme already loads both fonts.
    return f"""
    <style>
      .ttbar {{ display: flex; flex-wrap: wrap; align-items: stretch; background: {BAR}; color: {STUDIO};
               font-family: 'Barlow', system-ui, sans-serif; font-variant-numeric: tabular-nums; }}
      .ttbar * {{ box-sizing: border-box; margin: 0; }}
      .ttbar .gw {{ background: {DISC}; color: #fff; font: 900 30px/1 'Big Shoulders Display', 'Arial Narrow', sans-serif;
             padding: 12px 16px; display: grid; place-items: center; letter-spacing: .02em; }}
      .ttbar .title {{ padding: 8px 16px; display: grid; align-content: center; flex: 1 1 240px; min-width: 0; }}
      .ttbar .title b {{ font: 900 24px/1.05 'Big Shoulders Display', 'Arial Narrow', sans-serif; text-transform: uppercase; letter-spacing: .02em; }}
      .ttbar .title span {{ font-size: 12.5px; font-weight: 600; color: #4b544e; }}
      .ttbar .clock {{ background: {BOARD}; color: {BAR}; padding: 8px 18px; display: grid; align-content: center; justify-items: end; }}
      .ttbar .clock small {{ font-size: 11px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: {MARKER}; }}
      .ttbar .clock b {{ font: 800 30px/1 'Big Shoulders Display', 'Arial Narrow', sans-serif; letter-spacing: .02em; }}
      @media (max-width: 560px) {{
        .ttbar .clock {{ flex: 1 1 100%; justify-items: start; }}
        .ttbar .gw {{ font-size: 24px; padding: 10px 12px; }}
        .ttbar .title {{ padding: 7px 12px; }}
        .ttbar .title b {{ font-size: 19px; }}
      }}
    </style>
    <div class="ttbar" role="banner">
      <div class="gw">{html.escape(gw)}</div>
      <div class="title"><b>Fergie's Regression · Team talk</b><span>{html.escape(' · '.join(meta))}</span></div>
      <div class="clock" data-deadline="{'' if deadline_ms is None else deadline_ms}"><small>{html.escape(clock_label)}</small><b>—</b></div>
    </div>
    <script>
      // One page-wide timer: st.html re-runs this script on every Streamlit rerun, and the bar
      // itself may be replaced, so each tick looks the clock up again.
      if (!window.__ttClock) {{
        const tick = () => document.querySelectorAll(".ttbar .clock").forEach(c => {{
          const out = c.querySelector("b"), deadline = c.dataset.deadline ? +c.dataset.deadline : null;
          if (deadline === null) {{ out.textContent = "—"; return; }}
          const s = Math.floor((deadline - Date.now()) / 1000);
          if (s <= 0) {{ out.textContent = "DEADLINE PASSED"; return; }}
          const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60);
          out.textContent = d + "D " + String(h).padStart(2, "0") + "H " + String(m).padStart(2, "0") + "M";
        }});
        window.__ttClock = setInterval(tick, 30000);
        window.__ttTick = tick;
      }}
      window.__ttTick();
    </script>
    """


def board_data(starting: pd.DataFrame, bench: pd.DataFrame, value_col: str = "predicted_points") -> dict:
    """Magnets by line (GKP, DEF, MID, FWD) plus the bench, keyed by player_code."""
    def magnet(r: pd.Series) -> dict:
        v = r.get(value_col)
        return {"id": int(r["player_code"]), "n": _short(str(r["web_name"])), "full": str(r["web_name"]),
                "p": None if v is None or pd.isna(v) else round(float(v), 1),
                "cap": bool(r.get("is_captain", False)), "vice": bool(r.get("is_vice_captain", False))}
    lines: list[list[dict]] = [[], [], [], []]
    for _, r in starting.iterrows():
        lines[LINE_OF.get(r["position"], 2)].append(magnet(r))
    return {"xi": lines, "bench": [magnet(r) for _, r in bench.iterrows()]}


def board_calls(cap: dict | None, changes: dict | None, squad: pd.DataFrame, plan: pd.DataFrame | None) -> list[dict]:
    """The outstanding calls to draw, numbered in the hairdryer's order (captain, lineup, transfer)."""
    calls: list[dict] = []
    if cap is not None and not cap["matches"]:
        current = squad.loc[squad["is_captain"].astype(bool), "player_code"]
        if not current.empty:
            calls.append({"kind": "captain", "from": int(current.iloc[0]), "to": int(cap["recommended_code"]),
                          "label": f"Armband to {cap['recommended_name']}"})
    if changes and (changes["bring_in_codes"] or changes["bench_out_codes"]):
        pos = dict(zip(squad["player_code"], squad["position"]))
        names = dict(zip(squad["player_code"], squad["web_name"]))
        ons, offs = sorted(changes["bring_in_codes"]), sorted(changes["bench_out_codes"])
        pairs, left = [], list(ons)
        for off in offs:  # pair each benched starter with an incoming player of the same position if there is one
            same = next((o for o in left if pos.get(o) == pos.get(off)), left[0] if left else None)
            if same is not None:
                left.remove(same)
                pairs.append({"on": int(same), "off": int(off)})
        calls.append({"kind": "lineup", "pairs": pairs,
                      "label": ", ".join(names[p["on"]] for p in pairs) + " on, " + ", ".join(names[p["off"]] for p in pairs) + " off"})
    if plan is not None and not plan.empty:
        top = plan.iloc[0]
        sell = squad.loc[squad["web_name"] == top["sell"], "player_code"]
        extra = f" (+{len(plan) - 1} more in the plan)" if len(plan) > 1 else ""
        calls.append({"kind": "transfer", "sell": int(sell.iloc[0]) if not sell.empty else None,
                      "sell_name": str(top["sell"]), "buy_name": str(top["buy"]),
                      "net": round(float(top["net"]), 1), "hit": int(top["hit"]),
                      "label": f"Sell {top['sell']}, buy {top['buy']}{extra}"})
    for i, c in enumerate(calls, start=1):
        c["n"] = i
    return calls


def render_board_html(data: dict, calls: list[dict], caption: str,
                      calm: str | None = "Nothing to change: the model is happy with this XI.") -> str:
    """The magnetic tactics board. Landscape on wide screens (team attacks right), portrait on
    phones; magnets drop in, then the marker draws each call in order. calm is the line shown
    when there are no calls (None for a board that never has any, like the solver's XI)."""
    legend = "".join(
        f'<li><span class="n">{c["n"]}</span>{html.escape(c["label"])}</li>' for c in calls
    ) or (f'<li class="calm">{html.escape(calm)}</li>' if calm else "")
    desc = "Tactics board. " + "; ".join(
        f"{m['full']} {m['p']}" + (" captain" if m["cap"] else "") for line in data["xi"] for m in line
    ) + ". " + " ".join(f"{c['n']}: {c['label']}." for c in calls)
    return f"""
    {FONTS}
    <style>
      :root {{ --print: rgba(236,244,238,.30); --disc: {DISC}; --disc-ink: #fff; --marker: {MARKER};
               --note: {MARKER}; --note-ink: #1a1a12; --muted: {MUTED}; }}
      * {{ box-sizing: border-box; margin: 0; }}
      body {{ background: transparent; color: {INK}; font-family: 'Barlow', system-ui, sans-serif; }}
      .board {{ background: radial-gradient(120% 90% at 50% 40%, #185034 0%, {BOARD} 55%, {BOARD_EDGE} 100%);
                border-radius: 10px; padding: 10px; box-shadow: inset 0 0 0 6px #1d2a22, inset 0 0 0 7px #34463b; }}
      svg {{ display: block; width: 100%; height: auto; }}
      .cap {{ font-size: 12px; color: {MUTED}; margin: 8px 2px 0; }}
      ol {{ list-style: none; padding: 0; margin: 10px 0 0; display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 14px; font-weight: 600; }}
      li {{ display: flex; align-items: center; gap: 8px; }}
      li.calm {{ color: {GOOD}; }}
      .n {{ font-family: 'Permanent Marker', cursive; color: {MARKER}; width: 24px; height: 24px; border: 2px solid {MARKER};
            border-radius: 50%; display: grid; place-items: center; font-size: 14px; line-height: 1; flex: 0 0 auto; }}
      .disc {{ transform-box: fill-box; transform-origin: center; animation: drop .5s cubic-bezier(.2,.9,.3,1.3) both; }}
      .ink {{ stroke-dasharray: var(--len); stroke-dashoffset: var(--len); animation: draw .7s ease-out forwards; }}
      .fade {{ opacity: 0; animation: fade .4s ease-out forwards; }}
      @keyframes drop {{ from {{ opacity: 0; transform: translateY(-6px) scale(1.25); }} to {{ opacity: 1; transform: none; }} }}
      @keyframes draw {{ to {{ stroke-dashoffset: 0; }} }}
      @keyframes fade {{ to {{ opacity: 1; }} }}
      @media (prefers-reduced-motion: reduce) {{ .disc, .ink, .fade {{ animation: none !important; opacity: 1; stroke-dashoffset: 0; }} }}
    </style>
    <div class="board" id="board"><svg id="pitch" role="img" aria-label="{html.escape(desc)}"></svg></div>
    <ol>{legend}</ol>
    <p class="cap">{html.escape(caption)}</p>
    <script>
    const DATA = {json.dumps(data).replace("</", "<\\/")};
    const CALLS = {json.dumps(calls).replace("</", "<\\/")};
    const NS = "http://www.w3.org/2000/svg", svg = document.getElementById("pitch");
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let built = null;
    function el(tag, attrs, parent) {{
      const e = document.createElementNS(NS, tag);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      (parent || svg).appendChild(e); return e;
    }}
    function layout(portrait) {{
      if (!portrait) {{
        const W = 160, H = 100, benchY = 110;
        return {{ W, H: 118, pitch: {{ W, H }}, portrait, at: (l, i, n) => ({{ x: [10, 44, 86, 128][l], y: (i + .5) / n * H }}),
                 bench: i => ({{ x: 52 + i * 21, y: benchY }}), benchLabel: {{ x: 34, y: benchY }}, r: 4.4, fName: 2.5, fPts: 2.9 }};
      }}
      const W = 100, H = 150, benchY = 162;
      return {{ W, H: 172, pitch: {{ W, H }}, portrait, at: (l, i, n) => ({{ x: (i + .5) / n * W, y: [140, 110, 72, 30][l] }}),
               bench: i => ({{ x: 14 + i * 24, y: benchY }}), benchLabel: {{ x: 3, y: benchY - 7 }}, r: 5.6, fName: 3.4, fPts: 3.8 }};
    }}
    function drawPitch(L) {{
      const {{ W, H }} = L.pitch, s = {{ fill: "none", stroke: "var(--print)", "stroke-width": L.portrait ? .5 : .4 }};
      el("rect", {{ x: 1, y: 1, width: W - 2, height: H - 2, ...s }});
      if (!L.portrait) {{
        el("line", {{ x1: W / 2, y1: 1, x2: W / 2, y2: H - 1, ...s }}); el("circle", {{ cx: W / 2, cy: H / 2, r: 10, ...s }});
        el("rect", {{ x: 1, y: H / 2 - 22, width: 17, height: 44, ...s }}); el("rect", {{ x: W - 18, y: H / 2 - 22, width: 17, height: 44, ...s }});
        el("rect", {{ x: 1, y: H / 2 - 10, width: 6, height: 20, ...s }}); el("rect", {{ x: W - 7, y: H / 2 - 10, width: 6, height: 20, ...s }});
      }} else {{
        el("line", {{ x1: 1, y1: H / 2, x2: W - 1, y2: H / 2, ...s }}); el("circle", {{ cx: W / 2, cy: H / 2, r: 10, ...s }});
        el("rect", {{ x: W / 2 - 24, y: 1, width: 48, height: 17, ...s }}); el("rect", {{ x: W / 2 - 24, y: H - 18, width: 48, height: 17, ...s }});
        el("rect", {{ x: W / 2 - 11, y: 1, width: 22, height: 6, ...s }}); el("rect", {{ x: W / 2 - 11, y: H - 7, width: 22, height: 6, ...s }});
      }}
    }}
    function disc(pl, c, L, delay, bench) {{
      const g = el("g", {{ class: "disc", style: `animation-delay:${{delay}}s` }});
      el("circle", {{ cx: c.x, cy: c.y + .5, r: L.r, fill: "rgba(0,0,0,.35)" }}, g);
      el("circle", {{ cx: c.x, cy: c.y, r: L.r, fill: bench ? "#2a3a31" : "var(--disc)", stroke: "rgba(255,255,255,.55)", "stroke-width": .35 }}, g);
      const t = el("text", {{ x: c.x, y: c.y + L.fPts * .36, "text-anchor": "middle", "font-family": "Big Shoulders Display, Arial Narrow, sans-serif", "font-weight": 900, "font-size": L.fPts, fill: "var(--disc-ink)" }}, g);
      t.textContent = pl.p === null ? "–" : pl.p.toFixed(1);
      const ty = c.y + L.r + L.fName * 1.05, tw = pl.n.length * L.fName * .5 + 2.4;
      el("rect", {{ x: c.x - tw / 2, y: ty - L.fName * .85, width: tw, height: L.fName * 1.2, fill: "#f2efe2", rx: .3 }}, g);
      const nm = el("text", {{ x: c.x, y: ty, "text-anchor": "middle", "font-family": "Barlow, system-ui, sans-serif", "font-weight": 700, "font-size": L.fName * .85, fill: "#1a1a12" }}, g);
      nm.textContent = pl.n;
      if (pl.cap || pl.vice) {{
        const a = {{ x: c.x + L.r * .8, y: c.y - L.r * .8 }};
        el("circle", {{ cx: a.x, cy: a.y, r: L.r * .42, fill: pl.cap ? "#f2f5f0" : "#a9b8ae" }}, g);
        const ct = el("text", {{ x: a.x, y: a.y + L.r * .16, "text-anchor": "middle", "font-family": "Barlow, sans-serif", "font-weight": 800, "font-size": L.r * .5, fill: "#14402a" }}, g);
        ct.textContent = pl.cap ? "C" : "V";
      }}
    }}
    function stroke(d, parent, delay, w) {{
      const p = el("path", {{ d, fill: "none", stroke: "var(--marker)", "stroke-width": w, "stroke-linecap": "round", "stroke-linejoin": "round", class: reduce ? "" : "ink" }}, parent);
      p.style.setProperty("--len", p.getTotalLength()); p.style.animationDelay = delay + "s"; return p;
    }}
    function arrow(a, b, bend, parent, delay, w, head, ctrl) {{
      const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2, dx = b.x - a.x, dy = b.y - a.y;
      const cx = ctrl ? ctrl.x : mx - dy * bend, cy = ctrl ? ctrl.y : my + dx * bend;
      stroke(`M${{a.x}},${{a.y}} Q${{cx}},${{cy}} ${{b.x}},${{b.y}}`, parent, delay, w);
      const ang = Math.atan2(b.y - cy, b.x - cx);
      const p1 = {{ x: b.x - head * Math.cos(ang - .45), y: b.y - head * Math.sin(ang - .45) }};
      const p2 = {{ x: b.x - head * Math.cos(ang + .45), y: b.y - head * Math.sin(ang + .45) }};
      stroke(`M${{p1.x}},${{p1.y}} L${{b.x}},${{b.y}} L${{p2.x}},${{p2.y}}`, parent, delay + .55, w);
      return {{ x: (a.x + 2 * cx + b.x) / 4, y: (a.y + 2 * cy + b.y) / 4 }};
    }}
    function badge(n, c, parent, delay, L) {{
      const r = L.r * .72;
      stroke(`M${{c.x + r}},${{c.y}} A${{r}},${{r}} 0 1 1 ${{c.x + r - .01}},${{c.y - .6}}`, parent, delay, L.portrait ? .55 : .45);
      const t = el("text", {{ x: c.x, y: c.y + r * .5, "text-anchor": "middle", "font-family": "Permanent Marker, cursive", "font-size": r * 1.35, fill: "var(--marker)", class: reduce ? "" : "fade", style: `animation-delay:${{delay + .3}}s` }}, parent);
      t.textContent = n;
    }}
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    function build() {{
      const portrait = document.getElementById("board").clientWidth < 620;
      if (built === portrait) return; built = portrait;
      while (svg.firstChild) svg.firstChild.remove();
      const L = layout(portrait), r = L.r;
      svg.setAttribute("viewBox", `0 0 ${{L.W}} ${{L.H}}`);
      drawPitch(L);
      const where = {{}};
      DATA.xi.forEach((line, li) => line.forEach((pl, i) => {{ where[pl.id] = L.at(li, i, line.length); }}));
      DATA.bench.forEach((pl, i) => {{ where[pl.id] = L.bench(i); }});
      if (DATA.bench.length) {{
        const bl = el("text", {{ x: L.benchLabel.x, y: L.benchLabel.y + 1, "font-family": "Big Shoulders Display, sans-serif", "font-weight": 800, "font-size": L.fName * 1.1, fill: "var(--muted)" }});
        bl.textContent = "BENCH";
      }}
      let d = reduce ? 0 : .1;
      DATA.xi.forEach(line => line.forEach(pl => {{ disc(pl, where[pl.id], L, d, false); d += .05; }}));
      DATA.bench.forEach(pl => {{ disc(pl, where[pl.id], L, d, true); d += .05; }});
      const w = portrait ? .7 : .55, head = portrait ? 2.6 : 2.1;
      let t = reduce ? 0 : 1.2;
      for (const c of CALLS) {{
        const g = el("g", {{}});
        if (c.kind === "captain" && where[c.from] && where[c.to] && c.from !== c.to) {{
          const a = where[c.from], b = where[c.to];
          const mid = arrow({{ x: a.x + r * .9, y: a.y - r * 1.3 }}, {{ x: b.x - r * 1.1, y: b.y - r * .6 }}, portrait ? -.25 : .25, g, t + .2, w, head);
          badge(String(c.n), {{ x: clamp(mid.x, r, L.W - r), y: clamp(mid.y, r, L.pitch.H - r) }}, g, t + .4, L);
          t += 1.3;
        }} else if (c.kind === "lineup") {{
          c.pairs.forEach((p, k) => {{
            const off = where[p.off], on = where[p.on];
            if (!off || !on) return;
            stroke(`M${{off.x - r * 1.05}},${{off.y - r * 1.05}} L${{off.x + r * 1.05}},${{off.y + r * 1.05}}`, g, t, w);
            stroke(`M${{off.x + r * 1.05}},${{off.y - r * 1.05}} L${{off.x - r * 1.05}},${{off.y + r * 1.05}}`, g, t + .15, w);
            if (portrait) arrow({{ x: on.x + r * .9, y: on.y - r * 1.1 }}, {{ x: off.x + r * 1.5, y: off.y + r * 2.6 }}, 0, g, t + .3, w, head, {{ x: L.W + 4, y: (on.y + off.y) / 2 + 30 }});
            else arrow({{ x: on.x + r * 1.2, y: on.y - r * 1.2 }}, {{ x: off.x - r * 1.3, y: off.y + r * 2.4 }}, .1, g, t + .3, w, head);
            if (k === 0) badge(String(c.n), {{ x: clamp(off.x + r * 2.6, r, L.W - r), y: clamp(off.y + r * 1.6, r, L.pitch.H - r) }}, g, t + .5, L);
            t += .9;
          }});
          t += .4;
        }} else if (c.kind === "transfer") {{
          const lines = [`${{c.n}} · SELL ${{c.sell_name.toUpperCase()}}`, `BUY ${{c.buy_name.toUpperCase()}}`, `${{c.net >= 0 ? "+" : ""}}${{c.net.toFixed(1)}} · ${{c.hit ? "-" + c.hit + " hit" : "no hit"}}`];
          const fs = portrait ? 3.3 : 2.7, nh = fs * 1.35 * 3 + fs * .9;
          const nw = Math.max(26, Math.max(...lines.map(s => s.length)) * fs * .62 + 4);
          const anchor = c.sell !== null && where[c.sell] ? where[c.sell] : {{ x: L.W - nw / 2 - 3, y: nh + 6 }};
          // Scan the pitch for spots the note can sit without covering a magnet, and take the one
          // nearest the player being sold - fixed offsets (above, below, ...) always hit someone in a crowded line.
          const covers = (x, y) => Object.values(where).filter(m =>
            m.x + r * 1.6 > x && m.x - r * 1.6 < x + nw && m.y + r * 2.2 > y && m.y - r < y + nh).length;
          let best = null;
          for (let y = 2; y <= L.pitch.H - nh - 2; y += 2) {{
            for (let x = 2; x <= L.W - nw - 2; x += 2) {{
              const score = covers(x, y) * 1e6 + Math.hypot(x + nw / 2 - anchor.x, y + nh / 2 - anchor.y);
              if (best === null || score < best[2]) best = [x, y, score];
            }}
          }}
          const [nx, ny] = best;
          const note = el("g", {{ class: reduce ? "" : "fade", style: `animation-delay:${{t}}s`, transform: `rotate(${{portrait ? 3 : -3}} ${{nx + nw / 2}} ${{ny + nh / 2}})` }}, g);
          el("rect", {{ x: nx + .6, y: ny + .8, width: nw, height: nh, fill: "rgba(0,0,0,.35)" }}, note);
          el("rect", {{ x: nx, y: ny, width: nw, height: nh, fill: "var(--note)" }}, note);
          el("rect", {{ x: nx + nw / 2 - 5, y: ny - 1.2, width: 10, height: 3, fill: "rgba(240,240,230,.75)" }}, note);
          lines.forEach((s, i) => {{
            const tt = el("text", {{ x: nx + 2.2, y: ny + fs * 1.5 + i * fs * 1.35, "font-family": "Permanent Marker, cursive", "font-size": fs, fill: "var(--note-ink)" }}, note);
            tt.textContent = s;
          }});
        }}
      }}
    }}
    build();
    let rt; addEventListener("resize", () => {{ clearTimeout(rt); rt = setTimeout(build, 120); }});
    </script>
    """
