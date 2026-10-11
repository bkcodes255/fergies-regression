"""The hairdryer: This Week's calls as one tile that gets hotter the more predicted points you're
leaving on the table, named after Fergie's dressing-room rants.

build_calls() turns the tab's existing recommendations (captain_recommendation, lineup_changes,
the top of suggest_transfer_plan) into a list of calls; render_hairdryer_html() draws them. A call
the squad already satisfies arrives as done and can't be unticked; an outstanding one can be
ticked off in the tile (remembered per gameweek in the browser) for the gap between making the
change in FPL and the next ingest picking it up.
"""
from __future__ import annotations

import html
import json

# Points still on the table at which the hairdryer runs at full blast (level 10). A week's three
# calls typically add up to 3-10 predicted points, so 8 keeps a normal bad week near the top
# without pinning every week at 10.
FULL_BLAST_PTS = 8.0


def _names(squad, codes) -> list[str]:
    return squad.loc[squad["player_code"].isin(codes), "web_name"].tolist()


def _short(names: list[str]) -> str:
    """For shouting: "Wood", "Wood and Eze", "Wood and 3 more"."""
    if len(names) <= 2:
        return " and ".join(names)
    return f"{names[0]} and {len(names) - 1} more"


def build_calls(cap: dict | None, changes: dict, squad, starting, plan) -> list[dict]:
    """One dict per call: id, kind, title, detail, gain (pts, 0 when done), done, shout, mutter.
    Calls the model has no opinion on yet (no captain prediction) are left out entirely."""
    calls = []

    if cap is not None:
        if cap["matches"]:
            calls.append({"id": "cap", "kind": "Captain", "title": f"Armband on {cap['current_name']}",
                          "detail": "Already the model's top pick in your XI", "gain": 0.0, "done": True})
        else:
            # Moving the armband swaps which player's points are doubled, so the gain is the
            # difference between the two predictions, not the new captain's whole score.
            gain = float(cap["recommended_points"] - cap["current_points"])
            rec, cur = cap["recommended_name"], cap["current_name"]
            calls.append({
                "id": "cap", "kind": "Captain", "title": f"Armband to {rec}",
                "detail": f"{cap['recommended_points']:.1f} predicted against {cap['current_points']:.1f} for {cur}",
                "gain": gain, "done": False,
                "shout": f"The armband on {cur}? {rec}'s on {cap['recommended_points']:.1f}. Give him the armband.",
                "mutter": f"Nearly there. Get the armband on {rec} and we're done.",
            })

    bring_in, bench_out = _names(squad, changes["bring_in_codes"]), _names(squad, changes["bench_out_codes"])
    if not bring_in and not bench_out:
        calls.append({"id": "sub", "kind": "Lineup", "title": "Lineup is right",
                      "detail": "No swap improves this formation and squad", "gain": 0.0, "done": True})
    else:
        gain = float(changes["optimal_points"] - starting["predicted_points"].sum())
        on, off = ", ".join(bring_in), ", ".join(bench_out)
        s_on, s_off = _short(bring_in), _short(bench_out)
        calls.append({
            "id": "sub", "kind": "Lineup", "title": f"{on} on, {off} off",
            "detail": f"Best formation from your 15: {'-'.join(map(str, changes['formation']))}",
            "gain": gain, "done": False,
            "shout": f"{s_off} starting? Get {'them' if len(bench_out) > 1 else 'him'} off that pitch. {s_on} on. Now.",
            "mutter": f"Better. Now get {s_on} on for {s_off}.",
        })

    if plan is None or plan.empty:
        calls.append({"id": "xfer", "kind": "Transfer", "title": "No transfer needed",
                      "detail": "No move beats its cost this week", "gain": 0.0, "done": True})
    else:
        top = plan.iloc[0]
        hit = f"{int(top['hit'])}-point hit" if top["hit"] else "no hit"
        calls.append({
            "id": "xfer", "kind": "Transfer", "title": f"Sell {top['sell']}, buy {top['buy']}",
            "detail": f"Net {top['net']:+.1f} pts after costs, {hit}",
            "gain": float(top["net"]), "done": False,
            "shout": f"Why is {top['sell']} still in my squad? Sell him. Today.",
            "mutter": f"Good. One more thing: {top['buy']}. Go and sign him.",
        })
    return calls


def render_hairdryer_html(calls: list[dict], gameweek: int | None, font_css: str) -> str:
    # json.dumps output is safe inside <script> once "</" can't close the tag early.
    data = json.dumps({"calls": calls, "gw": gameweek, "full": FULL_BLAST_PTS}).replace("</", "<\\/")
    rows = []
    for c in calls:
        state = "checked disabled" if c["done"] else ""
        detail = html.escape(c["detail"]) + (" · done in FPL" if c["done"] else "")
        gain = "" if c["done"] else f"+{c['gain']:.1f}"
        rows.append(
            f'<li><label class="{"locked" if c["done"] else ""}">'
            f'<input type="checkbox" data-id="{html.escape(c["id"])}" {state}>'
            f'<span class="box" aria-hidden="true"></span>'
            f'<span class="t">{html.escape(c["title"])}<small>{html.escape(c["kind"])} · {detail}</small></span>'
            f'<span class="p">{gain}</span></label></li>'
        )
    rows = "".join(rows)
    return f"""
    {font_css}
    <link rel="preload" as="style" href="https://fonts.googleapis.com/css2?family=Anton&display=swap"
          onload="this.onload=null;this.rel='stylesheet'">
    <style>
      .hd {{ --heat: 1; color: #fff; padding: clamp(16px, 3vw, 32px); position: relative; overflow: hidden; border-radius: 16px;
        background: radial-gradient(120% 120% at 15% 50%,
          hsl(calc(48 - var(--heat) * 48) 95% calc(56% - var(--heat) * 8%)) 0%,
          hsl(calc(28 - var(--heat) * 28) 85% calc(30% - var(--heat) * 6%)) 70%, #160806 100%);
        transition: background .6s; display: grid; grid-template-columns: minmax(0, 0.8fr) minmax(0, 1.2fr);
        gap: clamp(16px, 3vw, 36px); align-items: center; }}
      @media (max-width: 640px) {{ .hd {{ grid-template-columns: 1fr; }} .hd svg {{ max-width: 150px; }} }}
      .hd svg {{ width: 100%; max-width: 320px; height: auto; display: block; margin: 0 auto; }}
      .wind path {{ stroke: rgba(255,255,255,.75); stroke-width: 6; stroke-linecap: round; fill: none; stroke-dasharray: 40 30;
        animation: wind calc(1.6s - var(--heat) * 1.1s) linear infinite; opacity: calc(.15 + var(--heat) * .85); }}
      @keyframes wind {{ to {{ stroke-dashoffset: -70; }} }}
      /* .disp * as well: font_css sets a font on every element, which would beat inheritance. */
      .disp, .disp * {{ font-family: 'Anton', Impact, 'Arial Narrow', sans-serif; font-weight: 400; }}
      .gauge {{ font-size: 14px; letter-spacing: .12em; text-transform: uppercase; opacity: .9; }}
      .level {{ font-size: clamp(56px, 12vw, 104px); line-height: .9; }}
      .level small {{ font-size: .35em; opacity: .8; }}
      .rant {{ font-size: clamp(20px, 4.4vw, 32px); line-height: 1.1; text-transform: uppercase; margin: 8px 0 16px;
        text-wrap: balance; animation: shake calc(.5s - var(--heat) * .35s) infinite; }}
      @keyframes shake {{ 25% {{ transform: translate(calc(var(--heat) * 2px), calc(var(--heat) * -1px)); }}
                          75% {{ transform: translate(calc(var(--heat) * -2px), calc(var(--heat) * 1px)); }} }}
      ul {{ display: grid; gap: 8px; margin: 0; padding: 0; list-style: none; }}
      label {{ display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 12px; align-items: center;
        background: rgba(0,0,0,.3); padding: 10px 14px; border-radius: 10px; cursor: pointer; }}
      label.locked {{ cursor: default; }}
      input {{ position: absolute; opacity: 0; width: 1px; height: 1px; }}
      .box {{ width: 24px; height: 24px; border: 2.5px solid #fff; border-radius: 6px; display: grid; place-items: center;
        font-weight: 800; color: #1d1f1a; }}
      input:checked ~ .box {{ background: #fff; }}
      input:checked ~ .box::after {{ content: "\\2713"; }}
      input:focus-visible ~ .box {{ outline: 3px solid #fff; outline-offset: 3px; }}
      .t {{ font-weight: 600; font-size: 15px; line-height: 1.3; min-width: 0; }}
      .t small {{ display: block; font-weight: 400; opacity: .85; font-size: 13px; }}
      .p {{ font-family: 'Anton', Impact, sans-serif; font-size: 22px; }}
      label:has(input:checked) .t, label:has(input:checked) .p {{ opacity: .6; }}
      label:has(input:checked:not(:disabled)) .t {{ text-decoration: line-through; }}
      .note {{ font-size: 12.5px; opacity: .85; margin-top: 12px; line-height: 1.4; }}
      @media (prefers-reduced-motion: reduce) {{ .rant, .wind path {{ animation: none; }} }}
    </style>
    <div class="hd" id="hd">
      <svg viewBox="0 0 400 300" aria-hidden="true">
        <g class="wind"><path d="M250 90 Q300 70 390 85"/><path d="M262 130 Q320 128 395 130"/><path d="M250 170 Q300 190 390 175"/></g>
        <rect x="70" y="80" width="190" height="110" rx="55" fill="#f2f2ee"/>
        <rect x="236" y="96" width="34" height="78" rx="10" fill="#c8c8c0"/>
        <circle cx="125" cy="135" r="34" fill="#1d1f1a"/><circle cx="125" cy="135" r="14" fill="#5b5f55"/>
        <path d="M120 185 L100 285 L150 285 L165 185 Z" fill="#e2e2dc"/>
        <rect x="128" y="205" width="16" height="30" rx="4" fill="#d7263d"/>
      </svg>
      <div>
        <div class="disp gauge">Hairdryer level</div>
        <div class="disp level"><span id="lvl">0</span><small> / 10</small></div>
        <p class="disp rant" id="rant" aria-live="polite"></p>
        <ul id="calls">{rows}</ul>
        <p class="note" id="note"></p>
      </div>
    </div>
    <script>
      const D = {data};
      const hd = document.getElementById("hd");
      const key = id => "fergie-hairdryer-gw" + D.gw + "-" + id;
      const boxes = [...document.querySelectorAll("input[data-id]")];
      boxes.forEach(b => {{
        if (b.disabled) return;
        try {{ if (localStorage.getItem(key(b.dataset.id)) === "1") b.checked = true; }} catch (e) {{}}
        b.addEventListener("change", () => {{
          try {{ b.checked ? localStorage.setItem(key(b.dataset.id), "1") : localStorage.removeItem(key(b.dataset.id)); }} catch (e) {{}}
          update();
        }});
      }});
      function update() {{
        const open = D.calls.filter(c => !c.done && !boxes.find(b => b.dataset.id === c.id).checked)
                            .sort((a, b) => b.gain - a.gain);
        const left = open.reduce((s, c) => s + c.gain, 0);
        const heat = Math.min(left / D.full, 1);
        hd.style.setProperty("--heat", heat.toFixed(3));
        document.getElementById("lvl").textContent = Math.round(heat * 10);
        const outstanding = D.calls.filter(c => !c.done).length;
        let line = "Aye. That'll do, son. Now go and enjoy your weekend.";
        if (open.length && open.length === outstanding && outstanding > 1) line = open.map(c => c.shout.split("?")[0] + "?").join(" ") + " Get in here. Now.";
        else if (open.length) line = heat > .5 ? open[0].shout : open[0].mutter;
        document.getElementById("rant").textContent = line;
        document.getElementById("note").textContent = left > 0
          ? "+" + left.toFixed(1) + " predicted points still on the table. Tick a call once you've made it in FPL; the next data refresh confirms it."
          : "Nothing left on the table this gameweek.";
      }}
      update();
    </script>
    """
