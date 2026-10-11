"""The plan payloads - transfer plans and post-transfer lineups as JSON.

The web site's This Week export (src/web_data/this_week.py) builds its transfer plans with
these; the read-only tools server will return the same payloads, so one builder keeps every
reader of "the plan" in agreement.

Conventions: money in decimal £m, times ISO-8601 UTC to the second, players by player_code with
web_name for display - never FPL's season-scoped element id.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.recommendations.horizon import DEFAULT_HORIZON
from src.recommendations.squad_optimizer import best_starting_xi


def _num(value: Any, digits: int = 2) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, digits)


def _int(value: Any) -> int | None:
    f = _num(value, 0)
    return None if f is None else int(f)


def _text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value) or None


def _iso(ts: Any) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.floor("s").isoformat().replace("+00:00", "Z")


def _hours_until(ts: Any) -> float | None:
    if ts is None or pd.isna(ts):
        return None
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return round((t - pd.Timestamp.now(tz="UTC")).total_seconds() / 3600, 1)


def _player(row: pd.Series, extra: dict | None = None) -> dict:
    out = {"player_code": _int(row.get("player_code")), "web_name": _text(row.get("web_name"))}
    if "position" in row:
        out["position"] = _text(row.get("position"))
    if "team" in row:
        out["team"] = _text(row.get("team"))
    if "price" in row:
        out["price"] = _num(row.get("price"), 1)
    out.update(extra or {})
    return out


def _xi_points(squad: pd.DataFrame, value_col: str) -> float:
    """Best XI's total on value_col with the captain (the XI's top scorer) counted twice."""
    xi, _ = best_starting_xi(squad, value_col=value_col)
    return float(xi[value_col].sum() + xi[value_col].max())


def transfer_plan_payload(rec: dict, horizon: str) -> dict:
    """One transfer plan, with player codes recovered from the squads before and after (the
    plan itself only carries names, which aren't unique league-wide)."""
    key = "1gw" if horizon == "1gw" else "5gw"
    plan_df = rec["plan_1gw"] if key == "1gw" else rec["plan_5gw"]
    before, after = rec["squad"], rec[f"squad_after_{key}"]
    value_col = "predicted_points" if key == "1gw" else "horizon_points"
    sold = before[~before["player_code"].isin(after["player_code"])]
    bought = after[~after["player_code"].isin(before["player_code"])]
    by_name_sold = {r["web_name"]: r for _, r in sold.iterrows()}
    by_name_bought = {r["web_name"]: r for _, r in bought.iterrows()}
    transfers = []
    for _, t in plan_df.iterrows():
        s, b = by_name_sold.get(t["sell"]), by_name_bought.get(t["buy"])
        transfers.append({
            "order": _int(t["transfer_num"]),
            "sell": _player(s, {"value": _num(s[value_col])}) if s is not None else {"web_name": t["sell"]},
            "buy": _player(b, {"value": _num(b[value_col])}) if b is not None else {"web_name": t["buy"]},
            "gain": _num(t["gain"]), "hit_cost": _int(t["hit"]), "net": _num(t["net"]),
            "uses_free_transfer": bool(t["free_transfer_used"]),
        })
    return {
        "horizon": key,
        "runs_if_silent": key == "1gw",
        "value": "predicted points next gameweek" if key == "1gw" else f"fixture-weighted points over {DEFAULT_HORIZON} gameweeks",
        "transfers": transfers,
        "total_net_gain": _num(plan_df["net"].sum()) if not plan_df.empty else 0.0,
        "total_hit_cost": int(plan_df["hit"].sum()) if not plan_df.empty else 0,
        "free_transfers": int(rec["free_transfers"]),
        "bank_before": _num(rec["bank"], 1), "bank_after": _num(rec[f"bank_after_{key}"], 1),
        "xi_points_before": _num(_xi_points(before, value_col)),
        "xi_points_after": _num(_xi_points(after, value_col)),
    }


def lineup_plan_payload(rec: dict, horizon: str, gameweek: int) -> dict:
    """The best XI, bench, captain and vice for a gameweek, built on the squad AFTER the
    given transfer plan - the rule the approval flow applies when a transfer executes."""
    key = "1gw" if horizon == "1gw" else "5gw"
    squad = rec[f"squad_after_{key}"]
    xi, formation = best_starting_xi(squad)
    bench = squad[~squad["player_code"].isin(xi["player_code"])].sort_values("predicted_points", ascending=False)
    ranked = xi.sort_values("predicted_points", ascending=False)
    captain, vice = ranked.iloc[0], ranked.iloc[1]
    pre_xi = set(rec["starting_xi"]["player_code"])
    plan_df = rec["plan_1gw"] if key == "1gw" else rec["plan_5gw"]
    return {
        "gameweek": gameweek,
        "built_on_transfers": [f"{t['sell']} -> {t['buy']}" for _, t in plan_df.iterrows()],
        "formation": f"{formation[0]}-{formation[1]}-{formation[2]}",
        "starting_xi": [_player(r, {"predicted_points": _num(r["predicted_points"])}) for _, r in xi.iterrows()],
        "bench": [_player(r, {"predicted_points": _num(r["predicted_points"]), "bench_order": i + 1})
                  for i, (_, r) in enumerate(bench.iterrows())],
        "captain": _player(captain, {"predicted_points": _num(captain["predicted_points"])}),
        "vice_captain": _player(vice, {"predicted_points": _num(vice["predicted_points"])}),
        "xi_predicted_points": _num(xi["predicted_points"].sum() + captain["predicted_points"]),
        "changed_from_pre_transfer_proposal": {
            "in": [r["web_name"] for _, r in xi.iterrows() if r["player_code"] not in pre_xi],
            "out": [r["web_name"] for _, r in rec["starting_xi"].iterrows() if r["player_code"] not in set(xi["player_code"])],
        },
        "note": (
            "The Telegram lineup approval is drawn up before any transfer; if the transfer executes, "
            "the bot re-plans the lineup to this. Bench order is by predicted points, as the bot sets it."
        ),
    }
