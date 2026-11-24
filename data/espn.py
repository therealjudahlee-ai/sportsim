"""ESPN public scoreboard / injuries helpers (unofficial API — fine for personal
use; swap for a licensed feed before charging money)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from engine.common import Diagnostics, get_json, norm_name

SITE = "https://site.api.espn.com/apis/site/v2/sports"


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def parse_odds(comp: dict, home_abbr: str, away_abbr: str) -> dict:
    """Return home_spread (book line for HOME, negative = home favored), total, moneylines.

    Uses the human-readable 'details' string ("KC -3.5") because it names the
    favorite explicitly — that removes any guessing about the sign of 'spread'.
    """
    out = {"home_spread": np.nan, "total_line": np.nan, "home_ml": np.nan, "away_ml": np.nan}
    odds = (comp.get("odds") or [None])[0]
    if not odds:
        return out
    out["total_line"] = _num(odds.get("overUnder"))
    hto, ato = odds.get("homeTeamOdds") or {}, odds.get("awayTeamOdds") or {}
    out["home_ml"] = _num(hto.get("moneyLine"))
    out["away_ml"] = _num(ato.get("moneyLine"))

    details = str(odds.get("details") or "").strip()
    m = re.match(r"^([A-Za-z0-9&'.\- ]+?)\s+([-+]?\d+(?:\.\d+)?)$", details)
    if details.upper() in ("EVEN", "PK", "PICK", "PICK'EM"):
        out["home_spread"] = 0.0
    elif m:
        fav, pts = m.group(1).strip().upper(), abs(float(m.group(2)))
        if fav == str(home_abbr).upper():
            out["home_spread"] = -pts
        elif fav == str(away_abbr).upper():
            out["home_spread"] = pts
    if np.isnan(out["home_spread"]):
        spread = _num(odds.get("spread"))
        if np.isfinite(spread):
            if hto.get("favorite") is True:
                out["home_spread"] = -abs(spread)
            elif ato.get("favorite") is True:
                out["home_spread"] = abs(spread)
    return out


def scoreboard(path: str, diag: Diagnostics, what: str, **params) -> pd.DataFrame:
    """Generic ESPN scoreboard -> one row per game.
    path like 'basketball/nba'."""
    data = get_json(f"{SITE}/{path}/scoreboard", diag, what, **params)
    rows = []
    for ev in (data or {}).get("events", []):
        try:
            comp = ev["competitions"][0]
            hc = next(c for c in comp["competitors"] if c.get("homeAway") == "home")
            ac = next(c for c in comp["competitors"] if c.get("homeAway") == "away")
            ht, at = hc["team"], ac["team"]
            row = {
                "game_id": str(ev.get("id")),
                "date": pd.to_datetime(ev.get("date"), utc=True),
                "home_id": str(ht.get("id")), "away_id": str(at.get("id")),
                "home_abbr": ht.get("abbreviation", ""), "away_abbr": at.get("abbreviation", ""),
                "home_name": ht.get("displayName", ""), "away_name": at.get("displayName", ""),
                "home_location": ht.get("location", ""), "away_location": at.get("location", ""),
                "week": (ev.get("week") or {}).get("number"),
                "neutral": bool(comp.get("neutralSite", False)),
                "state": (ev.get("status") or {}).get("type", {}).get("state", ""),
                "home_score": _num(hc.get("score")), "away_score": _num(ac.get("score")),
            }
            row.update(parse_odds(comp, row["home_abbr"], row["away_abbr"]))
            rows.append(row)
        except Exception as e:  # one malformed event shouldn't kill the slate
            diag.warn(f"Skipped malformed ESPN event in {what}", e)
    return pd.DataFrame(rows)


def injuries(path: str, diag: Diagnostics) -> pd.DataFrame:
    """ESPN injury report -> name_norm, name, status (lowercase), team_abbr, comment."""
    data = get_json(f"{SITE}/{path}/injuries", diag, f"{path} injuries")
    rows = []
    for blk in (data or {}).get("injuries", []):
        blk_team = blk.get("displayName", "")
        for it in blk.get("injuries", []):
            ath = it.get("athlete") or {}
            team = ((ath.get("team") or {}).get("abbreviation")) or ""
            status = str(it.get("status", "")).lower()
            type_desc = str((it.get("type") or {}).get("description", "")).lower()
            if "injured reserve" in type_desc or status in ("ir", "injured reserve"):
                status = "out"
            rows.append({
                "name": ath.get("displayName", ""), "name_norm": norm_name(ath.get("displayName", "")),
                "status": status, "team_abbr": team, "team_name": blk_team,
                "comment": it.get("shortComment", ""),
            })
    return pd.DataFrame(rows, columns=["name", "name_norm", "status", "team_abbr", "team_name", "comment"])


def status_weight(status: str) -> float:
    from engine.config import INJURY
    s = str(status).lower()
    for k, w in INJURY["status_weight"].items():
        if k in s:
            return w
    if "suspen" in s or "reserve" in s or s == "ir":
        return 1.0
    return 0.0
