"""Optional live DraftKings salaries (unofficial lobby endpoints — they change
without notice and DK's terms don't allow scraping for commercial use).
Uploading the salary CSV from the contest page is the reliable path."""
from __future__ import annotations

import pandas as pd

from engine.common import Diagnostics, get_json, norm_name
from engine.dfs import parse_positions


def live_salaries(sport: str, diag: Diagnostics) -> pd.DataFrame:
    lobby = get_json(f"https://www.draftkings.com/lobby/getcontests?sport={sport}", diag, "DK lobby")
    groups = (lobby or {}).get("DraftGroups", [])
    classic = [g for g in groups if str(g.get("GameTypeId", g.get("ContestTypeId", ""))) in ("1", "21", "70")
               and g.get("GameCount", 0) >= 2] or groups
    if not classic:
        return pd.DataFrame()
    main = max(classic, key=lambda g: g.get("GameCount", 0))
    data = get_json(f"https://api.draftkings.com/draftgroups/v1/draftgroups/{main.get('DraftGroupId')}/draftables",
                    diag, "DK draftables")
    rows, seen = [], set()
    for it in (data or {}).get("draftables", []):
        pid = it.get("playerId", it.get("draftableId"))
        if pid in seen or not it.get("salary"):
            continue
        seen.add(pid)
        pos = parse_positions(it.get("position", ""))
        team = str(it.get("teamAbbreviation", "")).upper()
        name = str(it.get("displayName", "")).strip()
        rows.append({
            "name": name, "id": str(it.get("draftableId")), "positions": pos, "team": team,
            "salary": int(it["salary"]), "site": "DK",
            "name_norm": f"{team.lower()} dst" if "DST" in pos else norm_name(name),
        })
    if rows:
        diag.note(f"Loaded {len(rows)} live DK salaries (draft group {main.get('DraftGroupId')})")
    return pd.DataFrame(rows)
