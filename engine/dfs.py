"""DFS lineup optimizer (DraftKings / FanDuel, NFL / NBA).

Every lineup is solved as a slot-assignment MILP: one binary variable per
(player, eligible slot). That guarantees lineups are actually legal for the
site — the old NBA optimizer picked any 8 players and labeled them
"PG, SG, SF..." by projection, so it could hand you three centers.

Pool columns: name, id, positions (list[str]), team, opp, game, salary,
              proj, ceiling, leverage
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

SITES = {
    ("nfl", "DK"): {"cap": 50000, "slots": [("QB", {"QB"}), ("RB", {"RB"}), ("RB", {"RB"}), ("WR", {"WR"}),
                                             ("WR", {"WR"}), ("WR", {"WR"}), ("TE", {"TE"}),
                                             ("FLEX", {"RB", "WR", "TE"}), ("DST", {"DST"})]},
    ("nfl", "FD"): {"cap": 60000, "slots": [("QB", {"QB"}), ("RB", {"RB"}), ("RB", {"RB"}), ("WR", {"WR"}),
                                             ("WR", {"WR"}), ("WR", {"WR"}), ("TE", {"TE"}),
                                             ("FLEX", {"RB", "WR", "TE"}), ("DEF", {"DST"})]},
    ("nba", "DK"): {"cap": 50000, "slots": [("PG", {"PG"}), ("SG", {"SG"}), ("SF", {"SF"}), ("PF", {"PF"}),
                                             ("C", {"C"}), ("G", {"PG", "SG"}), ("F", {"SF", "PF"}),
                                             ("UTIL", {"PG", "SG", "SF", "PF", "C"})]},
    ("nba", "FD"): {"cap": 60000, "slots": [("PG", {"PG"}), ("PG", {"PG"}), ("SG", {"SG"}), ("SG", {"SG"}),
                                             ("SF", {"SF"}), ("SF", {"SF"}), ("PF", {"PF"}), ("PF", {"PF"}),
                                             ("C", {"C"})]},
}

_POS_ALIASES = {"D": "DST", "DEF": "DST", "DST": "DST"}


def parse_positions(raw) -> list[str]:
    if isinstance(raw, (list, tuple, set)):
        parts = list(raw)
    else:
        parts = str(raw).replace(",", "/").split("/")
    return [_POS_ALIASES.get(p.strip().upper(), p.strip().upper()) for p in parts if p and p.strip()]


def optimize(pool: pd.DataFrame, sport: str, site: str, objective: str = "proj", num_lineups: int = 1,
             min_unique: int = 2, max_exposure: float = 1.0, locked=(), excluded=(),
             stack: str = "none", bringback: bool = False, no_opp_vs_dst: bool = True,
             max_1_rb_per_team: bool = False, max_per_team: int | None = None) -> list[pd.DataFrame]:
    """stack (NFL): 'none' | 'qb1' (QB + 1 pass catcher) | 'qb2' (QB + 2)."""
    spec = SITES[(sport, site)]
    slots = spec["slots"]
    pool = pool.reset_index(drop=True).copy()
    pool["positions"] = pool["positions"].apply(parse_positions)
    locked, excluded = set(locked), set(excluded)
    pool = pool[~pool["name"].isin(excluded)].reset_index(drop=True)
    n_p, n_s = len(pool), len(slots)
    if n_p == 0:
        return []

    pairs = [(p, s) for p in range(n_p) for s in range(n_s) if set(pool.at[p, "positions"]) & slots[s][1]]
    if not pairs:
        return []
    pi = np.array([p for p, _ in pairs])
    si = np.array([s for _, s in pairs])
    nv = len(pairs)
    score = pool[objective].to_numpy(float)[pi]
    sal = pool["salary"].to_numpy(float)[pi]

    def player_rows(mask):
        """Row over vars equal to sum of y_p for players in mask."""
        return mask[pi].astype(float)

    A, lo, hi = [], [], []

    def add(row, l, h):
        A.append(row); lo.append(l); hi.append(h)

    for s in range(n_s):
        add((si == s).astype(float), 1, 1)
    for p in np.unique(pi):
        add((pi == p).astype(float), 0, 1)
    add(sal, 0, spec["cap"])

    names = pool["name"].to_numpy()
    for nm in locked:
        m = names == nm
        if m.any():
            add(player_rows(m), 1, 1)

    pos_has = lambda pos: pool["positions"].apply(lambda ps: pos in ps).to_numpy()
    team = pool["team"].to_numpy()
    opp = pool["opp"].to_numpy()

    if sport == "nfl":
        is_qb, is_dst, is_rb = pos_has("QB"), pos_has("DST"), pos_has("RB")
        is_pc = pos_has("WR") | pos_has("TE")
        is_skill = is_pc | is_rb
        for t in np.unique(team):
            qb_t = player_rows(is_qb & (team == t))
            if not qb_t.any():
                continue
            k = {"qb1": 1, "qb2": 2}.get(stack, 0)
            if k:
                add(player_rows(is_pc & (team == t)) - k * qb_t, 0, np.inf)
            if bringback:
                add(player_rows(is_skill & (opp == t)) - qb_t, 0, np.inf)
        if no_opp_vs_dst:
            for t in np.unique(team):
                dst_t = player_rows(is_dst & (team == t))
                if dst_t.any():
                    # if we roster T's defense, no offensive players facing T
                    add(player_rows((is_skill | is_qb) & (opp == t)) + 8 * dst_t, 0, 8)
        if max_1_rb_per_team:
            for t in np.unique(team):
                add(player_rows(is_rb & (team == t)), 0, 1)
    if max_per_team:
        for t in np.unique(team):
            add(player_rows(team == t), 0, max_per_team)

    lineups, appear = [], np.zeros(n_p)
    max_app = max(1, int(np.ceil(num_lineups * max_exposure)))
    for _ in range(num_lineups):
        ub = np.ones(nv)
        for p in range(n_p):
            if appear[p] >= max_app and names[p] not in locked:
                ub[pi == p] = 0
        res = milp(c=-score, integrality=np.ones(nv), bounds=Bounds(np.zeros(nv), ub),
                   constraints=LinearConstraint(np.array(A), lo, hi))
        if not res.success:
            break
        chosen = np.where(res.x > 0.5)[0]
        sel_players = pi[chosen]
        appear[sel_players] += 1
        lu = pool.iloc[sel_players].copy()
        lu["slot"] = [slots[s][0] for s in si[chosen]]
        lu["_order"] = si[chosen]
        lineups.append(lu.sort_values("_order").drop(columns="_order").reset_index(drop=True))
        mask = np.zeros(n_p, bool)
        mask[sel_players] = True
        add(player_rows(mask), 0, n_s - min_unique)
    return lineups


def export_csv(lineups: list[pd.DataFrame], sport: str, site: str) -> bytes:
    """Bulk-upload CSV: one column per roster slot, values = player IDs."""
    slots = [s for s, _ in SITES[(sport, site)]["slots"]]
    buf = io.StringIO()
    buf.write(",".join(slots) + "\n")
    for lu in lineups:
        buf.write(",".join(str(x) for x in lu["id"].tolist()) + "\n")
    return buf.getvalue().encode()


def parse_salary_csv(df: pd.DataFrame) -> pd.DataFrame:
    """DraftKings or FanDuel salary export -> name, id, positions, team, salary."""
    from .common import norm_name

    cols = {c.lower().strip(): c for c in df.columns}
    out = pd.DataFrame()
    if "name + id" in cols or ("name" in cols and "id" in cols and "teamabbrev" in cols):  # DraftKings
        out["name"] = df[cols["name"]].astype(str).str.strip()
        out["id"] = df[cols["id"]].astype(str)
        out["positions"] = df[cols["position"]].astype(str)
        out["team"] = df[cols["teamabbrev"]].astype(str).str.upper()
        out["salary"] = pd.to_numeric(df[cols["salary"]], errors="coerce")
        if "game info" in cols:  # "KC@BUF 11/24/2026 08:20PM ET"
            matchup = df[cols["game info"]].astype(str).str.split(" ").str[0].str.upper()
            away = matchup.str.split("@").str[0]
            home = matchup.str.split("@").str[1]
            out["opp"] = np.where(out["team"] == away, home, away)
        out["site"] = "DK"
    elif "nickname" in cols or ("first name" in cols and "last name" in cols):  # FanDuel
        if "nickname" in cols:
            out["name"] = df[cols["nickname"]].astype(str).str.strip()
        else:
            out["name"] = (df[cols["first name"]].astype(str) + " " + df[cols["last name"]].astype(str)).str.strip()
        out["id"] = df[cols["id"]].astype(str)
        out["positions"] = df[cols["position"]].astype(str)
        out["team"] = df[cols["team"]].astype(str).str.upper()
        out["salary"] = pd.to_numeric(df[cols["salary"]], errors="coerce")
        if "opponent" in cols:
            out["opp"] = df[cols["opponent"]].astype(str).str.upper()
        out["site"] = "FD"
    else:
        raise ValueError("Unrecognized salary file — expected a DraftKings or FanDuel export.")
    out["positions"] = out["positions"].apply(parse_positions)
    out["name_norm"] = out["name"].apply(norm_name)
    is_dst = out["positions"].apply(lambda p: "DST" in p)
    out.loc[is_dst, "name_norm"] = out.loc[is_dst, "team"].str.lower() + " dst"
    return out.dropna(subset=["salary"])
