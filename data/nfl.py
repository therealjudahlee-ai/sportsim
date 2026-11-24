"""NFL data + context (nflverse via nflreadpy, ESPN injuries)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.common import Diagnostics, norm_name, now_et
from engine.config import INJURY, NFL_WIND_MPH, NFL_WIND_TOTAL_MULT, get_config
from engine.gamesim import predict_slate
from engine.props import NFL_VOLUME, nfl_env_factors, project_nfl, shrink_matchup
from engine.ratings import RatingsModel

from . import espn

SKILL = ("QB", "RB", "WR", "TE")
OUT_ROSTER_STATUS = {"RES", "CUT", "RET", "SUS", "NFI", "PUP", "EXE", "DEV", "UFA", "TRC"}


def _nfl():
    import nflreadpy as nfl  # imported lazily so tests/backtests on cached data don't need it
    return nfl


def _col(df, name):
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def load_games(seasons: list[int], diag: Diagnostics) -> pd.DataFrame:
    sched = _nfl().load_schedules(seasons).to_pandas()
    sched = sched[sched["game_type"] != "PRE"] if "game_type" in sched.columns else sched
    g = pd.DataFrame({
        "game_id": sched["game_id"].astype(str),
        "date": pd.to_datetime(sched["gameday"]),
        "season": sched["season"].astype(int),
        "week": sched["week"].astype(int),
        "home": sched["home_team"], "away": sched["away_team"],
        "home_name": sched["home_team"], "away_name": sched["away_team"],
        "home_pts": pd.to_numeric(sched["home_score"], errors="coerce"),
        "away_pts": pd.to_numeric(sched["away_score"], errors="coerce"),
        "neutral": sched.get("location", pd.Series("Home", index=sched.index)).astype(str).str.lower().eq("neutral"),
        # nflverse spread_line is POSITIVE when the home team is favored -> flip to book convention
        "home_spread": -pd.to_numeric(_col(sched, "spread_line"), errors="coerce"),
        "total_line": pd.to_numeric(_col(sched, "total_line"), errors="coerce"),
        "home_ml": pd.to_numeric(_col(sched, "home_moneyline"), errors="coerce"),
        "away_ml": pd.to_numeric(_col(sched, "away_moneyline"), errors="coerce"),
        "roof": sched.get("roof", pd.Series("", index=sched.index)).astype(str).str.lower(),
        "wind": pd.to_numeric(_col(sched, "wind"), errors="coerce"),
        "gametime": sched.get("gametime", pd.Series("", index=sched.index)).astype(str),
        "weekday": sched.get("weekday", pd.Series("", index=sched.index)).astype(str),
    })
    diag.note(f"NFL schedule: {len(g)} games, {g['home_pts'].notna().sum()} completed")
    return g


def _dfs_points(df: pd.DataFrame):
    def col(*names):
        for n in names:
            if n in df.columns:
                return df[n].fillna(0)
        return pd.Series(0.0, index=df.index)

    py, ptd, ints = col("passing_yards"), col("passing_tds"), col("passing_interceptions", "interceptions")
    ry, rtd = col("rushing_yards"), col("rushing_tds")
    rec, recy, rectd = col("receptions"), col("receiving_yards"), col("receiving_tds")
    fum = col("rushing_fumbles_lost") + col("receiving_fumbles_lost") + col("sack_fumbles_lost")
    sttd = col("special_teams_tds")
    two = col("passing_2pt_conversions") + col("rushing_2pt_conversions") + col("receiving_2pt_conversions")
    base = py * 0.04 + ptd * 4 - ints + ry * 0.1 + rtd * 6 + recy * 0.1 + rectd * 6 + sttd * 6 + two * 2
    dk = base + rec * 1.0 - fum + np.where(py >= 300, 3, 0) + np.where(ry >= 100, 3, 0) + np.where(recy >= 100, 3, 0)
    fd = base + rec * 0.5 - fum * 2
    return dk, fd


def build_context(diag: Diagnostics | None = None) -> dict:
    diag = diag or Diagnostics()
    nfl = _nfl()
    cfg = get_config("nfl")
    season, week = int(nfl.get_current_season()), int(nfl.get_current_week())
    seasons = [season - 1, season]

    games = load_games(seasons, diag)

    # ---------------- player logs
    ps = nfl.load_player_stats(seasons).to_pandas()
    reg = ps[ps["season_type"] == "REG"].copy()
    team_col = "team" if "team" in reg.columns else "recent_team"
    reg = reg.rename(columns={team_col: "team"})
    reg["sort_key"] = reg["season"] * 100 + reg["week"]
    for c in ("rushing_tds", "receiving_tds", "special_teams_tds"):
        if c not in reg.columns:
            reg[c] = 0
    reg["anytime_td"] = reg["rushing_tds"].fillna(0) + reg["receiving_tds"].fillna(0) + reg["special_teams_tds"].fillna(0)
    reg["dk_points"], reg["fd_points"] = _dfs_points(reg)
    reg["name_norm"] = reg["player_display_name"].map(norm_name)   # computed ONCE
    reg = reg.sort_values("sort_key")

    # ---------------- rosters, depth chart, injuries
    out_status: dict[str, str] = {}
    roster_team: dict[str, str] = {}
    try:
        ro = nfl.load_rosters_weekly([season]).to_pandas()
        ro = ro[ro["week"] == ro["week"].max()]
        nm = "full_name" if "full_name" in ro.columns else "player_name"
        for _, r in ro.iterrows():
            n = norm_name(r.get(nm, ""))
            stt = str(r.get("status", "ACT")).upper()
            if stt in OUT_ROSTER_STATUS:
                out_status[n] = "out"
            elif stt in ("ACT", "ACTIVE", "INA"):
                roster_team[n] = str(r.get("team", ""))
    except Exception as e:
        diag.warn("NFL weekly rosters unavailable", e)

    qb1: dict[str, str] = {}
    try:
        dc = nfl.load_depth_charts([season]).to_pandas()
        for c in ("dt", "week"):
            if c in dc.columns:
                dc = dc[dc[c] == dc[c].max()]
                break
        pos_c = next((c for c in ("pos_abb", "position", "depth_position") if c in dc.columns), None)
        rank_c = next((c for c in ("pos_rank", "depth_team") if c in dc.columns), None)
        name_c = next((c for c in ("player_name", "full_name") if c in dc.columns), None)
        team_c = next((c for c in ("team", "club_code") if c in dc.columns), None)
        if all((pos_c, rank_c, name_c, team_c)):
            q = dc[(dc[pos_c] == "QB") & (pd.to_numeric(dc[rank_c], errors="coerce") == 1)]
            qb1 = {str(r[team_c]): norm_name(r[name_c]) for _, r in q.iterrows()}
        else:
            diag.warn("NFL depth chart columns not recognized; QB1 detection disabled")
    except Exception as e:
        diag.warn("NFL depth charts unavailable", e)

    inj_rows = []
    try:
        inj = nfl.load_injuries([season]).to_pandas()
        inj = inj[inj["week"] == week]
        nm = "full_name" if "full_name" in inj.columns else "player_name"
        for _, r in inj.iterrows():
            s = str(r.get("report_status", "")).lower()
            if s and s != "nan":
                inj_rows.append({"name_norm": norm_name(r.get(nm, "")), "name": r.get(nm, ""),
                                 "status": s, "team": str(r.get("team", "")), "source": "nflverse"})
    except Exception as e:
        diag.warn("nflverse injury report unavailable", e)
    espn_inj = espn.injuries("football/nfl", diag)
    for _, r in espn_inj.iterrows():
        inj_rows.append({"name_norm": r["name_norm"], "name": r["name"], "status": r["status"],
                         "team": r["team_abbr"], "source": "espn"})
    injuries = pd.DataFrame(inj_rows, columns=["name_norm", "name", "status", "team", "source"])
    if not injuries.empty:
        injuries = injuries.drop_duplicates("name_norm", keep="last")   # ESPN is fresher
    for _, r in injuries.iterrows():
        if espn.status_weight(r["status"]) >= 0.75:
            out_status[r["name_norm"]] = r["status"]

    # ---------------- team injury adjustments (points off a team's projection)
    cur = reg[reg["season"] == season]
    ppg = cur.groupby("name_norm").agg(dk=("dk_points", "mean"), n=("dk_points", "size"),
                                       team=("team", "last"), pos=("position", "last"))
    team_adj: dict[str, float] = {}
    adj_notes: dict[str, list] = {}
    for _, r in injuries.iterrows():
        w = espn.status_weight(r["status"])
        if w <= 0 or r["name_norm"] not in ppg.index:
            continue
        p = ppg.loc[r["name_norm"]]
        team = roster_team.get(r["name_norm"], p["team"])
        if p["pos"] == "QB" and qb1.get(team) == r["name_norm"]:
            d = INJURY["nfl_qb1_out"] * w
        elif p["pos"] in SKILL and p["n"] >= 3 and p["dk"] >= 12:
            d = INJURY["nfl_skill_out"] * w
        else:
            continue
        team_adj[team] = team_adj.get(team, 0.0) - d
        adj_notes.setdefault(team, []).append(f"{r['name']} ({r['status']})")
    for t in team_adj:
        team_adj[t] = max(team_adj[t], -(INJURY["nfl_qb1_out"] + INJURY["nfl_skill_cap"]))

    # ---------------- ratings + this week's slate
    model = RatingsModel(games, cfg)
    as_of = pd.Timestamp(now_et().date())
    ratings = model.ratings_as_of(as_of, season)
    slate = games[(games["season"] == season) & (games["week"] == week)].copy()
    total_mult = {}
    for _, g in slate.iterrows():
        if g["roof"] in ("outdoors", "open") and pd.notna(g["wind"]) and g["wind"] >= NFL_WIND_MPH:
            total_mult[g["game_id"]] = NFL_WIND_TOTAL_MULT
    preds = predict_slate(ratings, slate, cfg, team_adj=team_adj, total_mult=total_mult)
    if not preds.empty:
        preds["injury_notes"] = preds.apply(
            lambda r: "; ".join(adj_notes.get(r["home"], []) + adj_notes.get(r["away"], [])), axis=1)
        missing = (~preds["known"]).sum()
        if missing:
            diag.warn(f"{missing} NFL games have a team with no rating")

    # ---------------- matchup multipliers & league baselines for props
    def_lookup = {}
    pos_df = reg[reg["position"].isin(SKILL)]
    for stat in ["passing_yards", "rushing_yards", "receiving_yards", "receptions", "anytime_td",
                 "passing_tds", "dk_points", "fd_points"]:
        if stat not in pos_df.columns:
            continue
        gt = (pos_df.groupby(["position", "opponent_team", "season", "week", "sort_key"])[stat]
              .sum().reset_index().sort_values("sort_key"))
        for pos, pgrp in gt.groupby("position"):
            lg = pgrp.tail(200)[stat].mean()
            if not lg or not np.isfinite(lg):
                continue
            for opp, grp in pgrp.groupby("opponent_team"):
                rec = grp.tail(12)
                w = np.exp(np.linspace(-1.2, 0.0, len(rec)))
                def_lookup[(opp, pos, stat)] = (float(np.sum(w * rec[stat]) / w.sum() / lg), len(rec))

    league = {}
    for pos in SKILL:
        pdf = reg[reg["position"] == pos]
        for stat, (vcol, vmin, _) in NFL_VOLUME.items():
            if vcol in pdf.columns and stat in pdf.columns:
                h = pdf[pdf[vcol] >= vmin]
                if h[vcol].sum() > 0:
                    league[("eff", pos, stat)] = float(h[stat].sum() / h[vcol].sum())
        reg_p = pdf[pdf["dk_points"] >= 5]
        if len(reg_p):
            league[("td_rate", pos)] = float(reg_p["anytime_td"].mean())
    team_games = games[games["home_pts"].notna() & (games["season"] == season)]
    league_team_pts = float(pd.concat([team_games["home_pts"], team_games["away_pts"]]).mean()) \
        if len(team_games) else cfg.league_avg_total / 2

    # ---------------- active player index for this week
    slate_teams = set(slate["home"]) | set(slate["away"])
    recent_cut = reg[(reg["season"] == season) & (reg["week"] >= max(1, week - 5))]
    if recent_cut.empty:
        recent_cut = reg[reg["season"] == season - 1]
    players = (recent_cut[recent_cut["position"].isin(SKILL)]
               .groupby("player_id").agg(name=("player_display_name", "last"), name_norm=("name_norm", "last"),
                                         position=("position", "last"), team=("team", "last"),
                                         dk=("dk_points", "mean"), att=("attempts", "mean"))
               .reset_index())
    players["team"] = players.apply(lambda r: roster_team.get(r["name_norm"], r["team"]), axis=1)
    players["status"] = players["name_norm"].map(out_status).fillna("")
    players = players[players["team"].isin(slate_teams)]
    if not players.empty:
        is_backup_qb = (players["position"] == "QB") & players.apply(
            lambda r: bool(qb1.get(r["team"])) and qb1.get(r["team"]) != r["name_norm"], axis=1)
        players = players[~is_backup_qb | (players["att"] >= 20)]
    players = players.sort_values("dk", ascending=False).reset_index(drop=True)

    game_info = {}
    for _, p in preds.iterrows():
        for side, other in (("home", "away"), ("away", "home")):
            game_info[p[side]] = {
                "opp": p[other], "game": f"{p['away']} @ {p['home']}", "game_id": p["game_id"],
                "team_pts": p[f"model_{side}_pts"], "opp_pts": p[f"model_{other}_pts"],
                "margin": p["model_margin"] if side == "home" else -p["model_margin"],
                "wind": p["wind"] if p["roof"] in ("outdoors", "open") and pd.notna(p["wind"]) else 0.0,
                "is_home": side == "home",
            }

    diag.note(f"NFL {season} week {week}: {len(slate)} games, {len(players)} active skill players")
    return {
        "sport": "nfl", "season": season, "week": week, "as_of": as_of,
        "games": games, "slate": slate, "predictions": preds, "ratings": ratings,
        "logs": reg, "players": players, "injuries": injuries, "out_status": out_status,
        "qb1": qb1, "team_adj": team_adj, "def_lookup": def_lookup, "league": league,
        "league_team_pts": league_team_pts, "game_info": game_info, "diag": diag.to_dict(),
    }


def project_player(ctx: dict, player_id, stat: str, bump: float = 1.0, opp: str | None = None) -> dict | None:
    pl = ctx["players"]
    row = pl[pl["player_id"] == player_id]
    logs = ctx["logs"][ctx["logs"]["player_id"] == player_id]
    if logs.empty:
        return None
    if row.empty:
        last = logs.iloc[-1]
        row = pd.Series({"player_id": player_id, "name": last["player_display_name"], "name_norm": last["name_norm"],
                         "position": last["position"], "team": last["team"], "status": ""})
    else:
        row = row.iloc[0]
    gi = ctx["game_info"].get(row["team"], {})
    opp = opp or gi.get("opp")
    raw, n = ctx["def_lookup"].get((opp, row["position"], stat), (1.0, 0))
    dm = shrink_matchup(raw, n)
    env = nfl_env_factors(stat, gi.get("team_pts"), gi.get("margin"), ctx["league_team_pts"], gi.get("wind", 0.0))
    res = project_nfl(logs, stat, row["position"], ctx["league"], def_mult=dm, env=env, bump=bump)
    if res is None:
        return None
    res.update({"player": row["name"], "player_id": player_id, "player_norm": row["name_norm"],
                "position": row["position"], "team": row["team"], "opp": opp or "",
                "game": gi.get("game", ""), "stat": stat, "def_mult": dm, "status": row.get("status", ""),
                "hist": logs[stat].tail(20).tolist() if stat in logs.columns else []})
    return res


def all_projections(ctx: dict, stats: list[str], include_out: bool = False) -> list[dict]:
    pos_stats = {"QB": {"passing_yards", "rushing_yards", "passing_tds", "anytime_td", "dk_points", "fd_points"},
                 "RB": {"rushing_yards", "receiving_yards", "receptions", "anytime_td", "dk_points", "fd_points"},
                 "WR": {"receiving_yards", "receptions", "anytime_td", "dk_points", "fd_points", "rushing_yards"},
                 "TE": {"receiving_yards", "receptions", "anytime_td", "dk_points", "fd_points"}}
    out = []
    for _, p in ctx["players"].iterrows():
        if not include_out and p["status"] and espn.status_weight(p["status"]) >= 0.75:
            continue
        for s in stats:
            if s in pos_stats.get(p["position"], set()):
                r = project_player(ctx, p["player_id"], s)
                if r:
                    out.append(r)
    return out


def dfs_pool(ctx: dict, site: str, salaries: pd.DataFrame | None) -> pd.DataFrame:
    stat = "dk_points" if site == "DK" else "fd_points"
    projs = all_projections(ctx, [stat])
    sal = salaries.set_index("name_norm") if salaries is not None and not salaries.empty else None
    rows = []
    for p in projs:
        if sal is not None:
            if p["player_norm"] not in sal.index:
                continue
            s = sal.loc[p["player_norm"]]
            s = s.iloc[0] if isinstance(s, pd.DataFrame) else s
            salary, pid, positions = int(s["salary"]), s["id"], s["positions"]
        else:
            # rough DK pricing: ~2.5-2.9 projected pts per $1k. FanDuel salaries run ~20% higher.
            per_k = {"QB": 2.9, "RB": 2.6, "WR": 2.6, "TE": 2.5}[p["position"]]
            salary = int(np.clip(round(p["mean"] / per_k * 10) * 100, 3000, 9800)) * (1 if site == "DK" else 1.2)
            pid, positions = p["player"], [p["position"]]
        rows.append({"name": p["player"], "id": pid, "positions": positions, "team": p["team"], "opp": p["opp"],
                     "game": p["game"], "salary": int(salary), "proj": p["mean"], "ceiling": p["dist"].quantile(0.85)})
    for team, gi in ctx["game_info"].items():
        dst_proj = float(np.clip(13.0 - 0.33 * gi["opp_pts"], 2.0, 12.0))
        key = f"{team.lower()} dst"
        if sal is not None:
            if key not in sal.index:
                continue
            s = sal.loc[key]
            s = s.iloc[0] if isinstance(s, pd.DataFrame) else s
            salary, pid = int(s["salary"]), s["id"]
        else:
            salary, pid = int(round((2200 + dst_proj * 220) / 100) * 100), f"{team} DST"
        rows.append({"name": f"{team} DST", "id": pid, "positions": ["DST"], "team": team, "opp": gi["opp"],
                     "game": gi["game"], "salary": salary, "proj": dst_proj, "ceiling": dst_proj + 6.0})
    return _finish_pool(pd.DataFrame(rows), ctx)


def _finish_pool(df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    if df.empty:
        return df
    df["value"] = df["proj"] / df["salary"] * 1000
    team_pts = df["team"].map(lambda t: ctx["game_info"].get(t, {}).get("team_pts", ctx["league_team_pts"]))
    # crude public-ownership guess: value + team total. Replace with a real ownership source when you have one.
    df["own_est"] = np.clip((df["value"] - df["value"].median()) * 9 + (team_pts - ctx["league_team_pts"]) * 1.1 + 8,
                            1, 40)
    df["leverage"] = df["ceiling"] * (1 - df["own_est"] / 100 * 0.45)
    return df.sort_values("proj", ascending=False).reset_index(drop=True)
