"""NFL data + context (nflverse via nflreadpy, ESPN injuries)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.common import Diagnostics, norm_name, now_et
from engine.config import INJURY, NFL_WIND_MPH, NFL_WIND_TOTAL_MULT, get_config
from engine.gamesim import predict_slate
from engine.props import NFL_VOLUME, market_offered, nfl_env_factors, project_nfl, shrink_matchup
from engine.nfl_epa import QB_OUT_CAP, NFLModel, load_pbp_lean, qb_epa_values, qb_out_points, team_game_epa

from . import espn

SKILL = ("QB", "RB", "WR", "TE")
OUT_ROSTER_STATUS = {"RES", "CUT", "RET", "SUS", "NFI", "PUP", "EXE", "DEV", "UFA", "TRC"}


def _nfl():
    import nflreadpy as nfl  # imported lazily so tests/backtests on cached data don't need it
    return nfl


def _col(df, name):
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


NFLDATA_GAMES_CSV = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def _retry(fn, what: str, diag: Diagnostics, tries: int = 3, wait: float = 20.0):
    """nflverse re-uploads its release files several times a day; for a minute or two a file can 404.
    Retry before giving up."""
    import time
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i < tries - 1:
                diag.note(f"{what} failed ({type(e).__name__}); retrying in {wait:.0f}s")
                time.sleep(wait)
    raise last


def _load_schedules(seasons: list[int], diag: Diagnostics) -> pd.DataFrame:
    try:
        return _retry(lambda: _nfl().load_schedules(seasons).to_pandas(), "nflverse schedules", diag)
    except Exception as e:
        diag.warn("nflverse schedules release unavailable; using the nfldata games.csv mirror", e)
        df = pd.read_csv(NFLDATA_GAMES_CSV)
        return df[df["season"].isin(seasons)].copy()


def load_games(seasons: list[int], diag: Diagnostics) -> pd.DataFrame:
    sched = _load_schedules(seasons, diag)
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
    ps = _retry(lambda: nfl.load_player_stats(seasons).to_pandas(), "nflverse player stats", diag)
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
    qb_ids: dict[str, dict] = {}     # team -> {"qb1": gsis_id, "qb2": gsis_id}
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
        id_c = next((c for c in ("gsis_id", "player_id") if c in dc.columns), None)
        if all((pos_c, rank_c, name_c, team_c)):
            q = dc[dc[pos_c] == "QB"].copy()
            q["_rank"] = pd.to_numeric(q[rank_c], errors="coerce")
            for _, r in q[q["_rank"] == 1].iterrows():
                qb1[str(r[team_c])] = norm_name(r[name_c])
            if id_c:
                for _, r in q[q["_rank"].isin([1, 2])].iterrows():
                    qb_ids.setdefault(str(r[team_c]), {})[f"qb{int(r['_rank'])}"] = str(r[id_c])
        else:
            diag.warn("NFL depth chart columns not recognized; QB1 detection disabled")
    except Exception as e:
        diag.warn("NFL depth charts unavailable", e)

    # ---------------- play-by-play efficiency (EPA) + QB values
    egames, qb_vals = pd.DataFrame(), {}
    try:
        pbp = load_pbp_lean(nfl, seasons)
        egames = team_game_epa(pbp, games)
        qb_vals = qb_epa_values(pbp)
        del pbp
        diag.note(f"NFL play-by-play: EPA for {len(egames)} games, {len(qb_vals)} quarterbacks")
    except Exception as e:
        diag.warn("NFL play-by-play unavailable; using score-only ratings and flat QB injury values", e)

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

    cur = reg[reg["season"] == season]
    ppg = cur.groupby("name_norm").agg(dk=("dk_points", "mean"), n=("dk_points", "size"),
                                       team=("team", "last"), pos=("position", "last"),
                                       pid=("player_id", "last"))

    # ---------------- ratings + this week's slate
    model = NFLModel(games, egames, cfg)
    as_of = pd.Timestamp(now_et().date())
    ratings = model.ratings_as_of(as_of, season)
    slate = games[(games["season"] == season) & (games["week"] == week)].copy()
    total_mult = {}
    for _, g in slate.iterrows():
        if g["roof"] in ("outdoors", "open") and pd.notna(g["wind"]) and g["wind"] >= NFL_WIND_MPH:
            total_mult[g["game_id"]] = NFL_WIND_TOTAL_MULT

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

    # ---------------- active player index for this week (statuses filled in by _finalize)
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
    players = players[players["team"].isin(slate_teams)]
    if not players.empty:
        is_backup_qb = (players["position"] == "QB") & players.apply(
            lambda r: bool(qb1.get(r["team"])) and qb1.get(r["team"]) != r["name_norm"], axis=1)
        players = players[~is_backup_qb | (players["att"] >= 20)]
    players = players.sort_values("dk", ascending=False).reset_index(drop=True)

    ctx = {
        "sport": "nfl", "season": season, "week": week, "as_of": as_of, "cfg": cfg,
        "games": games, "slate": slate, "total_mult": total_mult, "ratings": ratings,
        "logs": reg, "base_players": players, "base_injuries": injuries, "roster_out": out_status,
        "roster_team": roster_team, "ppg": ppg, "qb1": qb1, "qb_ids": qb_ids, "qb_vals": qb_vals,
        "def_lookup": def_lookup, "league": league, "league_team_pts": league_team_pts,
        "overrides": {}, "diag": diag.to_dict(),
    }
    _finalize(ctx, injuries)
    if not ctx["predictions"].empty and (~ctx["predictions"]["known"]).any():
        diag.warn(f"{(~ctx['predictions']['known']).sum()} NFL games have a team with no rating")
        ctx["diag"] = diag.to_dict()
    diag.note(f"NFL {season} week {week}: {len(slate)} games, {len(players)} active skill players")
    ctx["diag"] = diag.to_dict()
    return ctx


def injury_adjustments(ctx: dict, injuries: pd.DataFrame) -> tuple[dict, dict, dict]:
    """(team_adj points, notes per team, out_status by name) from an injury table."""
    out_status = dict(ctx["roster_out"])
    team_adj: dict[str, float] = {}
    notes: dict[str, list] = {}
    ppg, qb1, qb_ids = ctx["ppg"], ctx["qb1"], ctx["qb_ids"]
    for _, r in injuries.iterrows():
        w = espn.status_weight(r["status"])
        if str(r["status"]).lower() == "active":
            out_status.pop(r["name_norm"], None)
            continue
        if w >= 0.75:
            out_status[r["name_norm"]] = r["status"]
        if w <= 0 or r["name_norm"] not in ppg.index:
            continue
        p = ppg.loc[r["name_norm"]]
        team = ctx["roster_team"].get(r["name_norm"], p["team"])
        if p["pos"] == "QB" and qb1.get(team) == r["name_norm"]:
            ids = qb_ids.get(team, {})
            pts, why = qb_out_points(ctx["qb_vals"], ids.get("qb1", p["pid"]), ids.get("qb2"),
                                     INJURY["nfl_qb1_out"])
            d = pts * w
            note = f"{r['name']} ({r['status']}, -{pts:.1f}: {why})"
        elif p["pos"] in SKILL and p["n"] >= 3 and p["dk"] >= 12:
            d = INJURY["nfl_skill_out"] * w
            note = f"{r['name']} ({r['status']})"
        else:
            continue
        team_adj[team] = team_adj.get(team, 0.0) - d
        notes.setdefault(team, []).append(note)
    for t in team_adj:
        team_adj[t] = max(team_adj[t], -(QB_OUT_CAP + INJURY["nfl_skill_cap"]))
    return team_adj, notes, out_status


def _finalize(ctx: dict, injuries: pd.DataFrame) -> dict:
    """Everything that depends on injuries: team adjustments, predictions, statuses, game info."""
    team_adj, notes, out_status = injury_adjustments(ctx, injuries)
    preds = predict_slate(ctx["ratings"], ctx["slate"], ctx["cfg"], team_adj=team_adj, total_mult=ctx["total_mult"])
    if not preds.empty:
        preds["injury_notes"] = preds.apply(
            lambda r: "; ".join(notes.get(r["home"], []) + notes.get(r["away"], [])), axis=1)
    players = ctx["base_players"].copy()
    players["status"] = players["name_norm"].map(out_status).fillna("")
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
    ctx.update({"injuries": injuries, "team_adj": team_adj, "out_status": out_status, "predictions": preds,
                "players": players, "game_info": game_info})
    return ctx


def apply_overrides(ctx: dict, overrides: dict) -> dict:
    """overrides: {name_norm: (display name, status)} from manual clicks or news.
    status 'out'/'doubtful'/'questionable' or 'active' (clears an injury). Returns a NEW ctx."""
    if not overrides:
        return ctx
    rows = [{"name_norm": k, "name": v[0], "status": v[1], "team": "", "source": "manual"}
            for k, v in overrides.items()]
    inj = pd.concat([ctx["base_injuries"], pd.DataFrame(rows)], ignore_index=True)
    inj = inj.drop_duplicates("name_norm", keep="last")
    new = dict(ctx)
    new["overrides"] = dict(overrides)
    return _finalize(new, inj)


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
    res["offered"] = market_offered("nfl", stat, res, row["position"])
    return res


def all_projections(ctx: dict, stats: list[str], include_out: bool = False, markets_only: bool = True) -> list[dict]:
    """markets_only: drop player/stat combos books don't realistically post (see engine.props.MARKET_MIN)."""
    pos_stats = {"QB": {"passing_yards", "rushing_yards", "passing_tds", "anytime_td", "rushing_tds", "dk_points",
                        "fd_points"},
                 "RB": {"rushing_yards", "receiving_yards", "receptions", "anytime_td", "rushing_tds", "receiving_tds",
                        "dk_points", "fd_points"},
                 "WR": {"receiving_yards", "receptions", "anytime_td", "receiving_tds", "dk_points", "fd_points",
                        "rushing_yards"},
                 "TE": {"receiving_yards", "receptions", "anytime_td", "receiving_tds", "dk_points", "fd_points"}}
    out = []
    for _, p in ctx["players"].iterrows():
        if not include_out and p["status"] and espn.status_weight(p["status"]) >= 0.75:
            continue
        for s in stats:
            if s in pos_stats.get(p["position"], set()):
                r = project_player(ctx, p["player_id"], s)
                if r and (r["offered"] or not markets_only):
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
