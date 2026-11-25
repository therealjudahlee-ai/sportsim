"""NBA data + context (nba_api with per-season ESPN/hoopR fallback, ESPN slate & injuries)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.common import Diagnostics, norm_name
from engine.config import INJURY, get_config
from engine.gamesim import predict_slate
from engine.props import NBA_PROP_STATS, market_offered, project_nba, shrink_matchup
from engine.ratings import RatingsModel

from . import espn

ABBR_FIX = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "WSH": "WAS", "UTAH": "UTA", "PHO": "PHX", "BRK": "BKN"}
STAT_COLS = ["MIN", "PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "FGA", "FTA", "OREB"]


def season_start_year(d: pd.Timestamp) -> int:
    return d.year if d.month >= 10 else d.year - 1


def _from_nba_api(start_year: int, diag: Diagnostics) -> pd.DataFrame | None:
    try:
        from nba_api.stats.endpoints import leaguegamelog
        szn = f"{start_year}-{str(start_year + 1)[-2:]}"
        df = leaguegamelog.LeagueGameLog(season=szn, player_or_team_abbreviation="P",
                                         season_type_all_star="Regular Season", timeout=15).get_data_frames()[0]
        if df.empty:
            return None
        df["IS_HOME"] = df["MATCHUP"].str.contains("vs.", regex=False)
        df["OPP_ABBR"] = df["MATCHUP"].str.split().str[-1]
        df["POS"] = ""
        return df
    except Exception as e:
        diag.warn(f"nba_api failed for {start_year}-{start_year + 1} (stats.nba.com often blocks cloud servers); "
                  "falling back to ESPN data", e)
        return None


def _from_espn_parquet(start_year: int, diag: Diagnostics) -> pd.DataFrame | None:
    yr = start_year + 1  # hoopR files are named by the season's END year
    url = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
           f"espn_nba_player_boxscores/player_box_{yr}.parquet")
    try:
        raw = pd.read_parquet(url)
    except Exception as e:
        diag.warn(f"ESPN NBA player box {yr} unavailable", e)
        return None
    if "season_type" in raw.columns:
        raw = raw[raw["season_type"] == 2]
    num = lambda c: pd.to_numeric(raw[c], errors="coerce").fillna(0) if c in raw.columns else 0
    df = pd.DataFrame({
        "PLAYER_ID": raw["athlete_id"].astype(str), "PLAYER_NAME": raw["athlete_display_name"],
        "TEAM_ABBREVIATION": raw["team_abbreviation"], "OPP_ABBR": raw["opponent_team_abbreviation"],
        "GAME_ID": raw["game_id"].astype(str), "GAME_DATE": raw["game_date"],
        "MIN": num("minutes"), "PTS": num("points"), "REB": num("rebounds"), "AST": num("assists"),
        "FG3M": num("three_point_field_goals_made"), "STL": num("steals"), "BLK": num("blocks"),
        "TOV": num("turnovers"), "FGA": num("field_goals_attempted"), "FTA": num("free_throws_attempted"),
        "OREB": num("offensive_rebounds"),
        "IS_HOME": raw["home_away"].astype(str).str.lower().eq("home"),
        "POS": raw["athlete_position_abbreviation"] if "athlete_position_abbreviation" in raw.columns else "",
    })
    if "did_not_play" in raw.columns:
        df = df[~raw["did_not_play"].fillna(False).astype(bool)]
    return df


def load_logs(start_years: list[int], diag: Diagnostics) -> pd.DataFrame:
    dfs = []
    for sy in start_years:  # fall back PER SEASON (old code only fell back if both failed)
        df = _from_nba_api(sy, diag)
        src = "nba_api"
        if df is None:
            df, src = _from_espn_parquet(sy, diag), "espn"
        if df is None:
            continue
        df = df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "OPP_ABBR", "GAME_ID", "GAME_DATE",
                 "IS_HOME", "POS"] + STAT_COLS].copy()
        df["SEASON"] = sy + 1
        df["SOURCE"] = src
        dfs.append(df)
        diag.note(f"NBA {sy}-{sy + 1}: {len(df)} player-games from {src}")
    if not dfs:
        raise RuntimeError("No NBA game logs could be loaded from any source.")
    logs = pd.concat(dfs, ignore_index=True)
    logs["PLAYER_ID"] = logs["PLAYER_ID"].astype(str)
    logs["GAME_ID"] = logs["GAME_ID"].astype(str)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"]).dt.tz_localize(None).dt.normalize()
    for c in STAT_COLS:
        logs[c] = pd.to_numeric(logs[c], errors="coerce").fillna(0)
    logs["TEAM_ABBREVIATION"] = logs["TEAM_ABBREVIATION"].replace(ABBR_FIX)
    logs["OPP_ABBR"] = logs["OPP_ABBR"].replace(ABBR_FIX)
    logs["PRA"] = logs["PTS"] + logs["REB"] + logs["AST"]
    logs["PR"] = logs["PTS"] + logs["REB"]
    logs["PA"] = logs["PTS"] + logs["AST"]
    logs["RA"] = logs["REB"] + logs["AST"]
    dd = sum((logs[c] >= 10).astype(int) for c in ["PTS", "REB", "AST", "STL", "BLK"])
    logs["DK_PTS"] = (logs["PTS"] + 0.5 * logs["FG3M"] + 1.25 * logs["REB"] + 1.5 * logs["AST"] + 2 * logs["STL"]
                      + 2 * logs["BLK"] - 0.5 * logs["TOV"] + np.where(dd >= 2, 1.5, 0) + np.where(dd >= 3, 3, 0))
    logs["FD_PTS"] = (logs["PTS"] + 1.2 * logs["REB"] + 1.5 * logs["AST"] + 3 * logs["STL"] + 3 * logs["BLK"]
                      - logs["TOV"])
    logs["name_norm"] = logs["PLAYER_NAME"].map(norm_name)
    return logs.sort_values("GAME_DATE").reset_index(drop=True)


def games_from_logs(logs: pd.DataFrame) -> pd.DataFrame:
    t = (logs.groupby(["GAME_ID", "TEAM_ABBREVIATION"])
         .agg(PTS=("PTS", "sum"), FGA=("FGA", "sum"), FTA=("FTA", "sum"), OREB=("OREB", "sum"), TOV=("TOV", "sum"),
              IS_HOME=("IS_HOME", "first"), DATE=("GAME_DATE", "first"), SEASON=("SEASON", "first"))
         .reset_index())
    t["POSS"] = t["FGA"] + 0.44 * t["FTA"] - t["OREB"] + t["TOV"]
    h = t[t["IS_HOME"]].set_index("GAME_ID")
    a = t[~t["IS_HOME"]].set_index("GAME_ID")
    j = h.join(a, lsuffix="_h", rsuffix="_a", how="inner")
    return pd.DataFrame({
        "game_id": j.index, "date": j["DATE_h"].values, "season": j["SEASON_h"].astype(int).values,
        "home": j["TEAM_ABBREVIATION_h"].values, "away": j["TEAM_ABBREVIATION_a"].values,
        "home_pts": j["PTS_h"].values, "away_pts": j["PTS_a"].values, "neutral": False,
        "poss": ((j["POSS_h"] + j["POSS_a"]) / 2).values,
    }).reset_index(drop=True)


def team_last_game(games: pd.DataFrame) -> pd.Series:
    allg = pd.concat([games[["home", "date"]].rename(columns={"home": "team"}),
                      games[["away", "date"]].rename(columns={"away": "team"})])
    return allg.groupby("team")["date"].max()


def build_context(date_str: str, diag: Diagnostics | None = None) -> dict:
    diag = diag or Diagnostics()
    cfg = get_config("nba")
    target = pd.Timestamp(date_str)
    sy = season_start_year(target)
    season = sy + 1
    logs = load_logs([sy - 1, sy], diag)
    logs = logs[logs["GAME_DATE"] < target]
    games = games_from_logs(logs)

    # ---------------- slate
    sb = espn.scoreboard("basketball/nba", diag, f"NBA scoreboard {date_str}", dates=date_str.replace("-", ""))
    if not sb.empty:
        sb["home"] = sb["home_abbr"].replace(ABBR_FIX)
        sb["away"] = sb["away_abbr"].replace(ABBR_FIX)
        last = team_last_game(games)
        for side in ("home", "away"):
            sb[f"{side}_rest"] = sb[side].map(lambda t: (target - last[t]).days if t in last.index else np.nan)
    else:
        diag.warn(f"No NBA games found for {date_str}")

    # ---------------- injuries (applied in _finalize so news/overrides can re-run it)
    inj = espn.injuries("basketball/nba", diag)
    cur = logs[logs["SEASON"] == season]
    recent = cur.groupby("name_norm").tail(15)
    ppg = recent.groupby("name_norm").agg(dk=("DK_PTS", "mean"), mins=("MIN", "mean"),
                                          team=("TEAM_ABBREVIATION", "last"), name=("PLAYER_NAME", "last"))

    # ---------------- ratings
    model = RatingsModel(games, cfg)
    ratings = model.ratings_as_of(target, season)
    slate = sb[["game_id", "date", "home", "away", "home_name", "away_name", "neutral", "home_spread", "total_line",
                "home_ml", "away_ml", "home_rest", "away_rest", "state"]].copy() if not sb.empty else pd.DataFrame()

    # ---------------- matchup multipliers + league per-minute rates
    def_lookup = {}
    recent_logs = logs[logs["GAME_DATE"] >= logs["GAME_DATE"].max() - pd.Timedelta(days=120)]
    for stat in NBA_PROP_STATS + ["FD_PTS"]:
        gt = recent_logs.groupby(["OPP_ABBR", "GAME_ID", "GAME_DATE"])[stat].sum().reset_index().sort_values("GAME_DATE")
        lg = gt[stat].mean()
        if not lg:
            continue
        for opp, grp in gt.groupby("OPP_ABBR"):
            rec = grp.tail(20)
            w = np.exp(np.linspace(-1.2, 0, len(rec)))
            def_lookup[(opp, stat)] = (float(np.sum(w * rec[stat]) / w.sum() / lg), len(rec))
    played = recent_logs[recent_logs["MIN"] >= 10]
    league_rate = {s: float(played[s].sum() / max(played["MIN"].sum(), 1)) for s in NBA_PROP_STATS + ["FD_PTS"]}

    slate_teams = set(slate["home"]) | set(slate["away"]) if not slate.empty else set()
    last_seen = logs[logs["GAME_DATE"] >= target - pd.Timedelta(days=21)]
    players = (last_seen.groupby("PLAYER_ID")
               .agg(name=("PLAYER_NAME", "last"), name_norm=("name_norm", "last"), team=("TEAM_ABBREVIATION", "last"),
                    mins=("MIN", "mean"), dk=("DK_PTS", "mean"), pos=("POS", "last"))
               .reset_index().rename(columns={"PLAYER_ID": "player_id"}))
    players = players[players["team"].isin(slate_teams) & (players["mins"] >= 12)]
    players = players.sort_values("dk", ascending=False).reset_index(drop=True)

    ctx = {
        "sport": "nba", "season": season, "date": date_str, "as_of": target, "cfg": cfg,
        "games": games, "slate": slate, "ratings": ratings, "logs": logs, "ppg": ppg,
        "base_players": players, "base_injuries": inj, "def_lookup": def_lookup, "league_rate": league_rate,
        "overrides": {}, "diag": diag.to_dict(),
    }
    _finalize(ctx, inj)
    preds = ctx["predictions"]
    if not preds.empty and (~preds["known"]).any():
        diag.warn(f"{(~preds['known']).sum()} NBA games include a team with no rating "
                  f"(check abbreviations: {sorted(set(preds.loc[~preds['known'], 'home']) | set(preds.loc[~preds['known'], 'away']))})")
    diag.note(f"NBA {date_str}: {len(slate)} games, {len(players)} active rotation players")
    ctx["diag"] = diag.to_dict()
    return ctx


def injury_adjustments(ctx: dict, inj: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    """(team_adj, usage_bump, notes, out_players) from an injury table."""
    ppg = ctx["ppg"]
    team_adj, usage, notes, out_players = {}, {}, {}, {}
    for _, r in inj.iterrows():
        if str(r["status"]).lower() == "active":
            continue
        w = espn.status_weight(r["status"])
        if w <= 0 or r["name_norm"] not in ppg.index:
            continue
        p = ppg.loc[r["name_norm"]]
        out_players[r["name_norm"]] = r["status"]
        over = max(p["dk"] - INJURY["nba_replacement_dkpts"], 0.0)
        pts = min(over * INJURY["nba_per_dkpt_over_repl"], INJURY["nba_player_cap"]) * w
        if pts > 0.05:
            team_adj[p["team"]] = team_adj.get(p["team"], 0.0) - pts
            usage[p["team"]] = usage.get(p["team"], 0.0) + over * w
            notes.setdefault(p["team"], []).append(f"{p['name']} ({r['status']}, -{pts:.1f})")
    usage = {t: float(min(1.10, 1.0 + 0.004 * v)) for t, v in usage.items()}
    return team_adj, usage, notes, out_players


def _finalize(ctx: dict, inj: pd.DataFrame) -> dict:
    team_adj, usage, notes, out_players = injury_adjustments(ctx, inj)
    ratings, slate = ctx["ratings"], ctx["slate"]
    preds = predict_slate(ratings, slate, ctx["cfg"], team_adj=team_adj) if not slate.empty else pd.DataFrame()
    if not preds.empty:
        preds["injury_notes"] = preds.apply(
            lambda r: "; ".join(notes.get(r["home"], []) + notes.get(r["away"], [])), axis=1)
    game_info = {}
    for _, p in preds.iterrows():
        proj = ratings.project(p["home"], p["away"], bool(p["neutral"]))
        pace_mult = proj["poss"] / ratings.pace_mu if ratings.pace_mu else 1.0
        for side, other in (("home", "away"), ("away", "home")):
            rest = p.get(f"{side}_rest", np.nan)
            game_info[p[side]] = {
                "opp": p[other], "game": f"{p['away']} @ {p['home']}", "game_id": p["game_id"],
                "is_home": side == "home", "pace_mult": float(np.clip(pace_mult, 0.92, 1.08)),
                "blowout": p["blowout"], "b2b": bool(pd.notna(rest) and rest <= 1),
                "team_pts": p[f"model_{side}_pts"],
            }
    players = ctx["base_players"].copy()
    players["status"] = players["name_norm"].map(out_players).fillna("")
    ctx.update({"injuries": inj, "team_adj": team_adj, "usage_bump": usage, "predictions": preds,
                "game_info": game_info, "players": players})
    return ctx


def apply_overrides(ctx: dict, overrides: dict) -> dict:
    """overrides: {name_norm: (display name, status)}; status 'active' clears an injury. Returns a NEW ctx."""
    if not overrides:
        return ctx
    rows = [{"name": v[0], "name_norm": k, "status": v[1], "team_abbr": "", "team_name": "", "comment": "manual"}
            for k, v in overrides.items()]
    inj = pd.concat([ctx["base_injuries"], pd.DataFrame(rows)], ignore_index=True).drop_duplicates("name_norm", keep="last")
    new = dict(ctx)
    new["overrides"] = dict(overrides)
    return _finalize(new, inj)


def project_player(ctx: dict, player_id, stat: str, bump: float = 1.0) -> dict | None:
    logs = ctx["logs"][ctx["logs"]["PLAYER_ID"] == str(player_id)]
    if logs.empty:
        return None
    last = logs.iloc[-1]
    team = last["TEAM_ABBREVIATION"]
    gi = ctx["game_info"].get(team, {})
    raw, n = ctx["def_lookup"].get((gi.get("opp"), stat), (1.0, 0))
    dm = shrink_matchup(raw, n, k=12)
    mins_recent = logs["MIN"].tail(5).mean()
    minutes_mult = 0.95 if gi.get("blowout", 0) >= 0.34 and mins_recent >= 28 else 1.0
    fatigue = 0.97 if gi.get("b2b") else 1.0
    status_row = ctx["players"][ctx["players"]["player_id"] == str(player_id)]
    is_out = not status_row.empty and espn.status_weight(status_row.iloc[0]["status"]) >= 0.75
    team_bump = 1.0 if is_out else ctx["usage_bump"].get(team, 1.0)
    res = project_nba(logs, stat, ctx["league_rate"], def_mult=dm, pace_mult=gi.get("pace_mult", 1.0),
                      fatigue_mult=fatigue, minutes_mult=minutes_mult, bump=bump * team_bump,
                      home=gi.get("is_home"))
    if res is None:
        return None
    notes = res["notes"]
    if gi.get("b2b"):
        notes.append("back-to-back")
    if minutes_mult < 1:
        notes.append(f"blowout risk {gi['blowout']:.0%}")
    if team_bump > 1.0:
        notes.append(f"teammate(s) out, usage x{team_bump:.2f}")
    res.update({"player": last["PLAYER_NAME"], "player_id": str(player_id), "player_norm": last["name_norm"],
                "position": last.get("POS", ""), "team": team, "opp": gi.get("opp", ""), "game": gi.get("game", ""),
                "stat": stat, "def_mult": dm, "status": status_row.iloc[0]["status"] if not status_row.empty else "",
                "hist": logs[stat].tail(20).tolist()})
    res["offered"] = market_offered("nba", stat, res)
    return res


def all_projections(ctx: dict, stats: list[str], markets_only: bool = True) -> list[dict]:
    """markets_only: drop player/stat combos books don't realistically post (see engine.props.MARKET_MIN)."""
    out = []
    for _, p in ctx["players"].iterrows():
        if p["status"] and espn.status_weight(p["status"]) >= 0.75:
            continue
        for s in stats:
            r = project_player(ctx, p["player_id"], s)
            if r and (r["offered"] or not markets_only):
                out.append(r)
    return out


def dfs_pool(ctx: dict, site: str, salaries: pd.DataFrame) -> pd.DataFrame:
    """NBA DFS requires real salaries (they carry the position eligibility)."""
    if salaries is None or salaries.empty:
        return pd.DataFrame()
    stat = "DK_PTS" if site == "DK" else "FD_PTS"
    sal = salaries.drop_duplicates("name_norm").set_index("name_norm")
    rows = []
    for p in all_projections(ctx, [stat]):
        if p["player_norm"] not in sal.index:
            continue
        s = sal.loc[p["player_norm"]]
        rows.append({"name": p["player"], "id": s["id"], "positions": s["positions"], "team": p["team"],
                     "opp": p["opp"], "game": p["game"], "salary": int(s["salary"]), "proj": p["mean"],
                     "ceiling": p["dist"].quantile(0.85)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["value"] = df["proj"] / df["salary"] * 1000
    df["own_est"] = np.clip((df["value"] - df["value"].median()) * 8 + 10, 1, 45)
    df["leverage"] = df["ceiling"] * (1 - df["own_est"] / 100 * 0.45)
    return df.sort_values("proj", ascending=False).reset_index(drop=True)
