"""Men's college basketball + college football contexts.

Teams are keyed by ESPN team ID wherever possible (both the sportsdataverse
history files and the ESPN scoreboard carry it), which avoids the old
"Alabama" vs "Alabama Crimson Tide" name mismatch that silently gave teams
the default rating.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.common import Diagnostics, norm_team, now_et, pick_col
from engine.config import get_config
from engine.gamesim import predict_slate
from engine.ratings import RatingsModel

from . import espn

SDV = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"


# =============================================================================
# Shared
# =============================================================================
def _attach_rest(slate: pd.DataFrame, games: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    allg = pd.concat([games[["home", "date"]].rename(columns={"home": "team"}),
                      games[["away", "date"]].rename(columns={"away": "team"})])
    last = allg[allg["date"] < target].groupby("team")["date"].max()
    for side in ("home", "away"):
        slate[f"{side}_rest"] = slate[side].map(lambda t: (target - last[t]).days if t in last.index else np.nan)
    return slate


def _resolve_keys(slate: pd.DataFrame, games: pd.DataFrame, names: dict, diag: Diagnostics, label: str):
    """Use ESPN IDs if the history uses them; else match on normalized school names."""
    known = set(games["home"]) | set(games["away"])
    id_hits = slate["home_id"].isin(known).mean() if len(slate) else 0
    if id_hits >= 0.5:
        slate["home"], slate["away"] = slate["home_id"], slate["away_id"]
    else:
        by_name = {norm_team(v): k for k, v in names.items()}
        by_name.update({norm_team(k): k for k in known})
        def look(loc, full):
            return by_name.get(norm_team(loc)) or by_name.get(norm_team(full)) or f"?{full}"
        slate["home"] = [look(l, f) for l, f in zip(slate["home_location"], slate["home_name"])]
        slate["away"] = [look(l, f) for l, f in zip(slate["away_location"], slate["away_name"])]
        diag.note(f"{label}: matched teams by name (history has no ESPN IDs)")
    miss = sorted({n for k, n in zip(slate["home"], slate["home_name"]) if k not in known} |
                  {n for k, n in zip(slate["away"], slate["away_name"]) if k not in known})
    if miss:
        diag.warn(f"{label}: {len(miss)} teams on the slate have no history (rated as 'other'): "
                  f"{', '.join(miss[:12])}{'…' if len(miss) > 12 else ''}")
    return slate


SLATE_COLS = ["game_id", "date", "home", "away", "home_name", "away_name", "neutral", "home_spread",
              "total_line", "home_ml", "away_ml", "home_rest", "away_rest", "state"]


# =============================================================================
# Men's college basketball
# =============================================================================
def ncaab_season(d: pd.Timestamp) -> int:
    return d.year + 1 if d.month >= 10 else d.year   # named by END year


def load_ncaab_games(seasons: list[int], diag: Diagnostics) -> tuple[pd.DataFrame, dict]:
    frames = []
    for yr in seasons:
        try:
            frames.append(pd.read_parquet(f"{SDV}/espn_mens_college_basketball_team_boxscores/team_box_{yr}.parquet"))
        except Exception as e:
            diag.warn(f"CBB team box {yr} unavailable", e)
    if not frames:
        raise RuntimeError("No college basketball box scores could be loaded.")
    df = pd.concat(frames, ignore_index=True)
    c_tid = pick_col(df, "team_id")
    c_oid = pick_col(df, "opponent_team_id")
    c_tname = pick_col(df, "team_display_name", "team_short_display_name", required=True)
    c_oname = pick_col(df, "opponent_team_display_name", "opponent_team_short_display_name", required=True)
    c_ha = pick_col(df, "team_home_away")
    c_pts = pick_col(df, "team_score", "points", required=True)
    c_opts = pick_col(df, "opponent_team_score", required=True)
    c_neu = pick_col(df, "neutral_site", "game_neutral_site")
    use_ids = c_tid is not None and c_oid is not None
    if not use_ids:
        diag.warn("CBB box scores have no team IDs; keying teams by name")
    n = lambda c, d: pd.to_numeric(df[c], errors="coerce").fillna(d) if c in df.columns else d
    df["_poss"] = np.clip(n("field_goals_attempted", 58) + 0.44 * n("free_throws_attempted", 18)
                          - n("offensive_rebounds", 10) + n("turnovers", 12), 50, 95)
    df["_t"] = df[c_tid].astype(str) if use_ids else df[c_tname].astype(str)
    df["_o"] = df[c_oid].astype(str) if use_ids else df[c_oname].astype(str)
    names = dict(zip(df["_t"], df[c_tname]))
    names.update({k: v for k, v in zip(df["_o"], df[c_oname]) if k not in names})
    poss = df.groupby("game_id")["_poss"].mean()

    if c_ha:
        home_rows = df[df[c_ha].astype(str).str.lower() == "home"]
        neutral = home_rows[c_neu].fillna(False).astype(bool) if c_neu else pd.Series(False, index=home_rows.index)
    else:
        diag.warn("CBB box scores have no home/away column; treating all games as neutral")
        home_rows = df.sort_values("_t").drop_duplicates("game_id")
        neutral = pd.Series(True, index=home_rows.index)
    games = pd.DataFrame({
        "game_id": home_rows["game_id"].astype(str).values,
        "date": pd.to_datetime(home_rows["game_date"]).dt.tz_localize(None).values,
        "season": home_rows["season"].astype(int).values,
        "home": home_rows["_t"].values, "away": home_rows["_o"].values,
        "home_pts": pd.to_numeric(home_rows[c_pts], errors="coerce").values,
        "away_pts": pd.to_numeric(home_rows[c_opts], errors="coerce").values,
        "neutral": neutral.values,
        "poss": home_rows["game_id"].map(poss).values,
    })
    games = games.dropna(subset=["home_pts", "away_pts"])
    games = games[(games["home_pts"] > 0) & (games["away_pts"] > 0)]
    diag.note(f"CBB history: {len(games)} games across seasons {seasons}")
    return games.sort_values("date").reset_index(drop=True), names


def build_ncaab_context(date_str: str, diag: Diagnostics | None = None) -> dict:
    diag = diag or Diagnostics()
    cfg = get_config("ncaab")
    target = pd.Timestamp(date_str)
    season = ncaab_season(target)
    games, names = load_ncaab_games([season - 1, season], diag)
    model = RatingsModel(games, cfg)
    ratings = model.ratings_as_of(target, season)

    sb = espn.scoreboard("basketball/mens-college-basketball", diag, f"CBB scoreboard {date_str}",
                         dates=date_str.replace("-", ""), groups=50, limit=500)
    if sb.empty:
        diag.warn(f"No college basketball games found for {date_str}")
        preds = pd.DataFrame()
    else:
        sb = _resolve_keys(sb, games, names, diag, "CBB")
        sb = _attach_rest(sb, games, target)
        preds = predict_slate(ratings, sb[SLATE_COLS], cfg)
    return {"sport": "ncaab", "season": season, "date": date_str, "as_of": target, "games": games,
            "names": names, "ratings": ratings, "predictions": preds, "diag": diag.to_dict()}


# =============================================================================
# College football
# =============================================================================
def ncaaf_season(d: pd.Timestamp) -> int:
    return d.year if d.month >= 3 else d.year - 1


def _ncaaf_from_parquet(seasons, diag) -> tuple[pd.DataFrame, dict] | None:
    frames = []
    for yr in seasons:
        try:
            frames.append(pd.read_parquet(f"{SDV}/espn_cfb_schedules/cfb_schedules_{yr}.parquet"))
        except Exception as e:
            diag.warn(f"CFB schedule parquet {yr} unavailable", e)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    c = {
        "gid": pick_col(df, "game_id", "id"), "date": pick_col(df, "start_date", "game_date", "date"),
        "hid": pick_col(df, "home_id", "home_team_id"), "aid": pick_col(df, "away_id", "away_team_id"),
        "hname": pick_col(df, "home_display_name", "home_team_full", "home_team", "home_name", "home_location"),
        "aname": pick_col(df, "away_display_name", "away_team_full", "away_team", "away_name", "away_location"),
        "hpts": pick_col(df, "home_points", "home_score"), "apts": pick_col(df, "away_points", "away_score"),
        "neu": pick_col(df, "neutral_site", "neutral"), "season": pick_col(df, "season"),
    }
    needed = ["gid", "date", "hname", "aname", "hpts", "apts", "season"]
    if any(c[k] is None for k in needed):
        diag.warn(f"CFB parquet columns not recognized (have: {list(df.columns)[:30]}); using ESPN fallback")
        return None
    diag.note(f"CFB parquet columns used: { {k: v for k, v in c.items() if v} }")
    use_ids = c["hid"] is not None and c["aid"] is not None
    home = df[c["hid"]].astype(str).str.replace(r"\.0$", "", regex=True) if use_ids else df[c["hname"]].astype(str)
    away = df[c["aid"]].astype(str).str.replace(r"\.0$", "", regex=True) if use_ids else df[c["aname"]].astype(str)
    games = pd.DataFrame({
        "game_id": df[c["gid"]].astype(str), "date": pd.to_datetime(df[c["date"]], utc=True, errors="coerce")
        .dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize(),
        "season": pd.to_numeric(df[c["season"]], errors="coerce"), "home": home, "away": away,
        "home_pts": pd.to_numeric(df[c["hpts"]], errors="coerce"),
        "away_pts": pd.to_numeric(df[c["apts"]], errors="coerce"),
        "neutral": df[c["neu"]].fillna(False).astype(bool) if c["neu"] else False,
    })
    names = dict(zip(home, df[c["hname"]]))
    names.update(dict(zip(away, df[c["aname"]])))
    return games, names


def _ncaaf_from_espn(seasons, diag) -> tuple[pd.DataFrame, dict]:
    """Fallback: walk ESPN's scoreboard week by week (~16 calls per season)."""
    frames = []
    for yr in seasons:
        for st_type, weeks in ((2, range(1, 17)), (3, range(1, 2))):
            for wk in weeks:
                sb = espn.scoreboard("football/college-football", diag, f"CFB {yr} week {wk}",
                                     dates=yr, week=wk, seasontype=st_type, groups=80, limit=300)
                if not sb.empty:
                    sb["season"] = yr
                    frames.append(sb)
    if not frames:
        raise RuntimeError("No college football history could be loaded.")
    sb = pd.concat(frames, ignore_index=True).drop_duplicates("game_id")
    sb = sb[sb["state"] == "post"]
    games = pd.DataFrame({
        "game_id": sb["game_id"], "date": sb["date"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize(),
        "season": sb["season"], "home": sb["home_id"], "away": sb["away_id"],
        "home_pts": sb["home_score"], "away_pts": sb["away_score"], "neutral": sb["neutral"],
    })
    names = dict(zip(sb["home_id"], sb["home_name"]))
    names.update(dict(zip(sb["away_id"], sb["away_name"])))
    return games, names


def load_ncaaf_games(seasons: list[int], diag: Diagnostics) -> tuple[pd.DataFrame, dict]:
    res = _ncaaf_from_parquet(seasons, diag)
    if res is None or res[0]["home_pts"].notna().sum() < 100:
        res = _ncaaf_from_espn(seasons, diag)
    games, names = res
    games = games.dropna(subset=["date", "season"])
    games["season"] = games["season"].astype(int)
    diag.note(f"CFB history: {games['home_pts'].notna().sum()} completed games")
    return games.sort_values("date").reset_index(drop=True), names


def build_ncaaf_context(diag: Diagnostics | None = None) -> dict:
    diag = diag or Diagnostics()
    cfg = get_config("ncaaf")
    today = pd.Timestamp(now_et().date())
    season = ncaaf_season(today)
    games, names = load_ncaaf_games([season - 1, season], diag)
    ratings = RatingsModel(games, cfg).ratings_as_of(today, season)

    sb = espn.scoreboard("football/college-football", diag, "CFB scoreboard", groups=80, limit=300)
    if sb.empty:
        diag.warn("No upcoming FBS games on ESPN's scoreboard")
        preds = pd.DataFrame()
    else:
        sb = sb[sb["state"] != "post"].copy()
        sb = _resolve_keys(sb, games, names, diag, "CFB")
        sb = _attach_rest(sb, games, today)
        preds = predict_slate(ratings, sb[SLATE_COLS], cfg) if not sb.empty else pd.DataFrame()
    return {"sport": "ncaaf", "season": season, "as_of": today, "games": games, "names": names,
            "ratings": ratings, "predictions": preds, "diag": diag.to_dict()}
