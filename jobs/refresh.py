"""Scheduled job: log today's pre-game predictions and grade finished games.

    python -m jobs.refresh                 # all sports
    python -m jobs.refresh nba ncaab       # some sports

Run by .github/workflows/refresh.yml twice a day; the workflow commits
predictions/*.csv back to the repo, which is what the Track Record page reads.
(Streamlit Cloud's disk is wiped on restart, so the repo is the database for now.)
"""
from __future__ import annotations

import sys
import traceback

import pandas as pd

from engine import tracking
from engine.common import Diagnostics, now_et
from engine.config import get_config
from engine.market import merge_game_lines
from engine.odds_api import OddsClient, get_key


def _with_odds(preds, sport, diag):
    key = get_key()
    if not key or preds is None or preds.empty:
        return preds
    cfg = get_config(sport)
    lines = OddsClient(key, diag).game_lines(cfg.odds_api_key)
    preds, n = merge_game_lines(preds, lines, cfg)
    print(f"  {sport}: odds api lines matched for {n} games")
    return preds


def espn_results(path: str, dates: list[str], sport: str, diag: Diagnostics, **params) -> pd.DataFrame:
    from data import espn
    out = []
    for d in dates:
        sb = espn.scoreboard(path, diag, f"{sport} results {d}", dates=d.replace("-", ""), **params)
        if not sb.empty:
            sb = sb[sb["state"] == "post"]
            out.append(pd.DataFrame({"sport": sport, "game_id": sb["game_id"],
                                     "home_pts": sb["home_score"], "away_pts": sb["away_score"]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def ungraded_dates(sport: str, days_back: int = 10) -> list[str]:
    log = tracking.load_log()
    if log.empty:
        return []
    log = log[log["sport"] == sport]
    done = set()
    if tracking.RESULTS_PATH.exists():
        r = pd.read_csv(tracking.RESULTS_PATH, dtype={"game_id": str})
        done = set(r.loc[r["sport"] == sport, "game_id"])
    pending = log[~log["game_id"].isin(done)]
    d = pd.to_datetime(pending["date"], utc=True, errors="coerce").dt.tz_convert("America/New_York").dt.date
    today = now_et().date()
    return sorted({x.isoformat() for x in d.dropna() if 0 < (today - x).days <= days_back})


def run(sport: str, now) -> None:
    diag = Diagnostics()
    date_str = now.date().isoformat()
    if sport == "nfl":
        from data import nfl
        ctx = nfl.build_context(diag)
        g = ctx["games"].dropna(subset=["home_pts"])
        tracking.save_results(pd.DataFrame({"sport": "nfl", "game_id": g["game_id"],
                                            "home_pts": g["home_pts"], "away_pts": g["away_pts"]}))
    elif sport == "nba":
        from data import nba
        ctx = nba.build_context(date_str, diag)
        tracking.save_results(espn_results("basketball/nba", ungraded_dates("nba"), "nba", diag))
    elif sport == "ncaab":
        from data import college
        ctx = college.build_ncaab_context(date_str, diag)
        tracking.save_results(espn_results("basketball/mens-college-basketball", ungraded_dates("ncaab"),
                                           "ncaab", diag, groups=50, limit=500))
    elif sport == "ncaaf":
        from data import college
        ctx = college.build_ncaaf_context(diag)
        tracking.save_results(espn_results("football/college-football", ungraded_dates("ncaaf"),
                                           "ncaaf", diag, groups=80, limit=300))
    else:
        raise ValueError(sport)
    preds = _with_odds(ctx["predictions"], sport, diag)
    n = tracking.log_predictions(preds, sport, pd.Timestamp(now))
    print(f"  {sport}: logged {n} predictions; {len(diag.warnings)} warnings")
    for w in diag.warnings:
        print(f"    ! {w}")


def main(argv=None):
    sports = (argv if argv is not None else sys.argv[1:]) or ["nfl", "nba", "ncaab", "ncaaf"]
    now = now_et()
    failed = []
    for s in sports:
        print(f"[{now:%Y-%m-%d %H:%M} ET] refreshing {s}")
        try:
            run(s, now)
        except Exception:  # one sport failing shouldn't stop the others
            traceback.print_exc()
            failed.append(s)
    g = tracking.graded()
    if not g.empty:
        for s, grp in g.groupby("sport"):
            print(f"  track record {s}: {tracking.summary(grp)}")
    if failed and len(failed) == len(sports):
        sys.exit(1)


if __name__ == "__main__":
    main()
