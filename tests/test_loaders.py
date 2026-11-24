import sys

import numpy as np
import pandas as pd
import pytest

from data import espn
from tests import fakes


def test_parse_odds_sign():
    comp = {"odds": [{"details": "KC -3.5", "overUnder": 47.5}]}
    assert espn.parse_odds(comp, "KC", "BUF")["home_spread"] == -3.5
    assert espn.parse_odds(comp, "BUF", "KC")["home_spread"] == 3.5
    assert espn.parse_odds({"odds": [{"details": "EVEN"}]}, "A", "B")["home_spread"] == 0.0
    comp2 = {"odds": [{"details": "", "spread": 4.5, "awayTeamOdds": {"favorite": True}}]}
    assert espn.parse_odds(comp2, "A", "B")["home_spread"] == 4.5
    assert np.isnan(espn.parse_odds({}, "A", "B")["home_spread"])


def test_nfl_context(monkeypatch):
    monkeypatch.setitem(sys.modules, "nflreadpy", fakes.fake_nflreadpy())
    monkeypatch.setattr(espn, "get_json", fakes.fake_espn_json("nfl"))
    from data import nfl
    ctx = nfl.build_context()
    preds = ctx["predictions"]
    assert len(preds) == 4 and preds["known"].all()
    # nflverse spread_line=+3 (home favored) -> book line -3 for home
    assert (preds["home_spread"] == -3.0).all()
    # KC's starting QB is listed Out -> KC gets a negative adjustment
    assert ctx["team_adj"].get("KC", 0) < -3
    assert ctx["players"]["position"].isin(["QB", "RB", "WR", "TE"]).all()
    pid = ctx["players"].iloc[0]["player_id"]
    proj = nfl.project_player(ctx, pid, "dk_points")
    assert proj and proj["mean"] > 0
    projs = nfl.all_projections(ctx, ["receiving_yards", "anytime_td"])
    assert len(projs) > 10
    assert all("KC QB0" not in p["player"] for p in projs)  # out players excluded
    pool = nfl.dfs_pool(ctx, "DK", None)
    assert {"QB", "RB", "WR", "TE"} <= {p[0] for p in pool["positions"]} and (pool["name"].str.endswith("DST")).any()
    from engine.dfs import optimize
    lus = optimize(pool, "nfl", "DK", num_lineups=2, stack="qb1")
    assert len(lus) == 2


def test_nba_context(monkeypatch):
    frames = fakes.fake_nba_parquet()
    monkeypatch.setitem(sys.modules, "nba_api", None)  # force nba_api import failure -> fallback
    monkeypatch.setattr(pd, "read_parquet", lambda url: frames[int(url.split("_")[-1].split(".")[0])])
    slate = [("BOS", "NY", "1", "2", "BOS -6.5"), ("GS", "SA", "3", "4", "SA -1.5")]
    monkeypatch.setattr(espn, "get_json", fakes.fake_espn_json("nba", slate))
    from data import nba
    ctx = nba.build_context("2026-03-15")
    p = ctx["predictions"]
    assert len(p) == 2 and p["known"].all(), ctx["diag"]
    assert p.iloc[1]["home_spread"] == 1.5      # SA (away) favored by 1.5
    assert set(p["home"]) == {"BOS", "GSW"}       # abbreviation fix applied
    assert "BOS" in ctx["team_adj"]               # 'P1 Star' listed out
    assert ctx["usage_bump"]["BOS"] > 1.0
    projs = nba.all_projections(ctx, ["PTS", "REB"])
    assert projs and all(p["player"] != "P1 Star" for p in projs)


def test_ncaab_context(monkeypatch):
    frames = fakes.fake_cbb_parquet()
    monkeypatch.setattr(pd, "read_parquet", lambda url: frames[int(url.split("_")[-1].split(".")[0])])
    slate = [("S1", "S2", "101", "102", "S1 -4.5"), ("S3", "S9", "103", "999", "S9 -2")]
    monkeypatch.setattr(espn, "get_json", fakes.fake_espn_json("cbb", slate))
    from data import college
    ctx = college.build_ncaab_context("2026-02-10")
    p = ctx["predictions"]
    assert len(p) == 2
    assert bool(p.iloc[0]["known"]) and not bool(p.iloc[1]["known"])   # team 999 has no history
    assert any("no history" in w for w in ctx["diag"]["warnings"])


@pytest.mark.parametrize("style", ["cfbd", "espn"])
def test_ncaaf_context(monkeypatch, style):
    from tests.synth import synth_games
    games, _ = synth_games(n_teams=30, seasons=(2025, 2026), games_per_team=12, seed=11, start_month=9)
    def frame(yr):
        g = games[games["season"] == yr]
        if style == "cfbd":
            return pd.DataFrame({"id": g["game_id"], "season": yr, "start_date": g["date"].dt.strftime("%Y-%m-%dT17:00:00.000Z"),
                                 "home_id": g["home"].str[1:].astype(int) + 200, "away_id": g["away"].str[1:].astype(int) + 200,
                                 "home_team": "U" + g["home"], "away_team": "U" + g["away"],
                                 "home_points": g["home_pts"], "away_points": g["away_pts"], "neutral_site": g["neutral"]})
        return pd.DataFrame({"game_id": g["game_id"], "season": yr, "game_date": g["date"],
                             "home_team_full": "U" + g["home"], "away_team_full": "U" + g["away"],
                             "home_score": g["home_pts"], "away_score": g["away_pts"]})
    monkeypatch.setattr(pd, "read_parquet", lambda url: frame(int(url.split("_")[-1].split(".")[0])))
    slate = [("UT01", "UT02", "201", "202", "UT01 -7"), ("UT03", "UT04", "203", "204", "UT04 -3")]
    monkeypatch.setattr(espn, "get_json", fakes.fake_espn_json("cfb", slate))
    from data import college
    monkeypatch.setattr(college, "now_et", lambda: pd.Timestamp("2026-11-24", tz="America/New_York"))
    ctx = college.build_ncaaf_context()
    p = ctx["predictions"]
    assert len(p) == 2 and p["known"].all(), ctx["diag"]
    assert p.iloc[0]["home_spread"] == -7 and p.iloc[1]["home_spread"] == 3
