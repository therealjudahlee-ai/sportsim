"""Fake data sources so loaders can be tested offline."""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd

from tests.synth import synth_games

NFL_TEAMS = ["KC", "BUF", "PHI", "DAL", "SF", "SEA", "DET", "GB"]


class _PL:  # mimics polars .to_pandas()
    def __init__(self, df):
        self.df = df

    def to_pandas(self):
        return self.df.copy()


def fake_nflreadpy(season=2026, week=12):
    rng = np.random.default_rng(0)
    sched_rows, stat_rows = [], []
    gid = 0
    for s in (season - 1, season):
        for wk in range(1, 19 if s < season else week + 1):
            order = rng.permutation(NFL_TEAMS)
            for i in range(0, 8, 2):
                h, a = order[i], order[i + 1]
                done = (s < season) or (wk < week)
                day = pd.Timestamp(f"{s}-09-07") + pd.Timedelta(weeks=wk - 1)
                sched_rows.append({
                    "game_id": f"{s}_{wk:02d}_{a}_{h}", "season": s, "game_type": "REG", "week": wk,
                    "gameday": day.strftime("%Y-%m-%d"), "weekday": "Sunday", "gametime": "13:00",
                    "home_team": h, "away_team": a,
                    "home_score": rng.integers(10, 38) if done else np.nan,
                    "away_score": rng.integers(10, 35) if done else np.nan,
                    "location": "Home", "spread_line": 3.0, "total_line": 45.5,
                    "home_moneyline": -150, "away_moneyline": 130, "roof": "outdoors",
                    "wind": 18.0 if (wk == week and i == 0) else 5.0,
                })
                if not done:
                    continue
                for team, opp in ((h, a), (a, h)):
                    for pos, k in (("QB", 1), ("RB", 1), ("WR", 2), ("TE", 1)):
                        for j in range(k):
                            att = rng.integers(28, 40) if pos == "QB" else 0
                            car = rng.integers(10, 20) if pos == "RB" else (rng.integers(2, 6) if pos == "QB" else 0)
                            tg = rng.integers(4, 11) if pos in ("WR", "TE") else (rng.integers(1, 5) if pos == "RB" else 0)
                            rec = rng.binomial(tg, 0.66) if tg else 0
                            stat_rows.append({
                                "player_id": f"{team}-{pos}{j}", "player_display_name": f"{team} {pos}{j} Jr.",
                                "position": pos, "team": team, "opponent_team": opp, "season": s, "week": wk,
                                "season_type": "REG", "attempts": att, "carries": car, "targets": tg,
                                "receptions": rec, "passing_yards": att * rng.uniform(6, 8) if att else 0,
                                "passing_tds": rng.poisson(1.6) if att else 0, "passing_interceptions": rng.poisson(0.7) if att else 0,
                                "rushing_yards": car * rng.uniform(3, 5.5), "rushing_tds": rng.poisson(0.35) if car else 0,
                                "receiving_yards": rec * rng.uniform(8, 13), "receiving_tds": rng.poisson(0.3) if tg else 0,
                                "special_teams_tds": 0, "rushing_fumbles_lost": 0, "receiving_fumbles_lost": 0,
                            })
    sched, stats = pd.DataFrame(sched_rows), pd.DataFrame(stat_rows)
    mod = types.ModuleType("nflreadpy")
    mod.get_current_season = lambda: season
    mod.get_current_week = lambda: week
    mod.load_schedules = lambda seasons: _PL(sched[sched["season"].isin(seasons)])
    mod.load_player_stats = lambda seasons: _PL(stats[stats["season"].isin(seasons)])
    mod.load_rosters_weekly = lambda seasons: _PL(pd.DataFrame({
        "week": week, "full_name": stats["player_display_name"].unique(), "status": "ACT",
        "team": [n.split()[0] for n in stats["player_display_name"].unique()]}))
    mod.load_depth_charts = lambda seasons: _PL(pd.DataFrame({
        "dt": "2026-11-20", "pos_abb": "QB", "pos_rank": 1, "player_name": [f"{t} QB0" for t in NFL_TEAMS],
        "team": NFL_TEAMS}))
    mod.load_injuries = lambda seasons: _PL(pd.DataFrame({
        "week": [week], "full_name": ["KC QB0 Jr."], "report_status": ["Out"], "team": ["KC"]}))
    return mod


def fake_espn_json(sport_path_hint: str, slate: list[tuple] | None = None):
    """Returns a get_json replacement serving scoreboard + injuries."""
    slate = slate or []

    def get_json(url, diag=None, what="", timeout=10, **params):
        if url.endswith("/injuries"):
            return {"injuries": [{"displayName": "X", "injuries": [
                {"athlete": {"displayName": "P1 Star"}, "status": "Out", "type": {"description": "knee"}}]}]}
        if url.endswith("/scoreboard"):
            events = []
            for i, (h, a, hid, aid, details) in enumerate(slate):
                events.append({"id": f"e{i}", "date": "2026-11-24T19:00Z", "status": {"type": {"state": "pre"}},
                               "competitions": [{"neutralSite": False, "odds": [{"details": details, "overUnder": 150.5}],
                                                 "competitors": [
                                                     {"homeAway": "home", "team": {"id": hid, "abbreviation": h, "displayName": f"{h} Full", "location": h}},
                                                     {"homeAway": "away", "team": {"id": aid, "abbreviation": a, "displayName": f"{a} Full", "location": a}}]}]})
            return {"events": events}
        return None
    return get_json


def fake_nba_parquet(seasons_end=(2026, 2027)):
    """hoopR-style player box parquet frames keyed by url year."""
    games, _ = synth_games(n_teams=8, seasons=seasons_end, games_per_team=30, poss=True, seed=4)
    teams = ["BOS", "NY", "LAL", "DEN", "MIA", "GS", "SA", "PHX"]
    tmap = {f"T{i:02d}": t for i, t in enumerate(teams)}
    rng = np.random.default_rng(5)
    out = {}
    for s in seasons_end:
        rows = []
        for _, g in games[games["season"] == s].iterrows():
            for team, opp, pts, home in ((g["home"], g["away"], g["home_pts"], True), (g["away"], g["home"], g["away_pts"], False)):
                shares = rng.dirichlet(np.ones(8)) * pts
                for k in range(8):
                    rows.append({
                        "athlete_id": f"{tmap[team]}{k}", "athlete_display_name": f"P{k} {'Star' if k == 1 else tmap[team]}" if k == 1 and tmap[team] == "BOS" else f"P{k} {tmap[team]}",
                        "team_abbreviation": tmap[team], "opponent_team_abbreviation": tmap[opp],
                        "game_id": g["game_id"], "game_date": g["date"], "minutes": 36 - 3 * k + rng.normal(0, 2),
                        "points": round(shares[k]), "rebounds": rng.poisson(6), "assists": rng.poisson(4),
                        "three_point_field_goals_made": rng.poisson(2), "steals": rng.poisson(1), "blocks": rng.poisson(0.6),
                        "turnovers": rng.poisson(2), "field_goals_attempted": round(shares[k] / 1.1),
                        "free_throws_attempted": rng.poisson(2), "offensive_rebounds": rng.poisson(1),
                        "home_away": "home" if home else "away", "season_type": 2,
                        "athlete_position_abbreviation": "G",
                    })
        out[s] = pd.DataFrame(rows)
    return out


def fake_cbb_parquet(seasons_end=(2026, 2027)):
    games, _ = synth_games(n_teams=40, seasons=seasons_end, games_per_team=24, poss=True, seed=9)
    out = {}
    for s in seasons_end:
        rows = []
        for _, g in games[games["season"] == s].iterrows():
            for t, o, p, op, ha in ((g["home"], g["away"], g["home_pts"], g["away_pts"], "home"),
                                    (g["away"], g["home"], g["away_pts"], g["home_pts"], "away")):
                rows.append({"game_id": g["game_id"], "season": s, "game_date": g["date"],
                             "team_id": int(t[1:]) + 100, "opponent_team_id": int(o[1:]) + 100,
                             "team_display_name": f"School {t}", "opponent_team_display_name": f"School {o}",
                             "team_home_away": ha, "team_score": p, "opponent_team_score": op,
                             "field_goals_attempted": 58, "free_throws_attempted": 18,
                             "offensive_rebounds": 10, "turnovers": 12})
        out[s] = pd.DataFrame(rows)
    return out
