import numpy as np
import pandas as pd

from engine.props import StatDist, project_nfl, project_nba, parlay_prob, shrink_matchup, nfl_env_factors
from engine.odds_api import _consolidate_props, match_team, NFL_TEAMS
from engine.edges import score_props, build_parlays
from engine.common import devig_pair, prob_to_american, norm_name


def test_distributions_sane():
    g = StatDist("gamma", 70.0, (0.6 * 70) ** 2)
    assert 0.35 < g.prob_over(70) < 0.5          # right-skewed: median < mean
    assert abs(g.prob_over(60.5) + g.prob_under(60.5) - 1) < 1e-9
    nb = StatDist("nbinom", 5.2, 6.5)
    assert abs(nb.prob_over(4.5) + nb.prob_under(4.5) - 1) < 1e-9
    s = nb.sample(200_000, seed=1)
    assert abs((s > 4.5).mean() - nb.prob_over(4.5)) < 0.01
    po = StatDist("poisson", 0.5, 0.5)
    assert abs(po.prob_over(0.5) - (1 - np.exp(-0.5))) < 1e-9


def _nfl_logs(n=14, seed=0):
    rng = np.random.default_rng(seed)
    tg = rng.poisson(8, n)
    rec = rng.binomial(tg, 0.65)
    yds = np.round(rec * rng.gamma(4, 3, n))
    return pd.DataFrame({
        "targets": tg, "receptions": rec, "receiving_yards": yds,
        "anytime_td": rng.poisson(0.4, n), "dk_points": yds * 0.1 + rec,
    })


def test_project_nfl_receiver():
    logs = _nfl_logs()
    league = {("eff", "WR", "receiving_yards"): 8.0, ("eff", "WR", "receptions"): 0.64, ("td_rate", "WR"): 0.3}
    for stat in ["receiving_yards", "receptions", "anytime_td", "dk_points"]:
        out = project_nfl(logs, stat, "WR", league, def_mult=1.05, env=nfl_env_factors(stat, 25, -3, 22.5))
        assert out is not None and out["mean"] > 0
        assert out["p20"] <= out["median"] <= out["p80"]
    assert project_nfl(logs.head(2), "receiving_yards", "WR", league) is None


def test_project_nba():
    rng = np.random.default_rng(0)
    n = 20
    mins = rng.normal(34, 3, n)
    logs = pd.DataFrame({"MIN": mins, "PTS": rng.poisson(mins * 0.75), "REB": rng.poisson(mins * 0.3),
                         "IS_HOME": rng.random(n) < 0.5})
    out = project_nba(logs, "PTS", {"PTS": 0.5}, def_mult=1.02, pace_mult=1.01, home=True)
    assert 20 < out["mean"] < 30
    out2 = project_nba(logs, "REB", {"REB": 0.2})
    assert out2["dist"].family == "nbinom"


def test_parlay_correlation_direction():
    a = {"player": "QB", "team": "KC", "game": "g", "stat": "passing_yards", "prob": 0.55, "side": "over"}
    b = {"player": "WR", "team": "KC", "game": "g", "stat": "receiving_yards", "prob": 0.55, "side": "over"}
    c = {"player": "WR2", "team": "KC", "game": "g", "stat": "receiving_yards", "prob": 0.55, "side": "over"}
    indep = 0.55 * 0.55
    assert parlay_prob([a, b], "nfl") > indep + 0.03      # stack is positively correlated
    assert parlay_prob([b, c], "nfl") < indep             # two receivers compete for targets
    d = dict(b, game="other", team="BUF")
    assert abs(parlay_prob([a, d], "nfl") - indep) < 1e-9  # different games -> independent, exact


def test_consolidate_and_score():
    book = {
        ("Travis Kelce", "receiving_yards", 55.5): {"over": [(-115, "A"), (-110, "B")], "under": [(-105, "A"), (-110, "B")]},
        ("Travis Kelce", "receiving_yards", 52.5): {"over": [(-140, "C")], "under": [(110, "C")]},
        ("Isiah Pacheco", "anytime_td", 0.5): {"over": [(120, "A")], "under": []},
    }
    df = _consolidate_props(book, "ev1")
    k = df[df["player"] == "Travis Kelce"].iloc[0]
    assert k["line"] == 55.5 and k["devigged"] and k["best_over"] == -110
    assert 0.48 < k["market_over_prob"] < 0.53
    proj = [{"player": "Travis Kelce", "player_norm": norm_name("Travis Kelce"), "team": "KC", "opp": "BUF",
             "game": "BUF @ KC", "stat": "receiving_yards", "position": "TE",
             "dist": StatDist("gamma", 64.0, (0.55 * 64) ** 2)}]
    scored = score_props(proj, df)
    assert len(scored) == 1 and scored.iloc[0]["side"] == "over" and scored.iloc[0]["edge"] > 0


def test_build_parlays():
    legs = pd.DataFrame([
        {"player": f"P{i}", "team": "T", "game": f"g{i % 3}", "stat": "PTS", "line": 20.5, "side": "over",
         "prob": 0.58 + 0.01 * i, "market_prob": 0.52, "edge": 0.06 + 0.01 * i, "best_price": -110}
        for i in range(8)
    ])
    tix = build_parlays(legs, "nba", n_legs=3, max_tickets=3)
    assert 1 <= len(tix) <= 3
    assert all(len(t["legs"]) == 3 for t in tix)
    assert tix[0]["ev"] >= tix[-1]["ev"]


def test_misc():
    a, b = devig_pair(-110, -110)
    assert abs(a - 0.5) < 1e-9
    assert prob_to_american(0.75) == "-300" and prob_to_american(0.25) == "+300"
    assert norm_name("C.J. Stroud") == norm_name("CJ Stroud")
    assert norm_name("Jaren Jackson Jr.") == "jaren jackson"
    assert match_team("Kansas City Chiefs", {v: k for k, v in NFL_TEAMS.items()}) == "KC"
    assert match_team("Michigan St Spartans", {"127": "Michigan State Spartans"}) == "127"
    assert shrink_matchup(1.3, 8) == 1.15


def test_value_finder_and_fair_ladder():
    from engine.edges import value_finder, fair_ladder, hit_rate
    d = StatDist("gamma", 25.0, (0.35 * 25) ** 2)
    proj = [{"player": "A", "player_norm": "a", "team": "T", "opp": "O", "game": "g", "stat": "PTS",
             "median": d.quantile(0.5), "dist": d, "hist": [20, 30, 25, 28, 22, 31, 19, 27, 26, 24]}]
    lad = pd.DataFrame({"player": ["A"] * 4, "player_norm": ["a"] * 4, "stat": ["PTS"] * 4,
                        "line": [24.5, 27.5, 29.5, 34.5], "side": ["over"] * 4, "price": [-120, 190, 140, 260],
                        "book": ["X"] * 4, "n_books": [3] * 4})
    vf = value_finder(proj, lad, 2.0, 3.0)
    assert set(vf["line"]) == {27.5, 29.5}                 # -120 (1.83x) and +260 (3.6x) excluded
    r = vf[vf["line"] == 27.5].iloc[0]
    assert abs(r["ev"] - (d.prob_over(27.5) * 2.9 - 1)) < 1e-9
    fl = fair_ladder(proj, (2.0, 3.0)).iloc[0]
    for m in ("2", "3"):
        line = fl[f"line_{m}x"]
        assert d.prob_over(line) >= 1 / float(m) > d.prob_over(line + 1)
    assert hit_rate([1, 2, 3, 4, 5, 6], 3.5, 10) == 0.5


def test_alt_ladder_parsing():
    from engine.odds_api import OddsClient, alt_ladder, NBA_ALT_MARKETS
    c = OddsClient("k")
    c._get = lambda path, **p: {"bookmakers": [
        {"title": "A", "markets": [{"key": "player_points_alternate", "outcomes": [
            {"name": "Over", "description": "Jayson Tatum", "point": 29.5, "price": 140},
            {"name": "Over", "description": "Jayson Tatum", "point": 32.5, "price": 210}]}]},
        {"title": "B", "markets": [{"key": "player_points_alternate", "outcomes": [
            {"name": "Over", "description": "Jayson Tatum", "point": 29.5, "price": 155}]}]}]}
    df = alt_ladder(c, "basketball_nba", "e1", ["PTS"], NBA_ALT_MARKETS)
    r = df[df["line"] == 29.5].iloc[0]
    assert r["price"] == 155 and r["book"] == "B" and r["n_books"] == 2 and len(df) == 2


def test_realistic_markets():
    from engine.props import market_offered
    mean = lambda m: {"mean": m}
    assert not market_offered("nfl", "rushing_yards", mean(2.1), "QB")      # pocket passer: no market
    assert market_offered("nfl", "rushing_yards", mean(38.0), "QB")         # mobile QB: real market
    assert not market_offered("nfl", "passing_yards", mean(240.0), "WR")    # position gate
    assert not market_offered("nfl", "receptions", mean(0.9), "RB")         # barely used back
    assert market_offered("nfl", "dk_points", mean(3.0), "WR")              # DFS stats always allowed
    assert not market_offered("nba", "BLK", mean(0.2))
    assert market_offered("nba", "PTS", mean(18.0))
    assert not market_offered("nfl", "anytime_td", None, "RB")


def test_news_classify_and_match():
    from engine.news import classify, match_news, report_changes
    assert classify("Lakers' Anthony Davis has been ruled out tonight vs. Denver") == "out"
    assert classify("Davis has not been ruled out and is a game-time decision") == "questionable"
    assert classify("Source: Joel Embiid is expected to play tonight") == "active"
    assert classify("Hurts (knee) is doubtful for Sunday") == "doubtful"
    assert classify("Great win for the Bills tonight") is None
    players = pd.DataFrame({"name": ["Anthony Davis", "Joel Embiid", "Jalen Brunson"],
                            "name_norm": ["anthony davis", "joel embiid", "jalen brunson"],
                            "team": ["LAL", "PHI", "NYK"], "status": ["", "", ""]})
    posts = pd.DataFrame({"id": ["1", "2", "3"], "url": ["u1", "u2", "u3"],
                          "time": pd.to_datetime(["2026-11-24T20:00Z", "2026-11-24T21:00Z", "2026-11-24T21:30Z"], utc=True),
                          "text": ["Anthony Davis (ankle) is OUT tonight.", "Joel Embiid will play vs Boston",
                                   "Anthony Davis upgraded to available, will play"]})
    m = match_news(posts, players).set_index("name_norm")
    assert m.loc["anthony davis", "status"] == "active"        # newest report wins
    assert m.loc["joel embiid", "status"] == "active"
    fresh = pd.DataFrame({"name_norm": ["jalen brunson", "joel embiid"], "status": ["out", "questionable"],
                          "comment": ["ankle", ""]})
    rc = report_changes(fresh, set(), players)
    assert list(rc["player"]) == ["Jalen Brunson"]
