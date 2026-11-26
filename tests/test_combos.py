import numpy as np
import pandas as pd
from scipy.stats import norm

from engine import combos
from engine.config import DEFAULTS
from engine.props import StatDist

CFG = DEFAULTS["nfl"]
GAME = {"home": "KC", "away": "BUF", "model_margin": 3.0, "model_total": 48.0}
QB = {"player": "Pat Mahomes", "player_norm": "pat mahomes", "stat": "passing_yards", "team": "KC",
      "median": 265.0, "dist": StatDist("gamma", 270.0, (0.28 * 270) ** 2)}
PROJ = {(QB["player_norm"], QB["stat"]): QB}


def leg(kind, pick="YES", **kw):
    return {"kind": kind, "pick": pick, "team_side": kw.get("side"), "strike": kw.get("strike", np.nan),
            "stat": kw.get("stat"), "player_norm": kw.get("pn"), "contract": kind}


def test_break_even_price():
    for p in (0.2, 0.5, 0.8):
        q = combos.break_even_price(p)
        assert abs(p - q - 0.07 * q * (1 - q)) < 1e-12 and q < p


def test_single_legs_match_model():
    s = combos.simulate([leg("game", side="home")], GAME, CFG, "nfl", PROJ, n=200_000)
    assert abs(s["joint"] - norm.cdf(3.0 / CFG.margin_sigma)) < 0.005
    s = combos.simulate([leg("total", strike=45.5)], GAME, CFG, "nfl", PROJ, n=200_000)
    assert abs(s["joint"] - norm.sf(45.5, 48.0, CFG.total_sigma)) < 0.005
    s = combos.simulate([leg("prop", strike=274.5, stat="passing_yards", pn="pat mahomes")], GAME, CFG, "nfl",
                        PROJ, n=200_000)
    assert abs(s["joint"] - QB["dist"].prob_over(274.5)) < 0.006       # copula keeps the marginal intact
    s = combos.simulate([leg("spread", "NO", side="away", strike=3.5)], GAME, CFG, "nfl", PROJ, n=200_000)
    assert abs(s["joint"] - (1 - norm.sf(3.5, -3.0, CFG.margin_sigma))) < 0.005


def test_correlation_and_contradiction():
    legs = [leg("game", side="home"), leg("prop", strike=274.5, stat="passing_yards", pn="pat mahomes"),
            leg("total", strike=47.5)]
    s = combos.simulate(legs, GAME, CFG, "nfl", PROJ)
    assert s["joint"] > s["independent"] * 1.1          # KC win + Mahomes yards + over hit together
    s2 = combos.simulate([leg("game", side="home"), leg("game", side="away")], GAME, CFG, "nfl", PROJ)
    assert s2["joint"] == 0.0


def test_leg_options_and_suggest():
    edges = pd.DataFrame([
        {"game_key": "g", "kind": "game", "team_side": "home", "strike": np.nan, "stat": None, "player_norm": None,
         "contract": "Kansas City wins", "ticker": "A", "yes_ask": 0.55, "yes_bid": 0.53, "p_yes": 0.60,
         "suspect": False},
        {"game_key": "g", "kind": "total", "team_side": None, "strike": 45.5, "stat": None, "player_norm": None,
         "contract": "Over 45.5", "ticker": "B", "yes_ask": 0.52, "yes_bid": 0.50, "p_yes": 0.58, "suspect": False},
        {"game_key": "g", "kind": "prop", "team_side": None, "strike": 249.5, "stat": "passing_yards",
         "player_norm": "pat mahomes", "contract": "Mahomes 250+", "ticker": "C", "yes_ask": 0.55, "yes_bid": 0.52,
         "p_yes": 0.62, "suspect": False},
        {"game_key": "g", "kind": "prop", "team_side": None, "strike": 274.5, "stat": "passing_yards",
         "player_norm": "pat mahomes", "contract": "Mahomes 275+", "ticker": "D", "yes_ask": 0.40, "yes_bid": 0.37,
         "p_yes": 0.45, "suspect": False},
        {"game_key": "other", "kind": "game", "team_side": "home", "strike": np.nan, "stat": None,
         "player_norm": None, "contract": "x", "ticker": "E", "yes_ask": 0.5, "yes_bid": 0.48, "p_yes": 0.5,
         "suspect": False}])
    opts = combos.leg_options(edges, "g")
    assert len(opts) == 8 and set(opts["pick"]) == {"YES", "NO"}
    assert opts[opts["ticker"] == "A"].set_index("pick").loc["NO", "leg_price"] == 1 - 0.53
    sug = combos.suggest(opts, GAME, CFG, "nfl", PROJ, n_legs=2)
    assert sug
    for s in sug:
        tick = [l["ticker"] for l in s["legs"]]
        assert not ({"C", "D"} <= set(tick))           # two rungs of one ladder never paired
        assert 0 < s["max_quote"] < s["fair_price"]
    sug3 = combos.suggest(opts, GAME, CFG, "nfl", PROJ, n_legs=3)
    assert all(len(s["legs"]) == 3 for s in sug3)


def test_likely_mode():
    edges = pd.DataFrame([
        {"game_key": "g", "kind": "game", "team_side": "home", "strike": np.nan, "stat": None, "player_norm": None,
         "contract": "KC wins", "ticker": "A", "yes_ask": 0.70, "yes_bid": 0.68, "p_yes": 0.62, "suspect": False},
        {"game_key": "g", "kind": "total", "team_side": None, "strike": 41.5, "stat": None, "player_norm": None,
         "contract": "Over 41.5", "ticker": "B", "yes_ask": 0.80, "yes_bid": 0.78, "p_yes": 0.74, "suspect": False},
        {"game_key": "g", "kind": "prop", "team_side": None, "strike": 199.5, "stat": "passing_yards",
         "player_norm": "pat mahomes", "contract": "Mahomes 200+", "ticker": "C", "yes_ask": 0.86, "yes_bid": 0.84,
         "p_yes": 0.83, "suspect": False},
        {"game_key": "g", "kind": "prop", "team_side": None, "strike": 349.5, "stat": "passing_yards",
         "player_norm": "pat mahomes", "contract": "Mahomes 350+", "ticker": "D", "yes_ask": 0.12, "yes_bid": 0.10,
         "p_yes": 0.09, "suspect": False}])
    opts = combos.leg_options(edges, "g")
    ls = combos.likely_sides(opts, 0.65, 0.90)
    # C (YES 200+ at 86c) and D (NO 350+ at 90c) are the same Mahomes ladder: keep only the cheaper rung
    assert set(ls["ticker"]) == {"B", "C"}
    sug = combos.suggest(opts, GAME, CFG, "nfl", PROJ, n_legs=2, mode="likely", min_prob=0.65)
    assert sug and sug[0]["joint"] >= sug[-1]["joint"]            # ranked by chance to hit
    # value mode only ever uses legs whose model chance beats their price
    for sg in combos.suggest(opts, GAME, CFG, "nfl", PROJ, n_legs=2, mode="value"):
        assert all(l["leg_prob"] > l["leg_price"] for l in sg["legs"])


def test_ladder_rungs_never_stacked_even_with_nan_fields():
    """Regression: four rungs of one total ladder were suggested as a 4-leg combo (NaN != NaN broke dedupe)."""
    rows = []
    for k, strike in enumerate([226.5, 228.5, 230.5, 232.5]):
        rows.append({"game_key": "g", "kind": "total", "team_side": np.nan, "strike": strike, "stat": np.nan,
                     "player_norm": np.nan, "contract": f"Over {strike}", "ticker": f"T{k}",
                     "yes_ask": 0.40 - 0.05 * k, "yes_bid": 0.37 - 0.05 * k, "p_yes": 0.25, "suspect": False})
    rows.append({"game_key": "g", "kind": "game", "team_side": "home", "strike": np.nan, "stat": np.nan,
                 "player_norm": np.nan, "contract": "KC wins", "ticker": "W", "yes_ask": 0.55, "yes_bid": 0.53,
                 "p_yes": 0.75, "suspect": False})
    opts = combos.leg_options(pd.DataFrame(rows), "g")
    ls = combos.likely_sides(opts, 0.6, 0.95)
    assert (ls["kind"] == "total").sum() == 1                  # one rung of the total ladder survives
    assert ls[ls["kind"] == "total"].iloc[0]["leg_price"] == ls[ls["kind"] == "total"]["leg_price"].min()
    for mode in ("likely", "value"):
        for sg in combos.suggest(opts, GAME, CFG, "nfl", PROJ, n_legs=2, mode=mode, min_prob=0.6):
            assert sum(l["kind"] == "total" for l in sg["legs"]) <= 1


def test_simulate_multi_game_is_product():
    g2 = {"home": "PHI", "away": "DAL", "model_margin": -2.0, "model_total": 44.0}
    l1 = dict(leg("game", side="home"), game_key="g1")
    l2 = dict(leg("game", side="home"), game_key="g2")
    s = combos.simulate_multi([l1, l2], {"g1": GAME, "g2": g2}, CFG, "nfl", PROJ, n=200_000)
    expect = norm.cdf(3.0 / CFG.margin_sigma) * norm.cdf(-2.0 / CFG.margin_sigma)
    assert abs(s["joint"] - expect) < 0.006 and s["n_games"] == 2


def test_suggest_props_across_games():
    rng = np.random.default_rng(0)
    games = {"g1": GAME, "g2": {"home": "PHI", "away": "DAL", "model_margin": 4.0, "model_total": 46.0}}
    proj, rows = {}, []
    players = [("p1", "KC", "g1"), ("p2", "KC", "g1"), ("p3", "BUF", "g1"), ("p4", "PHI", "g2"),
               ("p5", "DAL", "g2"), ("p6", "PHI", "g2"), ("p7", "DAL", "g2")]
    for i, (pn, team, gk) in enumerate(players):
        d = StatDist("gamma", 60.0 + 5 * i, (0.5 * (60.0 + 5 * i)) ** 2)
        proj[(pn, "receiving_yards")] = {"player_norm": pn, "stat": "receiving_yards", "team": team, "dist": d,
                                         "median": d.quantile(.5)}
        for j, strike in enumerate([29.5, 39.5, 49.5]):          # a 3-rung ladder per player
            p = d.prob_over(strike)
            rows.append({"game_key": gk, "game": gk, "kind": "prop", "team_side": None, "strike": strike,
                         "stat": "receiving_yards", "player_norm": pn, "contract": f"{pn} {strike + .5:.0f}+",
                         "ticker": f"{pn}-{j}", "yes_ask": min(.97, p + 0.03), "yes_bid": max(.02, p - 0.02),
                         "p_yes": p, "suspect": False, "volume": 50})
    opts = combos.leg_options(pd.DataFrame(rows))
    for n in (2, 3, 5, 6):
        sug = combos.suggest_props(opts, games, CFG, "nfl", proj, n_legs=n, min_prob=0.6, max_price=0.95)
        assert sug, n
        for s in sug:
            assert len(s["legs"]) == n
            assert len({l["player_norm"] for l in s["legs"]}) == n      # one rung per player
            assert all(l["leg_prob"] >= 0.6 for l in s["legs"])
        assert [s["joint"] for s in sug] == sorted([s["joint"] for s in sug], reverse=True)
    assert any(s["n_games"] == 2 for s in combos.suggest_props(opts, games, CFG, "nfl", proj, n_legs=4,
                                                                 min_prob=0.6, max_price=0.95))
