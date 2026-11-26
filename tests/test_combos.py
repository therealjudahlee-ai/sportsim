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
    assert set(ls["ticker"]) == {"B", "C", "D"}                     # D via its NO side (91%) at 90c
    assert ls.set_index("ticker").loc["D", "pick"] == "NO"
    sug = combos.suggest(opts, GAME, CFG, "nfl", PROJ, n_legs=2, mode="likely", min_prob=0.65)
    assert sug and sug[0]["joint"] >= sug[-1]["joint"]            # ranked by chance to hit
    # value mode only ever uses legs whose model chance beats their price
    for sg in combos.suggest(opts, GAME, CFG, "nfl", PROJ, n_legs=2, mode="value"):
        assert all(l["leg_prob"] > l["leg_price"] for l in sg["legs"])
