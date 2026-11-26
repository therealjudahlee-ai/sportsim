import base64

import numpy as np
import pandas as pd
import pytest

from engine import kalshi
from engine.config import DEFAULTS
from engine.props import StatDist


def _m(ticker, event, title, sub, bid, ask, strike=None, last=0.5, vol=100):
    d = {"ticker": ticker, "event_ticker": event, "title": title, "yes_sub_title": sub,
         "yes_bid_dollars": f"{bid:.4f}", "yes_ask_dollars": f"{ask:.4f}", "last_price_dollars": f"{last:.4f}",
         "volume_fp": f"{vol:.2f}", "open_interest_fp": "10.00", "close_time": "2026-11-28T03:00:00Z",
         "status": "active", "result": ""}
    if strike is not None:
        d["floor_strike"] = strike
    return d


def _nba_markets():
    g = "26NOV27LALSAS"
    rows = []
    rows += kalshi._market_rows([
        _m(f"KXNBAGAME-{g}-SAS", f"KXNBAGAME-{g}", "San Antonio wins", "San Antonio", 0.60, 0.62),
        _m(f"KXNBAGAME-{g}-LAL", f"KXNBAGAME-{g}", "Los Angeles L wins", "Los Angeles L", 0.37, 0.40)], "game")
    rows += kalshi._market_rows([
        _m(f"KXNBASPREAD-{g}-SAS7", f"KXNBASPREAD-{g}", "San Antonio wins by over 6.5 points",
           "San Antonio wins by over 6.5 points", 0.30, 0.33, 6.5)], "spread")
    rows += kalshi._market_rows([
        _m(f"KXNBATOTAL-{g}-229", f"KXNBATOTAL-{g}", "Full Game: Over 228.5 points scored", "Over 228.5 points",
           0.48, 0.52, 228.5)], "total")
    rows += kalshi._market_rows([
        _m(f"KXNBAPTS-{g}-SASVWEMBY1-25", f"KXNBAPTS-{g}", "Victor Wembanyama: 25+ points", "Victor Wembanyama: 25+",
           0.40, 0.45, 24.5)], "prop", "PTS")
    rows += kalshi._market_rows([  # a game we don't have a prediction for
        _m("KXNBAGAME-26NOV27BOSMIA-BOS", "KXNBAGAME-26NOV27BOSMIA", "Boston wins", "Boston", 0.5, 0.52),
        _m("KXNBAGAME-26NOV27BOSMIA-MIA", "KXNBAGAME-26NOV27BOSMIA", "Miami wins", "Miami", 0.47, 0.5)], "game")
    return pd.DataFrame(rows)


def _nba_preds():
    return pd.DataFrame([{"game_id": "1", "home": "SAS", "away": "LAL", "home_name": "San Antonio Spurs",
                          "away_name": "Los Angeles Lakers", "home_win": 0.70, "away_win": 0.30,
                          "model_margin": 8.0, "model_total": 232.0, "model_home_pts": 120.0,
                          "model_away_pts": 112.0}])


def test_parse_event_and_fees():
    d, teams = kalshi.parse_event("KXNBAGAME-26NOV27LALSAS")
    assert d == pd.Timestamp("2026-11-27") and teams == "LALSAS"
    assert abs(float(kalshi.taker_fee(0.5)) - 0.0175) < 1e-12
    assert kalshi.order_fee(0.5, 100) == 1.75           # matches Kalshi's published table
    assert kalshi.order_fee(0.5, 1) == 0.02             # rounds UP to the cent per order
    assert kalshi.order_fee(0.5, 100, maker=True) == 0.44


def test_score_nba_markets():
    mk = _nba_markets()
    wemby = {"player": "Victor Wembanyama", "player_norm": "victor wembanyama", "stat": "PTS", "team": "SAS",
             "median": 27.0, "dist": StatDist("gamma", 27.5, (0.3 * 27.5) ** 2)}
    out = kalshi.score_markets(mk, _nba_preds(), DEFAULTS["nba"], "nba", projections=[wemby])
    assert set(out["kind"]) == {"game", "spread", "total", "prop"}
    assert "BOSMIA" not in " ".join(out["ticker"])          # unmatched game skipped
    g = out[out["ticker"].str.endswith("-SAS") & (out["kind"] == "game")].iloc[0]
    # model 70% vs 62c ask -> buy YES; EV = 0.70 - 0.62 - fee
    assert g["side"] == "YES" and abs(g["ev_taker"] - (0.70 - 0.62 - 0.07 * 0.62 * 0.38)) < 1e-9
    lal = out[out["ticker"].str.endswith("-LAL")].iloc[0]
    # model 30% for LAL; YES ask 0.40 is bad, NO costs 1 - 0.37 = 0.63 for a 70% shot -> NO
    assert lal["side"] == "NO" and abs(lal["price"] - 0.63) < 1e-9
    sp = out[out["kind"] == "spread"].iloc[0]
    assert sp["side"] == "YES" and sp["model_prob"] > 0.5      # model SAS by 8 vs 6.5 line
    pr = out[out["kind"] == "prop"].iloc[0]
    assert abs(pr["model_prob"] - wemby["dist"].prob_over(24.5)) < 1e-9 or pr["side"] == "NO"


def test_college_name_matching():
    rows = kalshi._market_rows([
        _m("KXNCAAMBGAME-26NOV29NORFWASH-WASH", "KXNCAAMBGAME-26NOV29NORFWASH", "Washington wins", "Washington", .9, .95),
        _m("KXNCAAMBGAME-26NOV29NORFWASH-NORF", "KXNCAAMBGAME-26NOV29NORFWASH", "Norfolk St. wins", "Norfolk St.", .04, .08),
    ], "game")
    preds = pd.DataFrame([
        {"home": "264", "away": "2450", "home_name": "Washington Huskies", "away_name": "Norfolk State Spartans",
         "home_win": .93, "away_win": .07, "model_margin": 18, "model_total": 140, "model_home_pts": 79,
         "model_away_pts": 61},
        {"home": "265", "away": "9", "home_name": "Washington State Cougars", "away_name": "Idaho Vandals",
         "home_win": .8, "away_win": .2, "model_margin": 9, "model_total": 140, "model_home_pts": 74.5,
         "model_away_pts": 65.5}])
    gm = kalshi.match_games(pd.DataFrame(rows), preds, "ncaab")
    (gk, g), = gm.items()
    assert g["row"]["home"] == "264" and g["codes"] == {"WASH": "home", "NORF": "away"}


def test_nfl_code_alias():
    rows = kalshi._market_rows([
        _m("KXNFLGAME-26DEC07JACLAR-LAR", "KXNFLGAME-26DEC07JACLAR", "Los Angeles R wins", "Los Angeles R", .6, .62),
        _m("KXNFLGAME-26DEC07JACLAR-JAC", "KXNFLGAME-26DEC07JACLAR", "Jacksonville wins", "Jacksonville", .37, .4),
    ], "game")
    preds = pd.DataFrame([{"home": "LA", "away": "JAX", "home_win": .6, "away_win": .4, "model_margin": 3,
                           "model_total": 44, "model_home_pts": 23.5, "model_away_pts": 20.5}])
    gm = kalshi.match_games(pd.DataFrame(rows), preds, "nfl")
    assert list(gm.values())[0]["codes"] == {"LAR": "home", "JAC": "away"}


@pytest.mark.parametrize("kind", ["rsa", "ed25519"])
def test_signing_verifies(kind):
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa
    key = rsa.generate_private_key(65537, 2048) if kind == "rsa" else ed25519.Ed25519PrivateKey.generate()
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption()).decode()
    h = kalshi.auth_headers("kid", pem, "get", "/trade-api/v2/portfolio/settlements?limit=5")
    msg = (h["KALSHI-ACCESS-TIMESTAMP"] + "GET" + "/trade-api/v2/portfolio/settlements").encode()
    sig = base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"])
    if kind == "rsa":
        key.public_key().verify(sig, msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
                                hashes.SHA256())
    else:
        key.public_key().verify(sig, msg)


def test_discover_series(monkeypatch):
    class R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"series": [
                {"ticker": "KXNBAAST", "title": "Pro Basketball Assists"},
                {"ticker": "KXNBA3PT", "title": "Pro Basketball Three Pointers Made"},
                {"ticker": "KXNBASTATLEADER", "title": "NBA Regular Season League Leader"},
                {"ticker": "KXNBAPTSPRECISE", "title": "NBA Player Points Range"},
                {"ticker": "KXNBA1HSPREAD", "title": "Pro Basketball 1st Half Spread"},
                {"ticker": "KXNFLRECYDS", "title": "Pro Football Receiving Yards"},
            ]}
    monkeypatch.setattr(kalshi, "http", lambda: type("S", (), {"get": lambda self, *a, **k: R()})())
    s = kalshi.discover_series("nba")
    assert s["props"]["KXNBAAST"] == "AST" and s["props"]["KXNBA3PT"] == "FG3M"
    assert "KXNBASTATLEADER" not in s["props"] and "KXNBAPTSPRECISE" not in s["props"]
    assert "KXNBA1HSPREAD" not in s["spread"]      # half markets excluded


def test_track_log_grade_summary(tmp_path, monkeypatch):
    from engine import kalshi_track
    ep, rp = tmp_path / "e.csv", tmp_path / "r.csv"
    edges = pd.DataFrame([
        {"sport": "nba", "kind": "game", "stat": None, "game": "LAL @ SAS", "contract": "San Antonio wins",
         "ticker": "T1", "side": "YES", "price": 0.62, "model_prob": 0.70, "ev_taker": 0.06, "liquid": True,
         "close_time": "2026-11-28T03:00:00Z"},
        {"sport": "nba", "kind": "total", "stat": None, "game": "LAL @ SAS", "contract": "Over 228.5",
         "ticker": "T2", "side": "NO", "price": 0.50, "model_prob": 0.56, "ev_taker": 0.04, "liquid": True,
         "close_time": "2026-11-28T03:00:00Z"},
        {"sport": "nba", "kind": "prop", "stat": "PTS", "game": "LAL @ SAS", "contract": "thin market",
         "ticker": "T3", "side": "YES", "price": 0.30, "model_prob": 0.40, "ev_taker": 0.08, "liquid": False,
         "close_time": "2026-11-28T03:00:00Z"},
        {"sport": "nba", "kind": "game", "stat": None, "game": "x", "contract": "low ev", "ticker": "T4",
         "side": "YES", "price": 0.5, "model_prob": 0.51, "ev_taker": -0.01, "liquid": True,
         "close_time": "2026-11-28T03:00:00Z"}])
    assert kalshi_track.log_edges(edges, "2026-11-26T14:00Z", path=ep) == 2   # illiquid + negative EV skipped
    kalshi_track.log_edges(edges.assign(price=edges["price"] + 0.03), "2026-11-26T22:00Z", path=ep)
    results = {"T1": {"result": "yes", "last_price_dollars": "0.6800"},
               "T2": {"result": "yes", "last_price_dollars": "0.5500"}}
    monkeypatch.setattr(kalshi_track, "fetch_market", lambda t, d=None: results.get(t))
    n = kalshi_track.grade(now="2026-12-02T00:00Z", edges_path=ep, results_path=rp)
    assert n == 2
    g = kalshi_track.graded(ep, rp)
    assert len(g) == 2                                 # first log per contract only
    t1 = g[g["ticker"] == "T1"].iloc[0]
    assert t1["won"] and abs(t1["price"] - 0.62) < 1e-9
    assert abs(t1["profit"] - (1 - 0.62 - 0.07 * 0.62 * 0.38)) < 1e-9
    assert abs(t1["clv"] - 0.06) < 1e-9                # bought 62, closed 68
    t2 = g[g["ticker"] == "T2"].iloc[0]
    assert not t2["won"] and abs(t2["clv"] - (0.45 - 0.50)) < 1e-9   # NO side: 1 - 0.55 = 0.45
    s = kalshi_track.summary(g)
    assert s["contracts"] == 2 and s["win_rate"] == 0.5


def test_fetch_settlements_parsing(monkeypatch):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    pem = ed25519.Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
    seen = {}

    class R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"settlements": [
                {"ticker": "KXNBAGAME-26NOV27LALSAS-SAS", "event_ticker": "E", "market_result": "yes",
                 "yes_count_fp": "10.00", "yes_total_cost_dollars": "6.2000", "no_count_fp": "0.00",
                 "no_total_cost_dollars": "0.0000", "revenue": 1000, "settled_time": "2026-11-28T06:00:00Z",
                 "fee_cost": "0.1700"},
                {"ticker": "KXNFLGAME-26DEC07DALSEA-DAL", "event_ticker": "E2", "market_result": "no",
                 "yes_count_fp": "5.00", "yes_total_cost_dollars": "1.5500", "no_count_fp": "0.00",
                 "no_total_cost_dollars": "0.0000", "revenue": 0, "settled_time": "2026-12-08T06:00:00Z",
                 "fee_cost": "0.0800"}], "cursor": ""}

    def get(self, url, params=None, timeout=None, headers=None):
        seen.update(headers or {})
        return R()
    monkeypatch.setattr(kalshi, "http", lambda: type("S", (), {"get": get})())
    df = kalshi.fetch_settlements("kid", pem)
    assert {"KALSHI-ACCESS-KEY", "KALSHI-ACCESS-SIGNATURE", "KALSHI-ACCESS-TIMESTAMP"} <= set(seen)
    nba = df[df["sport"] == "nba"].iloc[0]
    assert abs(nba["profit"] - (10.0 - 6.2 - 0.17)) < 1e-9
    assert abs(df[df["sport"] == "nfl"].iloc[0]["profit"] - (-1.55 - 0.08)) < 1e-9


def test_normalize_pem_handles_messy_paste():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
    for key in (ed25519.Ed25519PrivateKey.generate(), rsa.generate_private_key(65537, 2048)):
        fmt = serialization.PrivateFormat.PKCS8 if isinstance(key, ed25519.Ed25519PrivateKey) \
            else serialization.PrivateFormat.TraditionalOpenSSL
        pem = key.private_bytes(serialization.Encoding.PEM, fmt, serialization.NoEncryption()).decode()
        lines = pem.strip().splitlines()
        messy = [
            " ".join(lines),                                           # all on one line with spaces
            "\n".join(lines[:-1]) + " " + lines[-1],                   # END marker on the key's last line
            "\\n".join(lines),                                          # literal backslash-n
            '"' + pem + '"',                                            # wrapped in quotes
        ]
        for m in messy:
            h = kalshi.auth_headers("kid", m, "GET", "/trade-api/v2/portfolio/settlements")
            assert h["KALSHI-ACCESS-SIGNATURE"]


def test_passing_td_market_is_not_anytime_td(monkeypatch):
    """Regression: 'Joe Burrow: 1+ passing touchdowns' was scored as Burrow scoring a TD himself."""
    class R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"series": [{"ticker": "KXNFLPASSTDS", "title": "Pro Football Passing Touchdowns"},
                               {"ticker": "KXNFLRSHTDS", "title": "Pro Football Rushing Touchdowns"},
                               {"ticker": "KXNFLXTD", "title": "Pro Football Touchdowns"}]}
    monkeypatch.setattr(kalshi, "http", lambda: type("S", (), {"get": lambda self, *a, **k: R()})())
    s = kalshi.discover_series("nfl")
    assert s["props"]["KXNFLPASSTDS"] == "passing_tds"
    assert s["props"]["KXNFLRSHTDS"] == "rushing_tds"
    assert "KXNFLXTD" not in s["props"]            # ambiguous "touchdowns" is never guessed
    assert kalshi.title_matches_stat("passing_tds", "Joe Burrow: 1+ passing touchdowns")
    assert not kalshi.title_matches_stat("anytime_td", "Joe Burrow: 1+ passing touchdowns")
    assert not kalshi.title_matches_stat("PTS", "Steph Curry: 5+ three pointers")


def test_burrow_scored_sensibly_and_suspects_flagged():
    g = "26NOV29NOCIN"
    rows = kalshi._market_rows([
        _m(f"KXNFLGAME-{g}-CIN", f"KXNFLGAME-{g}", "Cincinnati wins", "Cincinnati", .6, .62),
        _m(f"KXNFLGAME-{g}-NO", f"KXNFLGAME-{g}", "New Orleans wins", "New Orleans", .37, .4)], "game")
    rows += kalshi._market_rows([
        _m(f"KXNFLPASSTDS-{g}-CINJBURROW9-1", f"KXNFLPASSTDS-{g}", "Joe Burrow: 1+ passing touchdowns",
           "Joe Burrow: 1+", .90, .94, 0.5)], "prop", "passing_tds")
    rows += kalshi._market_rows([   # wrongly-mapped stat: title says passing, stat says anytime -> skipped
        _m(f"KXNFLPASSTDS-{g}-CINJBURROW9-2", f"KXNFLPASSTDS-{g}", "Joe Burrow: 2+ passing touchdowns",
           "Joe Burrow: 2+", .55, .60, 1.5)], "prop", "anytime_td")
    preds = pd.DataFrame([{"home": "CIN", "away": "NO", "home_win": .62, "away_win": .38, "model_margin": 3,
                           "model_total": 47, "model_home_pts": 25, "model_away_pts": 22}])
    burrow_pass = {"player": "Joe Burrow", "player_norm": "joe burrow", "stat": "passing_tds", "team": "CIN",
                   "median": 2.0, "dist": StatDist("nbinom", 1.9, 2.1)}
    burrow_any = dict(burrow_pass, stat="anytime_td", median=0.0, dist=StatDist("poisson", 0.1, 0.1))
    out = kalshi.score_markets(pd.DataFrame(rows), preds, DEFAULTS["nfl"], "nfl", [burrow_pass, burrow_any])
    props = out[out["kind"] == "prop"]
    assert len(props) == 1                                   # the mis-mapped one is dropped
    p = props.iloc[0]
    assert abs((p["model_prob"] if p["side"] == "YES" else 1 - p["model_prob"]) - burrow_pass["dist"].prob_over(0.5)) < 1e-9
    assert not p["suspect"]
    wild = kalshi.score_markets(pd.DataFrame(rows), preds.assign(home_win=.99, away_win=.01),
                                DEFAULTS["nfl"], "nfl", [burrow_pass])
    assert wild[(wild["kind"] == "game")]["suspect"].any()   # 99% vs 62c -> flagged


def test_pem_bare_body_and_dashes():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    pem = ed25519.Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
    body = "".join(pem.strip().splitlines()[1:-1])
    for m in (body, pem.replace("-----", "–––––")):
        assert kalshi.auth_headers("kid", m, "GET", "/x")["KALSHI-ACCESS-SIGNATURE"]
    with pytest.raises(ValueError, match="Couldn't read KALSHI_PRIVATE_KEY"):
        kalshi.auth_headers("kid", "not a key at all", "GET", "/x")
