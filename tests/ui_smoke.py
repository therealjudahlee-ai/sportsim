"""Execute every page headlessly against fake data.   python tests/ui_smoke.py"""
import os
import runpy
import sys
import traceback

sys.path.insert(0, os.getcwd())
from tests import fake_streamlit  # noqa: E402

st = fake_streamlit.install()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import espn  # noqa: E402
from tests import fakes  # noqa: E402

# ---- fake sources
sys.modules["nflreadpy"] = fakes.fake_nflreadpy()
sys.modules["nba_api"] = None
nba_frames, cbb_frames = fakes.fake_nba_parquet(), fakes.fake_cbb_parquet()
_real_read_parquet = pd.read_parquet


def fake_read_parquet(url):
    yr = int(url.split("_")[-1].split(".")[0])
    return (nba_frames if "nba" in url else cbb_frames)[yr]


pd.read_parquet = fake_read_parquet
SLATES = {
    "nba": [("BOS", "NY", "1", "2", "BOS -6.5"), ("GS", "SA", "3", "4", "SA -1.5")],
    "mens-college-basketball": [("S1", "S2", "101", "102", "S1 -4.5"), ("S3", "S9", "103", "999", "S9 -2")],
}


def get_json(url, diag=None, what="", timeout=10, **params):
    for k, v in SLATES.items():
        if f"/{k}/" in url:
            return fakes.fake_espn_json(k, v)(url, diag, what, timeout, **params)
    return fakes.fake_espn_json("x", [])(url, diag, what, timeout, **params)


espn.get_json = get_json

from ui import loaders  # noqa: E402
from engine.common import today_et  # noqa: E402

# NBA/CBB fakes need a date inside the synthetic seasons
import datetime as _dt  # noqa: E402
import engine.common as _common  # noqa: E402
_common.today_et = lambda: _dt.date(2026, 3, 15)


def run(page, label):
    fake_streamlit.LOG.clear()
    try:
        runpy.run_path(page, run_name="__main__")
        status = "ok"
    except fake_streamlit.StopExecution:
        status = "stopped"
    except Exception:
        traceback.print_exc()
        status = "CRASH"
    errs = [e for e in fake_streamlit.LOG if e[0] == "error"]
    n_metrics = sum(1 for e in fake_streamlit.LOG if e[0] == "metric")
    n_tables = sum(1 for e in fake_streamlit.LOG if e[0] == "dataframe")
    print(f"{label:38s} {status:8s} metrics={n_metrics:3d} tables={n_tables:3d} errors={errs}")
    return status, fake_streamlit.LOG.copy()


results = {}
pages = [("app.py", "home"), ("pages/1_NFL.py", "nfl (no odds key)"), ("pages/2_NBA.py", "nba (no odds key)"),
         ("pages/3_College_Basketball.py", "cbb"), ("pages/4_College_Football.py", "cfb"),
         ("pages/5_Track_Record.py", "track record (empty)")]
for p, label in pages:
    results[label] = run(p, label)

# ---- now with an odds key + fake market data
loaders.odds_key = lambda: "test-key"


def fake_game_lines(sport):
    names = {"nfl": fakes.NFL_TEAMS}
    from engine.odds_api import NBA_TEAMS, NFL_TEAMS
    inv = {v: k for k, v in (NFL_TEAMS if sport == "nfl" else NBA_TEAMS).items()}
    preds = (loaders.nfl_ctx() if sport == "nfl" else loaders.nba_ctx("2026-03-15"))["predictions"]
    return pd.DataFrame({"home_name": preds["home"].map(inv), "away_name": preds["away"].map(inv),
                         "home_spread": -2.5, "total_line": 220.5 if sport == "nba" else 44.5,
                         "market_home_win_ml": 0.58, "n_books": 6}), "450", []


def fake_prop_lines(sport, matchups, stats):
    from engine.common import norm_name
    projs = loaders.projections_for(loaders.nfl_ctx() if sport == "nfl" else loaders.nba_ctx("2026-03-15"), tuple(stats))
    rng = np.random.default_rng(0)
    rows = [{"event_id": "e", "player": p["player"], "player_norm": norm_name(p["player"]), "stat": p["stat"],
             "line": 0.5 if p["stat"] == "anytime_td" else float(np.floor(p["median"] * rng.uniform(0.85, 1.1)) + 0.5),
             "best_over": -110 + rng.integers(-15, 25), "best_over_book": "BookA", "best_under": -110,
             "best_under_book": "BookB", "market_over_prob": 0.5, "devigged": True, "n_books": 5} for p in projs]
    return pd.DataFrame(rows), "440", []


def fake_alt_lines(sport, matchups, stats):
    from engine.common import norm_name
    ctx = loaders.nfl_ctx() if sport == "nfl" else loaders.nba_ctx("2026-03-15")
    projs = loaders.projections_for(ctx, tuple(stats))
    rows = []
    for p in projs:
        for k, price in enumerate([-150, 110, 150, 190, 240]):
            rows.append({"event_id": "e", "player": p["player"], "player_norm": norm_name(p["player"]),
                         "stat": p["stat"], "line": float(np.floor(p["median"] * (0.8 + 0.12 * k)) + 0.5),
                         "side": "over", "price": price, "book": "BookA", "n_books": 3})
    return pd.DataFrame(rows), "430", []


def fake_kalshi(sport):
    """Build Kalshi-format markets for whatever games the fake slates contain."""
    from engine import kalshi
    getter = {"nfl": lambda: loaders.nfl_ctx(), "nba": lambda: loaders.nba_ctx("2026-03-15"),
              "ncaab": lambda: loaders.ncaab_ctx("2026-03-15")}
    if sport not in getter:
        return pd.DataFrame(), []
    ctx = getter[sport]()
    raw_g, raw_s, raw_t, raw_p = [], [], [], []
    for _, p in ctx["predictions"].iterrows():
        h, a = str(p["home"]), str(p["away"])
        hn, an = (p.get("home_name") or h), (p.get("away_name") or a)
        if sport == "ncaab":  # Kalshi uses short school names
            hn, an = str(hn).replace(" Full", ""), str(an).replace(" Full", "")
        hc, ac = ("H" + h[-3:]).upper(), ("A" + a[-3:]).upper()
        if sport in ("nfl", "nba"):
            hc, ac = h, a
        ev = f"26NOV27{ac}{hc}"
        mk = lambda t, ser, title, sub, bid, ask, strike=None: {
            "ticker": t, "event_ticker": f"{ser}-{ev}", "title": title, "yes_sub_title": sub,
            "yes_bid_dollars": f"{bid:.4f}", "yes_ask_dollars": f"{ask:.4f}", "last_price_dollars": "0.5",
            "volume_fp": "50", "close_time": "2026-11-29T03:00:00Z", "status": "active", "result": "",
            **({"floor_strike": strike} if strike is not None else {})}
        raw_g += [mk(f"X-{ev}-{hc}", "XG", f"{hn} wins", hn, 0.45, 0.50), mk(f"X-{ev}-{ac}", "XG", f"{an} wins", an, 0.48, 0.53)]
        raw_s.append(mk(f"XS-{ev}-{hc}3", "XS", f"{hn} wins by over 2.5 points", "", 0.40, 0.45, 2.5))
        raw_t.append(mk(f"XT-{ev}-T", "XT", f"Over {round(p['model_total']) - 3}.5 points", "", 0.45, 0.50,
                        round(p["model_total"]) - 3 + 0.5))
    rows = (kalshi._market_rows(raw_g, "game") + kalshi._market_rows(raw_s, "spread")
            + kalshi._market_rows(raw_t, "total"))
    if sport in ("nfl", "nba"):
        stat = "receiving_yards" if sport == "nfl" else "PTS"
        for pr in loaders.projections_for(ctx, (stat,))[:15]:
            g = ctx["game_info"].get(pr["team"])
            if not g:
                continue
            home, away = g["game"].split(" @ ")[1], g["game"].split(" @ ")[0]
            ev = f"26NOV27{away}{home}"
            raw_p.append({"ticker": f"XP-{ev}-{pr['player_norm'][:5]}", "event_ticker": f"XP-{ev}",
                          "title": f"{pr['player']}: 20+", "yes_sub_title": f"{pr['player']}: 20+",
                          "yes_bid_dollars": "0.40", "yes_ask_dollars": "0.45", "last_price_dollars": "0.4",
                          "volume_fp": "20", "close_time": "2026-11-29T03:00:00Z", "status": "active",
                          "floor_strike": 19.5})
        rows += kalshi._market_rows(raw_p, "prop", stat)
    return pd.DataFrame(rows), []


loaders.kalshi_markets = fake_kalshi
loaders.alt_lines = fake_alt_lines
loaders.game_lines = fake_game_lines
loaders.prop_lines = fake_prop_lines
st.session_state.clear()
# breaking news: ESPN report + an X post, plus a manual override
nfl_base = loaders.nfl_ctx()
buf_wr = nfl_base["players"][(nfl_base["players"]["team"] == "BUF") & (nfl_base["players"]["position"] == "WR")].iloc[0]
phi_rb = nfl_base["players"][(nfl_base["players"]["team"] == "PHI") & (nfl_base["players"]["position"] == "RB")].iloc[0]
loaders.fresh_injuries = lambda sport: pd.DataFrame({"name": [buf_wr["name"]], "name_norm": [buf_wr["name_norm"]],
                                                     "status": ["out"], "team_abbr": ["BUF"], "team_name": [""],
                                                     "comment": ["hamstring"]})
loaders.x_token = lambda: "tok"
loaders.x_posts = lambda sport, handles, hours: (pd.DataFrame({
    "id": ["9"], "url": ["https://x.com/i/web/status/9"], "time": [pd.Timestamp.now(tz="UTC")],
    "text": [f"{phi_rb['name']} has been ruled out for Sunday."]}), [])
st.session_state["nfl_mark_out"] = [phi_rb["name"]]
# preselect two combo legs for the first matched NFL game
from engine import combos as _combos, kalshi as _kalshi
_mk, _ = fake_kalshi("nfl")
_edges = _kalshi.score_markets(_mk, nfl_base["predictions"], __import__("engine.config", fromlist=["get_config"]).get_config("nfl"),
                               "nfl", loaders.projections_for(nfl_base, ("receiving_yards",)))
_g = _edges.drop_duplicates("game_key").sort_values("game").iloc[0]
_opts = _combos.leg_options(_edges, _g["game_key"])
_opts = _opts.assign(label=_opts["label"] + _opts["suspect"].map({True: " ⚠️", False: ""}))
st.session_state["nfl_cb_legs"] = list(_opts.drop_duplicates("ticker")["label"].head(3))
results["nfl+odds"] = run("pages/1_NFL.py", "nfl (odds key + news + override)")
results["nba+odds"] = run("pages/2_NBA.py", "nba (odds key)")

# DFS for NBA needs salaries: simulate live DK salaries in session
from data import nba as _nba  # noqa: E402
ctx = loaders.nba_ctx("2026-03-15")
pos_cycle = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SF/PF", "C"]
sal = pd.DataFrame([{"name": r["name"], "id": str(1000 + i), "positions": pos_cycle[i % 8].split("/"), "team": r["team"],
                     "salary": int(3500 + r["dk"] * 120) // 100 * 100, "site": "DK", "name_norm": r["name_norm"]}
                    for i, (_, r) in enumerate(ctx["players"].iterrows())])
st.session_state["nba_live_sal"] = sal
results["nba+dfs"] = run("pages/2_NBA.py", "nba (odds + DK salaries)")

# ---- track record with graded data
from engine import tracking  # noqa: E402
import tempfile, pathlib  # noqa: E402
tmp = pathlib.Path(tempfile.mkdtemp())
tracking.LOG_PATH, tracking.RESULTS_PATH = tmp / "p.csv", tmp / "r.csv"
preds = loaders.nba_ctx("2026-03-15")["predictions"]
games = loaders.nba_ctx("2026-03-15")["games"].tail(40)
fake_preds = pd.DataFrame({"game_id": games["game_id"], "date": games["date"], "home": games["home"], "away": games["away"],
                           "home_name": games["home"], "away_name": games["away"], "model_margin": np.random.normal(2, 6, 40),
                           "model_total": 225.0, "home_win": np.random.uniform(0.3, 0.85, 40), "home_spread": -2.0,
                           "total_line": 224.5, "market_home_win": 0.55, "known": True})
tracking.log_predictions(fake_preds, "nba", pd.Timestamp.now(tz="UTC"))
tracking.save_results(pd.DataFrame({"sport": "nba", "game_id": games["game_id"], "home_pts": games["home_pts"],
                                    "away_pts": games["away_pts"]}))
results["track"] = run("pages/5_Track_Record.py", "track record (graded)")
results["home2"] = run("app.py", "home (with record)")

bad = {k: v[0] for k, v in results.items() if v[0] == "CRASH"}
print("\nCRASHES:", bad or "none")
for k in ("nfl+odds", "nba+odds"):
    lg = results[k][1]
    print(k, "combo metrics:", [e[2] for e in lg if e[0] == "metric" and e[1] in ("Model hit chance", "Max quote to take")][:6],
          "| combo cards:", sum(1 for e in lg if e[0] == "markdown" and str(e[1]).startswith("**Combo")))
    print(k, "kalshi cards:", sum(1 for e in lg if e[0] == "markdown" and str(e[1]).startswith("#### Buy")),
          "| kalshi caption:", [e[1][:60] for e in lg if e[0] == "caption" and "matched contracts shown" in e[1]])
log = results["nfl+odds"][1]
print("news expander:", [e for e in log if e[0] == "expander" and "injury update" in str(e[1])])
print("override banner:", [e[1][:90] for e in log if e[0] == "info" and "overrides" in e[1]])
print("apply buttons:", sum(1 for e in log if e[0] == "button" and e[1] == "Apply"))
for k in ("nfl+odds", "nba+dfs"):
    log = results[k][1]
    print(f"\n--- {k}: warnings/info/errors")
    for e in log:
        if e[0] in ("warning", "error", "info", "success"):
            print("  ", e)


def count(log, kind, prefix):
    return sum(1 for e in log if e[0] == kind and str(e[1]).startswith(prefix))

for k in ("nfl+odds", "nba+odds", "nba+dfs"):
    log = results[k][1]
    print(k, "| tickets:", count(log, "markdown", "**Ticket"), "| lineups:", count(log, "markdown", "**Lineup"),
          "| downloads:", [e[1] for e in log if e[0] == "download"],
          "| 2x3x cards:", count(log, "markdown", "#### "), "| 2x3x caption:", [e[1][:40] for e in log if e[0] == "caption" and "plays ·" in e[1]],
          "| props captions:", [e[1][:70] for e in log if e[0] == "caption" and "props shown" in e[1]])
sys.exit(1 if bad else 0)
