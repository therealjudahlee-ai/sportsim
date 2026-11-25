import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from engine import tracking
from engine.backtest import score, tune, walk_forward
from engine.config import DEFAULTS
from engine.market import merge_game_lines
from engine.gamesim import predict_slate
from engine.ratings import RatingsModel
from tests.synth import synth_games


def test_walk_forward_and_tune():
    games, _ = synth_games(n_teams=16, seasons=(2024, 2025, 2026), games_per_team=24, sigma=11, seed=2)
    cfg = DEFAULTS["nfl"]
    bt = walk_forward(games, cfg, [2025, 2026])
    assert len(bt) == (games["season"] >= 2025).sum()
    # no peeking: first game of the season uses only the prior
    s = score(bt[bt["known"]], cfg.margin_sigma, cfg.total_sigma)
    assert 0.5 < s["su_accuracy"] < 0.9 and "ats_pct" in s
    grid = {"half_life_days": [60, 240], "ridge_alpha": [3.0], "prior_regress": [0.6]}
    best, table = tune(games, cfg, [2025, 2026], grid=grid, log=lambda *_: None, max_rounds=1)
    assert len(table) == 2 and 8 < best["margin_sigma"] < 16


def test_log_grade_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "LOG_PATH", tmp_path / "p.csv")
    monkeypatch.setattr(tracking, "RESULTS_PATH", tmp_path / "r.csv")
    games, _ = synth_games(n_teams=10, seasons=(2025,), games_per_team=20, seed=1, poss=True)
    cfg = DEFAULTS["nba"]
    r = RatingsModel(games, cfg).ratings_as_of(games["date"].max(), 2025)
    slate = games.tail(20)[["game_id", "date", "home", "away", "neutral", "home_spread", "total_line"]].copy()
    preds = predict_slate(r, slate, cfg)
    tracking.log_predictions(preds, "nba", pd.Timestamp("2026-01-01 09:00", tz="America/New_York"),
                             path=tmp_path / "p.csv")
    moved = preds.copy()
    moved["home_spread"] = moved["home_spread"] - 1.0   # line moves toward home later
    tracking.log_predictions(moved, "nba", pd.Timestamp("2026-01-01 18:00", tz="America/New_York"),
                             path=tmp_path / "p.csv")
    tracking.save_results(pd.DataFrame({"sport": "nba", "game_id": slate["game_id"],
                                        "home_pts": games.tail(20)["home_pts"].values,
                                        "away_pts": games.tail(20)["away_pts"].values}), path=tmp_path / "r.csv")
    g = tracking.graded(tracking.load_log(tmp_path / "p.csv"), pd.read_csv(tmp_path / "r.csv", dtype={"game_id": str}))
    assert len(g) == 20
    # graded against the FIRST logged line; CLV positive for home picks, negative for away
    pick_home = g["model_margin"] > -g["home_spread"]
    assert np.allclose(g.loc[pick_home, "clv_pts"], 1.0) and np.allclose(g.loc[~pick_home, "clv_pts"], -1.0)
    s = tracking.summary(g)
    assert s["games"] == 20 and 0 <= s["brier"] <= 1
    cal = tracking.calibration(g)
    assert cal["n"].sum() == 20


def test_merge_game_lines():
    cfg = DEFAULTS["nba"]
    preds = pd.DataFrame({"game_id": ["1"], "home": ["BOS"], "away": ["NYK"], "home_name": ["Boston"],
                          "away_name": ["NY"], "model_margin": [5.0], "model_total": [225.0],
                          "home_spread": [np.nan], "total_line": [np.nan]})
    lines = pd.DataFrame({"home_name": ["Boston Celtics"], "away_name": ["New York Knicks"], "home_spread": [-4.5],
                          "total_line": [221.5], "market_home_win_ml": [0.64], "n_books": [7]})
    out, n = merge_game_lines(preds, lines, cfg)
    assert n == 1 and out.iloc[0]["home_spread"] == -4.5 and out.iloc[0]["market_home_win"] == 0.64
    assert out.iloc[0]["ats_pick"] == "home" and out.iloc[0]["total_pick"] == "over"


def test_refresh_job(monkeypatch, tmp_path):
    from tests import fakes
    from data import espn
    monkeypatch.setattr(tracking, "LOG_PATH", tmp_path / "p.csv")
    monkeypatch.setattr(tracking, "RESULTS_PATH", tmp_path / "r.csv")
    frames = fakes.fake_nba_parquet()
    monkeypatch.setitem(sys.modules, "nba_api", None)
    monkeypatch.setattr(pd, "read_parquet", lambda url: frames[int(url.split("_")[-1].split(".")[0])])
    monkeypatch.setattr(espn, "get_json", fakes.fake_espn_json("nba", [("BOS", "NY", "1", "2", "BOS -6.5")]))
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    from jobs import refresh
    monkeypatch.setattr(refresh, "now_et", lambda: pd.Timestamp("2026-03-15 09:00", tz="America/New_York"))
    refresh.main(["nba"])
    log = pd.read_csv(tmp_path / "p.csv")
    assert len(log) == 1 and log.iloc[0]["home"] == "BOS"


def test_tune_expands_past_grid_edge():
    from engine.backtest import _expand
    g = {"half_life_days": [20, 40, 80], "ridge_alpha": [2.5, 5, 10], "prior_regress": [0.4, 0.6, 0.75]}
    nxt = _expand(g, {"half_life_days": 80, "ridge_alpha": 2.5, "prior_regress": 0.6})
    assert nxt["half_life_days"] == [80, 160] and nxt["ridge_alpha"] == [1.25, 2.5] and nxt["prior_regress"] == [0.6]
    assert _expand(g, {"half_life_days": 40, "ridge_alpha": 5, "prior_regress": 0.6}) is None
    games, _ = synth_games(n_teams=12, seasons=(2024, 2025, 2026), games_per_team=16, seed=5)
    best, table = tune(games, DEFAULTS["nfl"], [2025, 2026],
                       grid={"half_life_days": [30, 60], "ridge_alpha": [3.0], "prior_regress": [0.6]},
                       log=lambda *_: None, max_rounds=3)
    assert len(table) >= 2 and "margin_sigma" in best
