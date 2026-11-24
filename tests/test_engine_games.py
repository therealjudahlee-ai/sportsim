import numpy as np
import pandas as pd
from dataclasses import replace

from engine.config import DEFAULTS
from engine.ratings import RatingsModel, fit_ratings
from engine.gamesim import game_probs, predict_slate, market_home_win
from tests.synth import synth_games


def test_ratings_recover_truth_points():
    games, true = synth_games(poss=False, games_per_team=60, sigma=10)
    cfg = replace(DEFAULTS["nfl"], half_life_days=10_000, ridge_alpha=1.0)
    end = games["date"].max() + pd.Timedelta(days=1)
    r = fit_ratings(games, end, cfg, season=2025)
    est = np.array([r.off[t] - r.deff[t] for t in sorted(true[2025]["off"])])
    tru = np.array([true[2025]["off"][t] - true[2025]["def"][t] for t in sorted(true[2025]["off"])])
    assert np.corrcoef(est, tru)[0, 1] > 0.9
    assert abs(r.hfa - 2.5) < 1.5


def test_ratings_recover_truth_possessions():
    games, true = synth_games(poss=True, games_per_team=60, sigma=10)
    cfg = replace(DEFAULTS["nba"], half_life_days=10_000, ridge_alpha=1.0)
    end = games["date"].max() + pd.Timedelta(days=1)
    r = fit_ratings(games, end, cfg, season=2025)
    teams = sorted(true[2025]["off"])
    est = np.array([r.off[t] - r.deff[t] for t in teams])
    tru = np.array([true[2025]["off"][t] - true[2025]["def"][t] for t in teams])
    assert np.corrcoef(est, tru)[0, 1] > 0.9
    pace_est = np.array([r.pace[t] for t in teams])
    pace_tru = np.array([true[2025]["pace"][t] for t in teams])
    assert np.corrcoef(pace_est, pace_tru)[0, 1] > 0.8


def test_prior_helps_early_season():
    games, true = synth_games(poss=False, games_per_team=40, carry=0.8, seed=3)
    cfg = DEFAULTS["nfl"]
    model = RatingsModel(games, cfg)
    s25 = games[games["season"] == 2025]
    early = s25["date"].min() + pd.Timedelta(days=5)  # only a handful of games played
    with_prior = model.ratings_as_of(early, 2025)
    no_prior = fit_ratings(games, early, cfg, prior=None, season=2025)
    teams = sorted(true[2025]["off"])
    tru = np.array([true[2025]["off"][t] - true[2025]["def"][t] for t in teams])
    e1 = np.array([with_prior.off.get(t, 0) - with_prior.deff.get(t, 0) for t in teams])
    e0 = np.array([no_prior.off.get(t, 0) - no_prior.deff.get(t, 0) for t in teams])
    assert np.corrcoef(e1, tru)[0, 1] > np.corrcoef(e0, tru)[0, 1]


def test_closed_form_matches_simulation():
    cfg = DEFAULTS["nfl"]
    p = game_probs(3.0, 45.0, cfg, home_spread=-2.5, total_line=44.5)
    sims = np.random.default_rng(1).normal(3.0, cfg.margin_sigma, 2_000_000)
    assert abs(p["home_win"] - (sims > 0).mean()) < 0.002
    assert abs(p["cover_home"] - ((sims - 2.5) > 0).mean()) < 0.002
    assert abs(p["blowout"] - (np.abs(sims) >= cfg.blowout_margin).mean()) < 0.002


def test_market_win_prefers_moneyline():
    cfg = DEFAULTS["nfl"]
    p = market_home_win(cfg, home_spread=-3, home_ml=-150, away_ml=130)
    assert 0.55 < p < 0.62
    p2 = market_home_win(cfg, home_spread=-3)
    assert 0.55 < p2 < 0.62


def test_predict_slate_columns():
    games, _ = synth_games(poss=True, games_per_team=30)
    cfg = DEFAULTS["nba"]
    model = RatingsModel(games, cfg)
    r = model.ratings_as_of(games["date"].max(), 2025)
    slate = games.tail(5)[["game_id", "date", "home", "away", "neutral", "home_spread", "total_line"]].copy()
    slate["home_rest"] = [1, 2, 3, 1, 2]
    slate["away_rest"] = 2
    out = predict_slate(r, slate, cfg, team_adj={slate.iloc[0]["home"]: -3.0})
    for c in ["model_margin", "home_win", "cover_home", "over_prob", "win_edge", "pick", "notes"]:
        assert c in out.columns
    assert out["home_win"].between(0, 1).all()
    assert "back-to-back" in out.iloc[0]["notes"]
