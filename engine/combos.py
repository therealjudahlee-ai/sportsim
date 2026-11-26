"""Same-game combos (Kalshi "combos" / parlays) priced by simulating the whole game.

Legs in one game are connected: if a team wins big, its QB probably threw for a lot and the
total probably went over. Multiplying leg probabilities ignores that. Instead we simulate
the game N times:
  * final margin and total ~ Normal(model margin, sigma) and Normal(model total, sigma)
    -> both teams' scores
  * each player stat = his own projected distribution, tied to his TEAM'S score through a
    Gaussian copula with correlation rho (QB passing yards move with team points much more
    than rebounds do)
then count how often every selected leg hits at once.

Kalshi prices combos by request-for-quote, so the app gives you the FAIR price and the
highest quote worth taking after fees.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import norm

RHO = {  # correlation between a player's stat and his team's points
    "nfl": {"passing_yards": 0.45, "passing_tds": 0.50, "receiving_yards": 0.35, "receptions": 0.25,
            "rushing_yards": 0.25, "anytime_td": 0.35, "rushing_tds": 0.30, "receiving_tds": 0.30},
    "nba": {"PTS": 0.35, "REB": 0.10, "AST": 0.30, "FG3M": 0.25, "PRA": 0.35, "PR": 0.30, "PA": 0.35,
            "RA": 0.20, "STL": 0.0, "BLK": 0.0},
}


def break_even_price(p: float, fee_rate: float = 0.07) -> float:
    """Highest price q where buying still has non-negative EV: p - q - fee_rate*q*(1-q) = 0."""
    p = float(np.clip(p, 0.0, 1.0))
    a, b = fee_rate, 1.0 + fee_rate
    return float((b - np.sqrt(b * b - 4 * a * p)) / (2 * a)) if a > 0 else p


def simulate(legs: list[dict], game: dict, cfg, sport: str, proj_idx: dict, n: int = 40_000,
             seed: int = 11) -> dict:
    """legs: rows from kalshi.score_markets (need kind, team_side, strike, stat, player_norm) plus
    'pick' = 'YES' or 'NO'. game: the matched prediction row (model_margin, model_total, home, away)."""
    rng = np.random.default_rng(seed)
    margin = rng.normal(game["model_margin"], cfg.margin_sigma, n)
    total = rng.normal(game["model_total"], cfg.total_sigma, n)
    home_pts, away_pts = (total + margin) / 2, (total - margin) / 2
    z_team = {"home": (home_pts - home_pts.mean()) / (home_pts.std() + 1e-9),
              "away": (away_pts - away_pts.mean()) / (away_pts.std() + 1e-9)}
    hits = np.ones(n, dtype=bool)
    marg = []
    for leg in legs:
        k = leg["kind"]
        if k == "game":
            team_margin = margin if leg["team_side"] == "home" else -margin
            yes = team_margin > 0
        elif k == "spread":
            team_margin = margin if leg["team_side"] == "home" else -margin
            yes = team_margin > leg["strike"]
        elif k == "total":
            yes = total > leg["strike"]
        elif k == "prop":
            pr = proj_idx.get((leg["player_norm"], leg["stat"]))
            if pr is None:
                raise ValueError(f"No projection for {leg.get('contract')}")
            side = "home" if pr["team"] == game["home"] else "away"
            rho = RHO.get(sport, {}).get(leg["stat"], 0.25)
            latent = rho * z_team[side] + np.sqrt(1 - rho * rho) * rng.standard_normal(n)
            stat = pr["dist"]._frozen().ppf(np.clip(norm.cdf(latent), 1e-6, 1 - 1e-6))
            yes = stat > leg["strike"]
        else:
            raise ValueError(k)
        won = yes if leg.get("pick", "YES") == "YES" else ~yes
        marg.append(float(won.mean()))
        hits &= won
    joint = float(hits.mean())
    indep = float(np.prod(marg)) if marg else 0.0
    return {"joint": joint, "independent": indep, "leg_probs": marg,
            "correlation_boost": joint / indep if indep > 0 else np.nan,
            "fair_price": joint, "max_quote": break_even_price(joint),
            "max_quote_maker": break_even_price(joint, 0.0175)}


def leg_options(edges: pd.DataFrame, game_key: str) -> pd.DataFrame:
    """Both sides of every contract in one game, with price and model probability per side."""
    g = edges[edges["game_key"] == game_key]
    rows = []
    for _, r in g.iterrows():
        if np.isfinite(r["yes_ask"]) and 0 < r["yes_ask"] < 1:
            rows.append({**r.to_dict(), "pick": "YES", "leg_price": r["yes_ask"], "leg_prob": r["p_yes"]})
        if np.isfinite(r["yes_bid"]) and 0 < r["yes_bid"] < 1:
            rows.append({**r.to_dict(), "pick": "NO", "leg_price": 1 - r["yes_bid"], "leg_prob": 1 - r["p_yes"]})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["label"] = out["pick"] + " · " + out["contract"] + " (" + (out["leg_price"] * 100).round(0).astype(int) \
            .astype(str) + "¢)"
    return out


def likely_sides(options: pd.DataFrame, min_prob: float = 0.65, max_price: float = 0.90) -> pd.DataFrame:
    """For each contract, the side the model thinks is more likely, if it's at least min_prob and the
    price isn't so high that it pays almost nothing."""
    if options is None or options.empty:
        return options
    o = options[(options["leg_prob"] >= min_prob) & (options["leg_price"] <= max_price)]
    if "suspect" in o.columns:
        o = o[~o["suspect"]]
    return o.sort_values("leg_prob", ascending=False).drop_duplicates("ticker")


def suggest(options: pd.DataFrame, game: dict, cfg, sport: str, proj_idx: dict, n_legs: int = 2,
            top: int = 5, min_joint: float = 0.08, pool: int = 12, mode: str = "value",
            min_prob: float = 0.65, max_price: float = 0.90) -> list[dict]:
    """Search combos in one game.
    mode="value":  legs that each beat their Kalshi price; ranked by fair price vs. legs-bought-separately.
    mode="likely": the model's most likely legs (each >= min_prob, price <= max_price); ranked by the
                   chance the whole combo hits. Kalshi's quote is the real test either way."""
    if options is None or options.empty:
        return []
    if mode == "likely":
        cand = likely_sides(options, min_prob, max_price)
    else:
        cand = options[(options["leg_prob"] > options["leg_price"]) & ~options.get("suspect", False)]
    cand = cand.sort_values("leg_prob", ascending=False)
    # one leg per contract, and don't pair a player prop ladder with itself
    cand = cand.drop_duplicates("ticker").head(pool).to_dict("records")
    out = []
    for combo in itertools.combinations(cand, n_legs):
        keys = [(c["kind"], c.get("player_norm"), c.get("stat"), c.get("team_side")) for c in combo]
        if len(set(keys)) < n_legs:     # two rungs of the same ladder are redundant
            continue
        try:
            sim = simulate(list(combo), game, cfg, sport, proj_idx, n=15_000)
        except ValueError:
            continue
        if sim["joint"] < min_joint:
            continue
        naive = float(np.prod([c["leg_price"] for c in combo]))
        out.append({"legs": combo, **sim, "naive_price": naive,
                    "value_ratio": sim["joint"] / naive if naive > 0 else np.nan})
    out.sort(key=lambda x: x["joint"] if mode == "likely" else x["value_ratio"], reverse=True)
    return out[:top]
