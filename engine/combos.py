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


def leg_options(edges: pd.DataFrame, game_key: str | None = None) -> pd.DataFrame:
    """Both sides of every contract (in one game, or all games), with price and model probability per side."""
    g = edges if game_key is None else edges[edges["game_key"] == game_key]
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


def _clean(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def ladder_key(leg: dict) -> tuple:
    """Rungs of one ladder (same game, market type, team/player, stat) are nested bets — e.g. UNDER 226.5
    already implies UNDER 228.5 — so a combo may hold at most one rung per ladder."""
    return (_clean(leg.get("game_key")), _clean(leg.get("kind")), _clean(leg.get("team_side")),
            _clean(leg.get("player_norm")), _clean(leg.get("stat")))


def best_rung_per_ladder(options: pd.DataFrame) -> pd.DataFrame:
    """Among qualifying rungs of each ladder, keep the cheapest (= biggest payout)."""
    if options is None or options.empty:
        return options
    o = options.copy()
    o["_ladder"] = [ladder_key(r) for r in o.to_dict("records")]
    o = o.sort_values("leg_price").drop_duplicates("_ladder", keep="first")
    return o.drop(columns="_ladder")


def simulate_multi(legs: list[dict], games: dict, cfg, sport: str, proj_idx: dict, n: int = 40_000) -> dict:
    """Legs can span several games: simulate each game separately (they're independent) and multiply."""
    by_game: dict[str, list] = {}
    for leg in legs:
        by_game.setdefault(leg["game_key"], []).append(leg)
    joint, indep = 1.0, 1.0
    probs = {}
    for gk, gl in by_game.items():
        if gk not in games:
            raise ValueError(f"No prediction for game {gk}")
        sim = simulate(gl, games[gk], cfg, sport, proj_idx, n=n)
        joint *= sim["joint"]
        indep *= sim["independent"]
        for leg, p in zip(gl, sim["leg_probs"]):
            probs[id(leg)] = p
    return {"joint": joint, "independent": indep, "leg_probs": [probs[id(l)] for l in legs],
            "correlation_boost": joint / indep if indep > 0 else np.nan, "fair_price": joint,
            "max_quote": break_even_price(joint), "max_quote_maker": break_even_price(joint, 0.0175),
            "n_games": len(by_game)}


def likely_sides(options: pd.DataFrame, min_prob: float = 0.65, max_price: float = 0.90) -> pd.DataFrame:
    """For each contract, the side the model thinks is more likely, if it's at least min_prob and the
    price isn't so high that it pays almost nothing."""
    if options is None or options.empty:
        return options
    o = options[(options["leg_prob"] >= min_prob) & (options["leg_price"] <= max_price)]
    if "suspect" in o.columns:
        o = o[~o["suspect"]]
    o = best_rung_per_ladder(o.sort_values("leg_prob", ascending=False).drop_duplicates("ticker"))
    return o.sort_values("leg_prob", ascending=False)


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
        cand = cand.assign(_edge=cand["leg_prob"] - cand["leg_price"]).sort_values("_edge", ascending=False)
        cand["_ladder"] = [ladder_key(r) for r in cand.to_dict("records")]
        cand = cand.drop_duplicates("_ladder").drop(columns=["_edge", "_ladder"])
    cand = cand.sort_values("leg_prob", ascending=False)
    cand = cand.drop_duplicates("ticker").head(pool).to_dict("records")
    games = game if isinstance(game, dict) and "model_margin" not in game else None
    out = []
    for combo in itertools.combinations(cand, n_legs):
        if len({ladder_key(c) for c in combo}) < n_legs:     # never two rungs of one ladder
            continue
        try:
            sim = (simulate_multi(list(combo), games, cfg, sport, proj_idx, n=15_000) if games is not None
                   else simulate(list(combo), game, cfg, sport, proj_idx, n=15_000))
        except ValueError:
            continue
        if sim["joint"] < min_joint:
            continue
        naive = float(np.prod([c["leg_price"] for c in combo]))
        out.append({"legs": combo, **sim, "naive_price": naive,
                    "value_ratio": sim["joint"] / naive if naive > 0 else np.nan})
    out.sort(key=lambda x: x["joint"] if mode == "likely" else x["value_ratio"], reverse=True)
    return out[:top]


def suggest_props(options: pd.DataFrame, games: dict, cfg, sport: str, proj_idx: dict, n_legs: int = 3,
                  min_prob: float = 0.65, max_price: float = 0.90, top: int = 5, pool: int = 12) -> list[dict]:
    """Most-likely player-prop combos (one rung per player/stat; may span games).
    Up to 4 legs: exhaustive search over the `pool` most likely legs. 5+: greedy build from each top leg."""
    cand = likely_sides(options, min_prob, max_price)
    if cand is None or cand.empty or len(cand) < n_legs:
        return []
    cand = cand.head(pool if n_legs <= 4 else 20).to_dict("records")

    def run(combo):
        try:
            sim = simulate_multi(list(combo), games, cfg, sport, proj_idx, n=12_000)
        except ValueError:
            return None
        naive = float(np.prod([c["leg_price"] for c in combo]))
        return {"legs": list(combo), **sim, "naive_price": naive}

    out = []
    if n_legs <= 4:
        for combo in itertools.combinations(cand, n_legs):
            if len({ladder_key(c) for c in combo}) == n_legs:
                r = run(combo)
                if r:
                    out.append(r)
    else:
        for start in cand[:top]:
            chosen = [start]
            while len(chosen) < n_legs:
                best, best_p = None, -1.0
                for c in cand:
                    if c in chosen or ladder_key(c) in {ladder_key(x) for x in chosen}:
                        continue
                    r = run(chosen + [c])
                    if r and r["joint"] > best_p:
                        best, best_p = c, r["joint"]
                if best is None:
                    break
                chosen.append(best)
            if len(chosen) == n_legs:
                r = run(chosen)
                if r and {tuple(sorted(l["ticker"] for l in r["legs"]))} != {
                        tuple(sorted(l["ticker"] for l in o["legs"])) for o in out}:
                    out.append(r)
    out.sort(key=lambda x: x["joint"], reverse=True)
    seen, uniq = set(), []
    for o in out:
        k = tuple(sorted(l["ticker"] for l in o["legs"]))
        if k not in seen:
            seen.add(k)
            uniq.append(o)
    return uniq[:top]
