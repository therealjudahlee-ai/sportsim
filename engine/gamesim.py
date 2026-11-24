"""Game-level probabilities.

Margin ~ Normal(projected margin, sigma). Under that model, win / cover /
blowout / over probabilities have exact closed forms, so there's no need for
1,000,000 random draws per game: norm.cdf gives the identical answer
instantly and with zero simulation noise.

Sign convention everywhere: `home_spread` is the sportsbook line for the HOME
team (negative = home favored, e.g. -3.5). Market-implied home margin is
therefore -home_spread.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .common import devig_pair, prob_to_american
from .config import SportConfig
from .ratings import TeamRatings


def game_probs(margin: float, total: float, cfg: SportConfig,
               home_spread: float | None = None, total_line: float | None = None) -> dict:
    s = cfg.margin_sigma
    out = {
        "home_win": float(norm.cdf(margin / s)),
        "blowout": float(norm.sf(cfg.blowout_margin, margin, s) + norm.cdf(-cfg.blowout_margin, margin, s)),
    }
    out["away_win"] = 1.0 - out["home_win"]
    if home_spread is not None and np.isfinite(home_spread):
        out["cover_home"] = float(norm.cdf((margin + home_spread) / s))
    if total_line is not None and np.isfinite(total_line):
        out["over"] = float(norm.sf(total_line, total, cfg.total_sigma))
    return out


def market_home_win(cfg: SportConfig, home_spread=None, home_ml=None, away_ml=None) -> float:
    """Market win probability: de-vigged moneylines if we have them, else the spread."""
    if home_ml is not None and away_ml is not None and np.isfinite(home_ml) and np.isfinite(away_ml):
        return devig_pair(home_ml, away_ml)[0]
    if home_spread is not None and np.isfinite(home_spread):
        return float(norm.cdf(-home_spread / cfg.margin_sigma))
    return np.nan


def margin_density(margin: float, cfg: SportConfig, n: int = 241) -> pd.DataFrame:
    """For charts: density of the final home margin."""
    s = cfg.margin_sigma
    x = np.linspace(margin - 3.5 * s, margin + 3.5 * s, n)
    return pd.DataFrame({"margin": x, "density": norm.pdf(x, margin, s)})


def _num_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def apply_market(preds: pd.DataFrame, cfg: SportConfig) -> pd.DataFrame:
    """(Re)compute every market-relative column from model_margin/model_total and the
    current market fields (home_spread, total_line, home_ml, away_ml, market_home_win_ml).
    Call again after swapping in fresher lines (e.g. from The Odds API)."""
    if preds is None or preds.empty:
        return preds
    df = preds.copy()
    s, ts = cfg.margin_sigma, cfg.total_sigma
    m, t = df["model_margin"].astype(float), df["model_total"].astype(float)
    hs, tl = _num_col(df, "home_spread"), _num_col(df, "total_line")
    df["home_win"] = norm.cdf(m / s)
    df["away_win"] = 1 - df["home_win"]
    df["blowout"] = norm.sf(cfg.blowout_margin, m, s) + norm.cdf(-cfg.blowout_margin, m, s)

    ml_prob = _num_col(df, "market_home_win_ml").copy()
    if "home_ml" in df.columns and "away_ml" in df.columns:
        hml = pd.to_numeric(df["home_ml"], errors="coerce")
        aml = pd.to_numeric(df["away_ml"], errors="coerce")
        ok = hml.notna() & aml.notna() & ml_prob.isna()
        ml_prob[ok] = [devig_pair(h, a)[0] for h, a in zip(hml[ok], aml[ok])]
    df["market_home_win"] = ml_prob.where(ml_prob.notna(), pd.Series(norm.cdf(-hs / s), index=df.index).where(hs.notna()))
    df["market_margin"] = -hs

    home_pick = df["home_win"] >= 0.5
    df["pick"] = np.where(home_pick, df["home"], df["away"])
    if "home_name" in df.columns:
        df["pick_name"] = np.where(home_pick, df["home_name"], df["away_name"])
    else:
        df["pick_name"] = df["pick"]
    df["pick_prob"] = np.where(home_pick, df["home_win"], df["away_win"])
    df["pick_fair_ml"] = [prob_to_american(p) for p in df["pick_prob"]]
    df["market_pick_prob"] = np.where(home_pick, df["market_home_win"], 1 - df["market_home_win"])
    df["win_edge"] = df["pick_prob"] - df["market_pick_prob"]

    df["cover_home"] = np.where(hs.notna(), norm.cdf((m + hs) / s), np.nan)
    df["ats_pick"] = np.where(hs.isna(), None, np.where(df["cover_home"] >= 0.5, "home", "away"))
    df["ats_prob"] = np.where(hs.notna(), np.maximum(df["cover_home"], 1 - df["cover_home"]), np.nan)
    df["spread_edge_pts"] = m - (-hs)
    df["over_prob"] = np.where(tl.notna(), norm.sf(tl, t, ts), np.nan)
    df["total_pick"] = np.where(tl.isna(), None, np.where(df["over_prob"] >= 0.5, "over", "under"))
    df["total_prob"] = np.where(tl.notna(), np.maximum(df["over_prob"], 1 - df["over_prob"]), np.nan)
    df["total_edge_pts"] = t - tl
    return df


def predict_slate(ratings: TeamRatings, slate: pd.DataFrame, cfg: SportConfig,
                  team_adj: dict | None = None, total_mult: dict | None = None) -> pd.DataFrame:
    """Project every game on a slate.

    team_adj:   {team_key: points} added to that team's projected score
                (injuries). Negative = worse.
    total_mult: {game_id: multiplier} on both teams' points (e.g. wind).
    """
    team_adj = team_adj or {}
    total_mult = total_mult or {}
    rows = []
    for _, g in slate.iterrows():
        p = ratings.project(g["home"], g["away"], bool(g.get("neutral", False)))
        hp = p["home_pts"] + team_adj.get(g["home"], 0.0)
        ap = p["away_pts"] + team_adj.get(g["away"], 0.0)
        notes = []
        if cfg.rest_penalty_b2b:
            for side in ("home", "away"):
                rest = g.get(f"{side}_rest", np.nan)
                if pd.notna(rest) and rest <= 1:
                    if side == "home":
                        hp -= cfg.rest_penalty_b2b
                    else:
                        ap -= cfg.rest_penalty_b2b
                    notes.append(f"{g.get(side + '_name', g[side])} on back-to-back")
        mult = total_mult.get(g.get("game_id"), 1.0)
        if mult != 1.0:
            hp, ap = hp * mult, ap * mult
            notes.append("wind")
        for side in ("home", "away"):
            d = team_adj.get(g[side], 0.0)
            if d <= -0.5:
                notes.append(f"{g.get(side + '_name', g[side])} injuries {d:+.1f}")
        rows.append({
            **{k: g[k] for k in slate.columns},
            "model_home_pts": hp, "model_away_pts": ap,
            "model_margin": hp - ap, "model_total": hp + ap,
            "known": p["home_known"] and p["away_known"],
            "notes": "; ".join(notes),
        })
    return apply_market(pd.DataFrame(rows), cfg)
