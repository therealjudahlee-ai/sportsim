"""Walk-forward backtest + parameter tuning.

For every game date in the evaluation seasons, ratings are fit using ONLY
games before that date (plus last season's ratings as the prior), then that
day's games are predicted. No peeking. The tuner searches recency half-life,
ridge strength and prior carry-over to minimize margin error, then sets the
margin/total sigmas to the actual residual spread so probabilities are
calibrated.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import SportConfig
from .ratings import RatingsModel


def walk_forward(games: pd.DataFrame, cfg: SportConfig, eval_seasons: list[int],
                 min_date_gap_days: int = 0, model_factory=None) -> pd.DataFrame:
    """model_factory(cfg) -> object with ratings_as_of(as_of, season); default = score-only ratings."""
    model = model_factory(cfg) if model_factory else RatingsModel(games, cfg)
    done = games.dropna(subset=["home_pts", "away_pts"])
    rows = []
    for s in eval_seasons:
        sg = done[done["season"] == s]
        last_fit_day, ratings = None, None
        for d, day in sg.groupby("date"):
            if ratings is None or last_fit_day is None or (d - last_fit_day).days >= min_date_gap_days:
                ratings = model.ratings_as_of(d, s)
                last_fit_day = d
            for _, g in day.iterrows():
                p = ratings.project(g["home"], g["away"], bool(g["neutral"]))
                rows.append({
                    "game_id": g["game_id"], "date": d, "season": s, "home": g["home"], "away": g["away"],
                    "pred_margin": p["margin"], "pred_total": p["total"],
                    "act_margin": g["home_pts"] - g["away_pts"], "act_total": g["home_pts"] + g["away_pts"],
                    "home_spread": g.get("home_spread", np.nan), "total_line": g.get("total_line", np.nan),
                    "known": p["home_known"] and p["away_known"],
                })
    return pd.DataFrame(rows)


def score(bt: pd.DataFrame, margin_sigma: float, total_sigma: float) -> dict:
    if bt.empty:
        return {}
    res = bt["act_margin"] - bt["pred_margin"]
    p = norm.cdf(bt["pred_margin"] / margin_sigma).clip(1e-4, 1 - 1e-4)
    won = (bt["act_margin"] > 0).astype(float)
    won[bt["act_margin"] == 0] = 0.5
    out = {
        "n": int(len(bt)),
        "margin_mae": float(res.abs().mean()),
        "margin_rmse": float(np.sqrt((res ** 2).mean())),
        "total_rmse": float(np.sqrt(((bt["act_total"] - bt["pred_total"]) ** 2).mean())),
        "su_accuracy": float(((p >= 0.5) == (bt["act_margin"] > 0))[bt["act_margin"] != 0].mean()),
        "brier": float(((p - won) ** 2).mean()),
        "logloss": float((-(won * np.log(p) + (1 - won) * np.log(1 - p))).mean()),
    }
    sp = bt.dropna(subset=["home_spread"])
    if len(sp):
        mkt = -sp["home_spread"]
        out["market_margin_rmse"] = float(np.sqrt(((sp["act_margin"] - mkt) ** 2).mean()))
        out["model_margin_rmse_same_games"] = float(np.sqrt(((sp["act_margin"] - sp["pred_margin"]) ** 2).mean()))
        ats = sp["act_margin"] + sp["home_spread"]
        pick_home = sp["pred_margin"] > mkt
        res_ats = ((ats > 0) == pick_home)[ats != 0]
        out["ats_pct"] = float(res_ats.mean())
        out["ats_n"] = int(len(res_ats))
        edge = (sp["pred_margin"] - mkt).abs()
        for thr in (2, 4):
            m = (edge >= thr) & (ats != 0)
            if m.sum():
                out[f"ats_pct_edge{thr}+"] = float(((ats > 0) == pick_home)[m].mean())
                out[f"ats_n_edge{thr}+"] = int(m.sum())
    return out


BOUNDED = {"prior_regress": (0.0, 0.95), "epa_weight": (0.0, 1.0)}


def _expand(grid: dict, best: dict) -> dict | None:
    """If the best value sits on the edge of its grid, push that axis outward."""
    new, moved = {}, False
    for k, vals in grid.items():
        vals = sorted(vals)
        b = best[k]
        if k in BOUNDED:
            lo_b, hi_b = BOUNDED[k]
            step = 0.15
            if b == vals[-1] and b < hi_b:
                vals = [b, min(hi_b, b + step)]
                moved = True
            elif b == vals[0] and b > lo_b:
                vals = [max(lo_b, b - step), b]
                moved = True
            else:
                vals = [b]
        else:
            lo_lim, hi_lim = {"half_life_days": (5.0, 2000.0), "ridge_alpha": (0.1, 100.0)}.get(k, (1e-3, 1e6))
            if b == vals[-1] and b * 2 <= hi_lim:
                vals = [b, b * 2]
                moved = True
            elif b == vals[0] and b / 2 >= lo_lim:
                vals = [b / 2, b]
                moved = True
            else:
                vals = [b]
        new[k] = sorted(set(round(v, 4) for v in vals))
    return new if moved else None


def tune(games: pd.DataFrame, cfg: SportConfig, eval_seasons: list[int], grid: dict | None = None,
         min_date_gap_days: int = 0, log=print, max_rounds: int = 4,
         model_factory=None) -> tuple[dict, pd.DataFrame]:
    """Grid search; if the winner lands on the edge of the grid, keep searching outward
    (up to max_rounds) instead of stopping at an arbitrary boundary."""
    grid = grid or {
        "half_life_days": sorted({cfg.half_life_days * f for f in (0.5, 1.0, 2.0)}),
        "ridge_alpha": sorted({cfg.ridge_alpha * f for f in (0.5, 1.0, 2.0)}),
        "prior_regress": sorted({max(0.0, min(1.0, cfg.prior_regress + d)) for d in (-0.2, 0.0, 0.15)}),
    }
    if model_factory is not None and "epa_weight" not in grid:
        grid["epa_weight"] = [0.3, 0.6, 0.9]
    keys = list(grid)
    seen: dict[tuple, dict] = {}
    t0 = time.time()
    for rnd in range(max_rounds):
        for combo in itertools.product(*grid.values()):
            ck = tuple(round(v, 4) for v in combo)
            if ck in seen:
                continue
            params = dict(zip(keys, combo))
            bt = walk_forward(games, replace(cfg, **params), eval_seasons, min_date_gap_days, model_factory)
            bt = bt[bt["known"]]
            rmse = float(np.sqrt(((bt["act_margin"] - bt["pred_margin"]) ** 2).mean()))
            seen[ck] = {**params, "margin_rmse": rmse, "n": len(bt)}
            log(f"  [round {rnd + 1}] {params} -> margin RMSE {rmse:.3f}  ({time.time() - t0:.0f}s)")
        table = pd.DataFrame(seen.values()).sort_values("margin_rmse").reset_index(drop=True)
        best = table.iloc[0][keys].to_dict()
        nxt = _expand(grid, best)
        if nxt is None:
            log(f"  best setting is inside the search range after round {rnd + 1}")
            break
        log(f"  best was on the edge of the range; searching further: {nxt}")
        grid = nxt
    c = replace(cfg, **best)
    bt = walk_forward(games, c, eval_seasons, min_date_gap_days, model_factory)
    bt = bt[bt["known"]]
    best["margin_sigma"] = float((bt["act_margin"] - bt["pred_margin"]).std())
    best["total_sigma"] = float((bt["act_total"] - bt["pred_total"]).std())
    return best, table
