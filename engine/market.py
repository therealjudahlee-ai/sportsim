"""Swap fresher sportsbook lines (The Odds API) into model predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SportConfig
from .gamesim import apply_market
from .odds_api import NBA_TEAMS, NFL_TEAMS, match_team


def merge_game_lines(preds: pd.DataFrame, lines: pd.DataFrame, cfg: SportConfig) -> tuple[pd.DataFrame, int]:
    """Returns (updated predictions, number of games matched)."""
    if preds is None or preds.empty or lines is None or lines.empty:
        return preds, 0
    df = preds.copy()
    if cfg.key == "nfl":
        to_key = lambda name: NFL_TEAMS.get(name)
    elif cfg.key == "nba":
        to_key = lambda name: NBA_TEAMS.get(name)
    else:
        cands = dict(zip(df["home"], df["home_name"]))
        cands.update(dict(zip(df["away"], df["away_name"])))
        to_key = lambda name: match_team(name, cands)
    idx = {}
    for _, ln in lines.iterrows():
        h, a = to_key(ln["home_name"]), to_key(ln["away_name"])
        if h and a:
            idx[(h, a)] = ln
    if "market_home_win_ml" not in df.columns:
        df["market_home_win_ml"] = np.nan
    df["line_source"] = df.get("line_source", "schedule")
    n = 0
    for i, r in df.iterrows():
        ln = idx.get((r["home"], r["away"]))
        if ln is None:
            continue
        n += 1
        if np.isfinite(ln["home_spread"]):
            df.at[i, "home_spread"] = ln["home_spread"]
        if np.isfinite(ln["total_line"]):
            df.at[i, "total_line"] = ln["total_line"]
        if np.isfinite(ln["market_home_win_ml"]):
            df.at[i, "market_home_win_ml"] = ln["market_home_win_ml"]
        df.at[i, "line_source"] = f"odds api ({int(ln['n_books'])} books)"
    return apply_market(df, cfg), n
