"""Backtest / tune the game model on past seasons.

    python -m jobs.backtest nba                 # score current settings
    python -m jobs.backtest nba --tune          # grid-search + save tuned params
    python -m jobs.backtest ncaab --tune --seasons 2025 2026

Tuned params land in engine/tuned_params.json and the app picks them up
automatically. Commit that file so Streamlit Cloud uses them too.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from engine.backtest import score, tune, walk_forward
from engine.common import Diagnostics, now_et
from engine.config import get_config, save_tuned


def load_history(sport: str, seasons: list[int], diag: Diagnostics) -> pd.DataFrame:
    """Loads eval seasons PLUS the season before (used only as the prior)."""
    all_seasons = sorted({seasons[0] - 1, *seasons})
    if sport == "nfl":
        from data.nfl import load_games
        return load_games(all_seasons, diag)
    if sport == "nba":
        from data.nba import games_from_logs, load_logs
        return games_from_logs(load_logs([s - 1 for s in all_seasons], diag))  # season = end year
    if sport == "ncaab":
        from data.college import load_ncaab_games
        return load_ncaab_games(all_seasons, diag)[0]
    if sport == "ncaaf":
        from data.college import load_ncaaf_games
        return load_ncaaf_games(all_seasons, diag)[0]
    raise ValueError(sport)


def default_seasons(sport: str) -> list[int]:
    today = now_et()
    y, m = today.year, today.month
    if sport == "nfl":
        cur = y if m >= 9 else y - 1
    elif sport == "ncaaf":
        cur = y if m >= 3 else y - 1
    else:  # basketball seasons named by end year
        cur = y + 1 if m >= 10 else y
    return [cur - 2, cur - 1]   # last two COMPLETED seasons


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sport", choices=["nfl", "nba", "ncaab", "ncaaf"])
    ap.add_argument("--seasons", type=int, nargs="+")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--refit-every", type=int, default=None,
                    help="days between refits (speeds up big sports; default 1 for NFL/NBA, 3 for college)")
    a = ap.parse_args(argv)
    seasons = a.seasons or default_seasons(a.sport)
    gap = a.refit_every if a.refit_every is not None else (3 if a.sport in ("ncaab", "ncaaf") else 1)
    diag = Diagnostics()
    games = load_history(a.sport, seasons, diag)
    cfg = get_config(a.sport)
    print(f"{a.sport}: {len(games)} games loaded; evaluating seasons {seasons}")

    if a.tune:
        best, table = tune(games, cfg, seasons, min_date_gap_days=gap)
        print("\nTop settings:\n", table.head(8).to_string(index=False))
        save_tuned(a.sport, best)
        print(f"\nSaved tuned params for {a.sport}: {json.dumps(best, indent=2)}")
        cfg = get_config(a.sport)

    bt = walk_forward(games, cfg, seasons, min_date_gap_days=gap)
    bt = bt[bt["known"]]
    print("\nBacktest results:")
    for k, v in score(bt, cfg.margin_sigma, cfg.total_sigma).items():
        print(f"  {k:32s} {v:.4f}" if isinstance(v, float) else f"  {k:32s} {v}")


if __name__ == "__main__":
    main()
