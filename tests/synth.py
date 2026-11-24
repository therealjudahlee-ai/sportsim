"""Synthetic data generators for tests (no network needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def synth_games(n_teams=30, seasons=(2024, 2025), games_per_team=40, poss=False, hfa=2.5,
                carry=0.6, sigma=12.0, seed=0, start_month=10):
    """Season-chained synthetic league with known true ratings."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    true = {}
    rows = []
    prev = np.zeros(n_teams)
    gid = 0
    for s in seasons:
        strength = carry * prev + rng.normal(0, 5.0, n_teams)
        off = strength / 2 + rng.normal(0, 1.0, n_teams)
        deff = -(strength / 2) + rng.normal(0, 1.0, n_teams)
        pace = rng.normal(0, 3.0, n_teams)
        true[s] = {"off": dict(zip(teams, off)), "def": dict(zip(teams, deff)), "pace": dict(zip(teams, pace))}
        prev = strength
        start = pd.Timestamp(year=s - 1 if start_month >= 8 else s, month=start_month, day=1)
        n_games = n_teams * games_per_team // 2
        for k in range(n_games):
            h, a = rng.choice(n_teams, 2, replace=False)
            date = start + pd.Timedelta(days=int(k * 150 / n_games))
            neutral = rng.random() < 0.05
            hc = 0 if neutral else 0.5
            base = 100.0 if poss else 24.0
            eh = base + off[h] + deff[a] + hfa * hc
            ea = base + off[a] + deff[h] - hfa * hc
            if poss:
                p = 68 + pace[h] + pace[a] + rng.normal(0, 3)
                hp = eh * p / 100 + rng.normal(0, sigma / np.sqrt(2))
                ap = ea * p / 100 + rng.normal(0, sigma / np.sqrt(2))
            else:
                p = np.nan
                hp = eh + rng.normal(0, sigma / np.sqrt(2))
                ap = ea + rng.normal(0, sigma / np.sqrt(2))
            rows.append({
                "game_id": f"g{gid}", "date": date, "season": s,
                "home": teams[h], "away": teams[a],
                "home_pts": round(hp), "away_pts": round(ap),
                "neutral": neutral, "poss": p,
                "home_spread": -round((eh - ea) * 2) / 2 + rng.normal(0, 1.5),
                "total_line": eh + ea,
            })
            gid += 1
    return pd.DataFrame(rows), true
