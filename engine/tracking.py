"""Prediction log + grading + accuracy metrics (the public track record).

Every refresh appends the current pre-game predictions to
predictions/predictions.csv. Grading compares the FIRST logged prediction for
each game (what you'd have bet) with the result, and compares the first vs
LAST logged market line to measure closing-line value (CLV) — whether the
market moved toward your side after you'd have bet it, which is the
standard sharp-bettor test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "predictions" / "predictions.csv"
RESULTS_PATH = ROOT / "predictions" / "results.csv"

LOG_COLS = ["logged_at", "sport", "game_id", "date", "home", "away", "home_name", "away_name",
            "model_margin", "model_total", "home_win", "home_spread", "total_line", "market_home_win",
            "cover_home", "over_prob"]


def log_predictions(preds: pd.DataFrame, sport: str, logged_at: pd.Timestamp, path: Path | None = None) -> int:
    path = path or LOG_PATH
    if preds is None or preds.empty:
        return 0
    df = preds.copy()
    if "state" in df.columns:
        df = df[df["state"].isin(["pre", ""]) | df["state"].isna()]
    df = df[df.get("known", True) == True]  # noqa: E712
    if df.empty:
        return 0
    df["sport"] = sport
    df["logged_at"] = pd.Timestamp(logged_at)
    for c in LOG_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df["home_name"] = df["home_name"].fillna(df["home"])
    df["away_name"] = df["away_name"].fillna(df["away"])
    df = df[LOG_COLS]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(df)


def save_results(results: pd.DataFrame, path: Path | None = None) -> None:
    """results: sport, game_id, home_pts, away_pts."""
    path = path or RESULTS_PATH
    if results is None or results.empty:
        return
    results = results[["sport", "game_id", "home_pts", "away_pts"]].dropna()
    if path.exists():
        old = pd.read_csv(path, dtype={"game_id": str})
        results = pd.concat([old, results]).drop_duplicates(["sport", "game_id"], keep="last")
    results.to_csv(path, index=False)


def load_log(path: Path | None = None) -> pd.DataFrame:
    path = path or LOG_PATH
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLS)
    df = pd.read_csv(path, dtype={"game_id": str})
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce", format="mixed")
    return df


def graded(log: pd.DataFrame | None = None, results: pd.DataFrame | None = None) -> pd.DataFrame:
    log = load_log() if log is None else log
    if results is None:
        results = pd.read_csv(RESULTS_PATH, dtype={"game_id": str}) if RESULTS_PATH.exists() else pd.DataFrame()
    if log.empty or results.empty:
        return pd.DataFrame()
    log = log.sort_values("logged_at")
    first = log.groupby(["sport", "game_id"]).first().reset_index()
    last = log.groupby(["sport", "game_id"])[["home_spread", "total_line", "market_home_win"]].last().reset_index()
    last = last.rename(columns={"home_spread": "close_spread", "total_line": "close_total",
                                "market_home_win": "close_home_win"})
    g = first.merge(last, on=["sport", "game_id"]).merge(results, on=["sport", "game_id"])
    if g.empty:
        return g
    margin = g["home_pts"] - g["away_pts"]
    g["home_won"] = (margin > 0).astype(float)
    g.loc[margin == 0, "home_won"] = 0.5
    p = g["home_win"].clip(1e-4, 1 - 1e-4)
    g["pick_correct"] = np.where(margin == 0, np.nan, ((p >= 0.5) == (margin > 0)).astype(float))
    g["brier"] = (p - g["home_won"]) ** 2
    g["logloss"] = -(g["home_won"] * np.log(p) + (1 - g["home_won"]) * np.log(1 - p))
    mp = g["market_home_win"].clip(1e-4, 1 - 1e-4)
    g["market_brier"] = np.where(g["market_home_win"].notna(), (mp - g["home_won"]) ** 2, np.nan)

    has_sp = g["home_spread"].notna()
    ats_margin = margin + g["home_spread"]
    pick_home = g["model_margin"] > -g["home_spread"]
    g["ats_edge"] = (g["model_margin"] - (-g["home_spread"])).abs()
    g["ats_result"] = np.where(~has_sp, np.nan, np.where(ats_margin == 0, 0.5,
                               ((ats_margin > 0) == pick_home).astype(float)))
    # CLV in points: did the line move toward our side between first and last log?
    move = (-g["close_spread"]) - (-g["home_spread"])   # change in market home margin
    g["clv_pts"] = np.where(has_sp & g["close_spread"].notna(), np.where(pick_home, move, -move), np.nan)

    has_t = g["total_line"].notna()
    tot = g["home_pts"] + g["away_pts"]
    pick_over = g["model_total"] > g["total_line"]
    g["total_result"] = np.where(~has_t, np.nan, np.where(tot == g["total_line"], 0.5,
                                 ((tot > g["total_line"]) == pick_over).astype(float)))
    return g


def summary(g: pd.DataFrame, min_edge: float = 0.0) -> dict:
    if g.empty:
        return {}
    ats = g[(g["ats_edge"] >= min_edge) & g["ats_result"].notna() & (g["ats_result"] != 0.5)]
    tots = g[g["total_result"].notna() & (g["total_result"] != 0.5)]
    out = {
        "games": int(len(g)),
        "su_accuracy": float(g["pick_correct"].mean()),
        "brier": float(g["brier"].mean()),
        "market_brier": float(g["market_brier"].mean()) if g["market_brier"].notna().any() else np.nan,
        "logloss": float(g["logloss"].mean()),
        "ats_record": f"{int(ats['ats_result'].sum())}-{int(len(ats) - ats['ats_result'].sum())}",
        "ats_pct": float(ats["ats_result"].mean()) if len(ats) else np.nan,
        "totals_pct": float(tots["total_result"].mean()) if len(tots) else np.nan,
        "avg_clv_pts": float(g["clv_pts"].mean()) if g["clv_pts"].notna().any() else np.nan,
    }
    return out


def calibration(g: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    if g.empty:
        return pd.DataFrame()
    # fold to the favorite's perspective so bins run 50% -> 100%
    p = np.where(g["home_win"] >= 0.5, g["home_win"], 1 - g["home_win"])
    won = np.where(g["home_win"] >= 0.5, g["home_won"], 1 - g["home_won"])
    edges = np.linspace(0.5, 1.0, bins // 2 + 1)
    b = pd.cut(p, edges, include_lowest=True)
    df = pd.DataFrame({"bin": b, "p": p, "won": won}).groupby("bin", observed=True).agg(
        predicted=("p", "mean"), actual=("won", "mean"), n=("won", "size")).reset_index()
    df["bin"] = df["bin"].astype(str)
    return df
