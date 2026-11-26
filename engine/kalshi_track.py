"""Track record for Kalshi edges — no account or key needed.

Every scheduled refresh logs the contracts the model flags (price at that moment),
then grades them once Kalshi settles the market. That answers the real question:
"if I'd bought what the app flagged, at the price it showed, would I be up after fees?"
Closing-line value (CLV) compares the logged price with the last traded price before
the market closed — positive CLV means the market moved toward the model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import Diagnostics
from .kalshi import fetch_market, taker_fee

ROOT = Path(__file__).resolve().parents[1]
EDGES_PATH = ROOT / "predictions" / "kalshi_edges.csv"
RESULTS_PATH = ROOT / "predictions" / "kalshi_results.csv"
COLS = ["logged_at", "sport", "kind", "stat", "game", "contract", "ticker", "side", "price", "model_prob",
        "ev_taker", "close_time"]


def log_edges(edges: pd.DataFrame, logged_at, min_ev: float = 0.02, liquid_only: bool = True,
              path: Path | None = None) -> int:
    path = path or EDGES_PATH
    if edges is None or edges.empty:
        return 0
    e = edges[edges["ev_taker"] >= min_ev]
    if liquid_only and "liquid" in e.columns:
        e = e[e["liquid"]]
    if e.empty:
        return 0
    e = e.copy()
    e["logged_at"] = pd.Timestamp(logged_at)
    for c in COLS:
        if c not in e.columns:
            e[c] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    e[COLS].to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(e)


def _load(path: Path, cols: list | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=cols or [])
    return pd.read_csv(path, dtype={"ticker": str})


def grade(now=None, diag: Diagnostics | None = None, edges_path: Path | None = None,
          results_path: Path | None = None, max_lookups: int = 400) -> int:
    """Look up settled results for logged contracts whose markets have closed."""
    edges_path, results_path = edges_path or EDGES_PATH, results_path or RESULTS_PATH
    log = _load(edges_path, COLS)
    if log.empty:
        return 0
    done = _load(results_path)
    done_t = set(done["ticker"]) if not done.empty else set()
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    close = pd.to_datetime(log["close_time"], utc=True, errors="coerce")
    # Kalshi game markets close a couple of days after tip; settle checks start once the game is likely over
    pending = log[~log["ticker"].isin(done_t)].assign(_c=close)
    pending = pending[pending["_c"].isna() | (pending["_c"] - pd.Timedelta(days=3) <= now)]
    rows = []
    for t in pending["ticker"].drop_duplicates().head(max_lookups):
        m = fetch_market(t, diag)
        if not m:
            continue
        res = str(m.get("result", "")).lower()
        if res in ("yes", "no"):
            last = pd.to_numeric(m.get("last_price_dollars"), errors="coerce")
            rows.append({"ticker": t, "result": res, "close_price": last})
    if rows:
        new = pd.DataFrame(rows)
        out = pd.concat([done, new], ignore_index=True).drop_duplicates("ticker", keep="last") if not done.empty else new
        results_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(results_path, index=False)
    return len(rows)


def graded(edges_path: Path | None = None, results_path: Path | None = None) -> pd.DataFrame:
    log = _load(edges_path or EDGES_PATH, COLS)
    res = _load(results_path or RESULTS_PATH)
    if log.empty or res.empty:
        return pd.DataFrame()
    log["logged_at"] = pd.to_datetime(log["logged_at"], utc=True, errors="coerce", format="mixed")
    first = log.sort_values("logged_at").drop_duplicates(["ticker", "side"], keep="first")   # what you'd have bought
    g = first.merge(res, on="ticker")
    if g.empty:
        return g
    g["won"] = ((g["side"] == "YES") & (g["result"] == "yes")) | ((g["side"] == "NO") & (g["result"] == "no"))
    g["fee"] = taker_fee(g["price"])
    g["profit"] = np.where(g["won"], 1.0 - g["price"], -g["price"]) - g["fee"]      # per contract, after fees
    side_close = np.where(g["side"] == "YES", g["close_price"], 1 - g["close_price"])
    g["clv"] = np.where(pd.notna(g["close_price"]) & (g["close_price"] > 0), side_close - g["price"], np.nan)
    return g


def summary(g: pd.DataFrame) -> dict:
    if g is None or g.empty:
        return {}
    staked = float((g["price"] + g["fee"]).sum())
    return {
        "contracts": int(len(g)), "win_rate": float(g["won"].mean()),
        "avg_price": float(g["price"].mean()), "avg_model_prob": float(g["model_prob"].mean()),
        "profit_per_contract": float(g["profit"].mean()), "total_profit": float(g["profit"].sum()),
        "roi": float(g["profit"].sum() / staked) if staked else np.nan,
        "avg_clv": float(g["clv"].mean()) if g["clv"].notna().any() else np.nan,
    }
