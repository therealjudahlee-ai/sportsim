"""Compare model projections to REAL market lines and build parlays from the edges."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .common import american_to_decimal, expected_value, fair_decimal, prob_to_american
from .props import parlay_prob


def score_props(projections: list[dict], market: pd.DataFrame, min_books: int = 1) -> pd.DataFrame:
    """projections: dicts with player_norm, player, team, opp, game, stat, position, dist.
    market: output of OddsClient.player_props (concatenated over events)."""
    if market is None or market.empty or not projections:
        return pd.DataFrame()
    proj_idx = {(p["player_norm"], p["stat"]): p for p in projections}
    rows = []
    for _, m in market.iterrows():
        p = proj_idx.get((m["player_norm"], m["stat"]))
        if p is None or m["n_books"] < min_books or not np.isfinite(m["market_over_prob"]):
            continue
        d = p["dist"]
        p_over, p_under = d.prob_over(m["line"]), d.prob_under(m["line"])
        mk_over = m["market_over_prob"]
        e_over, e_under = p_over - mk_over, p_under - (1 - mk_over)
        side = "over" if e_over >= e_under or not np.isfinite(m["best_under"]) else "under"
        if side == "under" and not np.isfinite(m["best_under"]):
            continue
        prob = p_over if side == "over" else p_under
        mprob = mk_over if side == "over" else 1 - mk_over
        price = m["best_over"] if side == "over" else m["best_under"]
        rows.append({
            "player": p["player"], "position": p.get("position", ""), "team": p["team"], "opp": p.get("opp", ""),
            "game": p["game"], "stat": m["stat"], "line": m["line"], "side": side,
            "proj_mean": d.mean, "proj_median": p.get("median", d.quantile(0.5)),
            "prob": prob, "market_prob": mprob, "edge": prob - mprob,
            "best_price": price, "book": m["best_over_book"] if side == "over" else m["best_under_book"],
            "ev": expected_value(prob, price) if np.isfinite(price) else np.nan,
            "fair_odds": prob_to_american(prob), "n_books": m["n_books"], "devigged": m["devigged"],
        })
    out = pd.DataFrame(rows)
    return out.sort_values("edge", ascending=False).reset_index(drop=True) if not out.empty else out


def build_parlays(legs: pd.DataFrame, sport: str, n_legs: int = 3, max_tickets: int = 5,
                  min_edge: float = 0.02, min_prob: float = 0.0, game: str | None = None,
                  pool_size: int = 16) -> list[dict]:
    """Search combos of positive-edge legs; rank by EV of the straight parlay at best prices.
    Correlations between legs in the same game are handled by parlay_prob (copula)."""
    if legs is None or legs.empty:
        return []
    pool = legs[(legs["edge"] >= min_edge) & (legs["prob"] >= min_prob)]
    if game:
        pool = pool[pool["game"] == game]
    pool = pool.sort_values("edge", ascending=False).drop_duplicates(subset=["player", "stat"])
    pool = pool.head(pool_size if n_legs <= 3 else min(pool_size, 13)).to_dict("records")
    if len(pool) < n_legs:
        return []

    scored = []
    for combo in itertools.combinations(pool, n_legs):
        if len({c["player"] for c in combo}) < n_legs:
            continue
        joint = parlay_prob(list(combo), sport, n=40_000)
        indep_market = float(np.prod([c["market_prob"] for c in combo]))
        if all(np.isfinite(c["best_price"]) for c in combo):
            dec = float(np.prod([american_to_decimal(c["best_price"]) for c in combo]))
            ev = joint * dec - 1.0
        else:
            dec, ev = np.nan, np.nan
        scored.append({
            "legs": combo, "model_prob": joint, "market_prob": indep_market,
            "edge_ratio": joint / indep_market if indep_market > 0 else np.nan,
            "parlay_decimal": dec, "ev": ev, "fair_payout": fair_decimal(joint),
            "same_game": len({c["game"] for c in combo}) == 1,
        })
    key = "ev" if any(np.isfinite(s["ev"]) for s in scored) else "edge_ratio"
    scored.sort(key=lambda s: (s[key] if np.isfinite(s[key]) else -9), reverse=True)

    picked, used = [], {}
    for s in scored:  # diversify: each player on at most 2 tickets
        if any(used.get(c["player"], 0) >= 2 for c in s["legs"]):
            continue
        for c in s["legs"]:
            used[c["player"]] = used.get(c["player"], 0) + 1
        picked.append(s)
        if len(picked) >= max_tickets:
            break
    return picked


# ----------------------------------------------------------------------------
# 2x-3x value finder
# ----------------------------------------------------------------------------
def hit_rate(hist, line: float, n: int) -> float:
    h = [x for x in (hist or []) if x is not None and np.isfinite(x)][-n:]
    return float(np.mean([x > line for x in h])) if len(h) >= min(n, 5) else np.nan


def value_finder(projections: list[dict], ladder: pd.DataFrame, min_payout: float = 2.0, max_payout: float = 3.0,
                 sides: tuple = ("over",)) -> pd.DataFrame:
    """Real alt-line rungs priced between min_payout and max_payout (decimal, i.e. 2x = +100, 3x = +200),
    scored by the model. EV is at the actual best price, so it already accounts for the book's cut."""
    if ladder is None or ladder.empty or not projections:
        return pd.DataFrame()
    proj_idx = {(p["player_norm"], p["stat"]): p for p in projections}
    lad = ladder.copy()
    lad["payout"] = lad["price"].map(american_to_decimal)
    lad = lad[lad["payout"].between(min_payout, max_payout) & lad["side"].isin(sides)]
    rows = []
    for _, r in lad.iterrows():
        p = proj_idx.get((r["player_norm"], r["stat"]))
        if p is None:
            continue
        d = p["dist"]
        prob = d.prob_over(r["line"]) if r["side"] == "over" else d.prob_under(r["line"])
        implied = 1.0 / r["payout"]
        rows.append({
            "player": p["player"], "position": p.get("position", ""), "team": p["team"], "opp": p.get("opp", ""),
            "game": p["game"], "stat": r["stat"], "side": r["side"], "line": r["line"],
            "price": r["price"], "payout": r["payout"], "book": r["book"], "n_books": r["n_books"],
            "proj_median": p["median"], "prob": prob, "implied": implied, "edge": prob - implied,
            "ev": prob * r["payout"] - 1.0, "fair_payout": fair_decimal(prob),
            "hit_l10": hit_rate(p.get("hist"), r["line"], 10), "hit_l20": hit_rate(p.get("hist"), r["line"], 20),
            "notes": "; ".join(p.get("notes", [])),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("ev", ascending=False).reset_index(drop=True) if not out.empty else out


def fair_ladder(projections: list[dict], payouts=(2.0, 2.5, 3.0)) -> pd.DataFrame:
    """No odds key: for each player, the highest half-point line the model says is worth
    taking at each payout (P(over) >= 1/payout). Compare these to what your app/book offers."""
    rows = []
    for p in projections:
        d = p["dist"]
        r = {"player": p["player"], "team": p["team"], "opp": p.get("opp", ""), "game": p["game"],
             "stat": p["stat"], "proj_median": p["median"]}
        for m in payouts:
            target = 1.0 / m
            q = d.quantile(1.0 - target)              # line where P(over) ~= 1/m
            line = np.floor(q) - 0.5 if np.floor(q) + 0.5 > q else np.floor(q) + 0.5
            while d.prob_over(line + 1.0) >= target:  # step up while still worth it
                line += 1.0
            while line > 0 and d.prob_over(line) < target:
                line -= 1.0
            r[f"line_{m:g}x"] = line
            r[f"prob_{m:g}x"] = d.prob_over(line)
            r[f"hit_l10_{m:g}x"] = hit_rate(p.get("hist"), line, 10)
        rows.append(r)
    return pd.DataFrame(rows)
