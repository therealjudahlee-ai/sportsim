"""Real sportsbook lines from The Odds API (https://the-odds-api.com).

Free tier = 500 credits/month. Game lines cost ~3 credits per call (one per
market); player props cost (markets x regions) per EVENT, so props scans are
the expensive part — results are cached by the app for 30 minutes.

Set the key in .streamlit/secrets.toml as ODDS_API_KEY = "..." (or env var).
Without a key, the app still runs: games use ESPN/nflverse lines, and the
prop tools fall back to projections only (no fake lines).
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd

from .common import Diagnostics, devig_pair, http, norm_name, norm_team

BASE = "https://api.the-odds-api.com/v4"

NFL_PROP_MARKETS = {
    "passing_yards": "player_pass_yds",
    "rushing_yards": "player_rush_yds",
    "receiving_yards": "player_reception_yds",
    "receptions": "player_receptions",
    "anytime_td": "player_anytime_td",
    "passing_tds": "player_pass_tds",
}
NBA_PROP_MARKETS = {
    "PTS": "player_points", "REB": "player_rebounds", "AST": "player_assists",
    "FG3M": "player_threes", "PRA": "player_points_rebounds_assists",
    "PR": "player_points_rebounds", "PA": "player_points_assists",
    "RA": "player_rebounds_assists", "BLK": "player_blocks", "STL": "player_steals",
}

NFL_TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
NBA_TEAMS = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


def get_key(secrets=None) -> str:
    if secrets is not None:
        try:
            k = secrets.get("ODDS_API_KEY", "")
            if k:
                return str(k)
        except Exception:
            pass
    return os.environ.get("ODDS_API_KEY", "")


class OddsClient:
    def __init__(self, key: str, diag: Diagnostics | None = None, regions: str = "us"):
        self.key = key
        self.diag = diag or Diagnostics()
        self.regions = regions
        self.remaining: str | None = None

    def _get(self, path: str, **params):
        params = {"apiKey": self.key, **params}
        try:
            r = http().get(f"{BASE}{path}", params=params, timeout=15)
            self.remaining = r.headers.get("x-requests-remaining", self.remaining)
            if r.status_code == 401:
                self.diag.warn("Odds API: invalid key")
                return None
            if r.status_code == 422:
                return None  # market not offered for this event — normal
            r.raise_for_status()
            return r.json()
        except Exception as e:
            self.diag.warn(f"Odds API request failed ({path})", e)
            return None

    # ------------------------------------------------------------------ games
    def game_lines(self, sport_key: str) -> pd.DataFrame:
        """Consensus (median across books) moneyline, spread and total per event."""
        data = self._get(f"/sports/{sport_key}/odds", regions=self.regions,
                         markets="h2h,spreads,totals", oddsFormat="american") or []
        rows = []
        for ev in data:
            home, away = ev.get("home_team"), ev.get("away_team")
            ml_h, ml_a, sp_h, tot = [], [], [], []
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    oc = {o.get("name"): o for o in mk.get("outcomes", [])}
                    if mk.get("key") == "h2h" and home in oc and away in oc:
                        ml_h.append(oc[home]["price"]); ml_a.append(oc[away]["price"])
                    elif mk.get("key") == "spreads" and home in oc:
                        sp_h.append(oc[home].get("point"))
                    elif mk.get("key") == "totals" and "Over" in oc:
                        tot.append(oc["Over"].get("point"))
            # de-vig each book's moneyline then take the median probability
            hw = [devig_pair(h, a)[0] for h, a in zip(ml_h, ml_a)]
            rows.append({
                "event_id": ev.get("id"), "commence_time": pd.to_datetime(ev.get("commence_time"), utc=True),
                "home_name": home, "away_name": away,
                "home_spread": _median(sp_h), "total_line": _median(tot),
                "market_home_win_ml": _median(hw), "n_books": len(ev.get("bookmakers", [])),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ props
    def player_props(self, sport_key: str, event_id: str, stats: list[str], market_map: dict) -> pd.DataFrame:
        markets = ",".join(market_map[s] for s in stats if s in market_map)
        data = self._get(f"/sports/{sport_key}/events/{event_id}/odds", regions=self.regions,
                         markets=markets, oddsFormat="american")
        if not data:
            return pd.DataFrame()
        rev = {v: k for k, v in market_map.items()}
        # (player, stat, point) -> {"over": [(price, book)], "under": [...]}
        book = defaultdict(lambda: {"over": [], "under": []})
        for bk in data.get("bookmakers", []):
            for mk in bk.get("markets", []):
                stat = rev.get(mk.get("key"))
                if not stat:
                    continue
                for o in mk.get("outcomes", []):
                    side = {"over": "over", "yes": "over", "under": "under", "no": "under"}.get(str(o.get("name")).lower())
                    player = o.get("description")
                    if not side or not player:
                        continue
                    point = o.get("point", 0.5 if stat == "anytime_td" else None)
                    if point is None:
                        continue
                    book[(player, stat, float(point))][side].append((o["price"], bk.get("title", bk.get("key"))))
        return _consolidate_props(book, event_id)


def _consolidate_props(book: dict, event_id: str) -> pd.DataFrame:
    """Pick the most-offered line per player/stat; best price + de-vigged fair prob."""
    by_ps = defaultdict(list)
    for (player, stat, point), sides in book.items():
        by_ps[(player, stat)].append((point, sides))
    rows = []
    for (player, stat), opts in by_ps.items():
        point, sides = max(opts, key=lambda x: len(x[1]["over"]) + len(x[1]["under"]))
        overs, unders = sides["over"], sides["under"]
        fair_over = []
        books_over = {b: p for p, b in overs}
        for p_u, b in unders:
            if b in books_over:
                fair_over.append(devig_pair(books_over[b], p_u)[0])
        best_over = max(overs, key=lambda x: x[0]) if overs else (np.nan, "")
        best_under = max(unders, key=lambda x: x[0]) if unders else (np.nan, "")
        if fair_over:
            mkt = float(np.median(fair_over))
            devigged = True
        elif overs:  # one-sided market (e.g. anytime TD): strip a typical ~4.5% hold
            mkt = float(np.median([1 / (1 + (p / 100 if p > 0 else 100 / -p)) for p, _ in overs])) / 1.045
            devigged = False
        else:
            mkt = np.nan
            devigged = False
        rows.append({
            "event_id": event_id, "player": player, "player_norm": norm_name(player), "stat": stat,
            "line": point, "best_over": best_over[0], "best_over_book": best_over[1],
            "best_under": best_under[0], "best_under_book": best_under[1],
            "market_over_prob": mkt, "devigged": devigged, "n_books": max(len(overs), len(unders)),
        })
    return pd.DataFrame(rows)


def _median(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else np.nan


def match_team(name: str, candidates: dict[str, str]) -> str | None:
    """Map an Odds API team name onto one of our team keys.
    candidates: {our_key: display_name}."""
    import difflib
    target = norm_team(name)
    normed = {norm_team(v): k for k, v in candidates.items()}
    if target in normed:
        return normed[target]
    hit = difflib.get_close_matches(target, list(normed), n=1, cutoff=0.85)
    return normed[hit[0]] if hit else None


# ----------------------------------------------------------------------------
# Alternate lines (the 2x-3x ladders)
# ----------------------------------------------------------------------------
NFL_ALT_MARKETS = {
    "passing_yards": "player_pass_yds_alternate",
    "rushing_yards": "player_rush_yds_alternate",
    "receiving_yards": "player_reception_yds_alternate",
    "receptions": "player_receptions_alternate",
    "passing_tds": "player_pass_tds_alternate",
}
NBA_ALT_MARKETS = {
    "PTS": "player_points_alternate", "REB": "player_rebounds_alternate", "AST": "player_assists_alternate",
    "FG3M": "player_threes_alternate", "PRA": "player_points_rebounds_assists_alternate",
    "PR": "player_points_rebounds_alternate", "PA": "player_points_assists_alternate",
    "RA": "player_rebounds_assists_alternate", "BLK": "player_blocks_alternate", "STL": "player_steals_alternate",
}


def alt_ladder(client: OddsClient, sport_key: str, event_id: str, stats: list[str], alt_map: dict) -> pd.DataFrame:
    """Every rung of every alternate ladder: best price per (player, stat, line, side) across books."""
    markets = ",".join(alt_map[s] for s in stats if s in alt_map)
    if not markets:
        return pd.DataFrame()
    data = client._get(f"/sports/{sport_key}/events/{event_id}/odds", regions=client.regions,
                       markets=markets, oddsFormat="american")
    if not data:
        return pd.DataFrame()
    rev = {v: k for k, v in alt_map.items()}
    best: dict = {}
    for bk in data.get("bookmakers", []):
        book = bk.get("title", bk.get("key"))
        for mk in bk.get("markets", []):
            stat = rev.get(mk.get("key"))
            if not stat:
                continue
            for o in mk.get("outcomes", []):
                side = {"over": "over", "under": "under"}.get(str(o.get("name")).lower())
                player, point, price = o.get("description"), o.get("point"), o.get("price")
                if not side or not player or point is None or price is None:
                    continue
                k = (player, stat, float(point), side)
                cur = best.get(k)
                n = (cur[2] + 1) if cur else 1
                if cur is None or price > cur[0]:
                    best[k] = (price, book, n)
                else:
                    best[k] = (cur[0], cur[1], n)
    rows = [{"event_id": event_id, "player": p, "player_norm": norm_name(p), "stat": s, "line": pt, "side": sd,
             "price": pr, "book": b, "n_books": n} for (p, s, pt, sd), (pr, b, n) in best.items()]
    return pd.DataFrame(rows)
