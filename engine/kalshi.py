"""Kalshi prediction-market integration (no API key needed for prices).

Kalshi contract prices ARE probabilities: a YES at 62¢ pays $1 if it happens.
Each game has an "event" shared across series, e.g.
    KXNBAGAME-26NOV27LALSAS     winner markets  ("San Antonio wins")
    KXNBASPREAD-26NOV27LALSAS   spread ladder   ("San Antonio wins by over 8.5 points", floor_strike 8.5)
    KXNBATOTAL-26NOV27LALSAS    total ladder    ("Over 252.5 points", floor_strike 252.5)
    KXNBAPTS-26NOV27LALSAS      prop ladders    ("Victor Wembanyama: 25+", floor_strike 24.5)
so everything for one game shares the suffix "26NOV27LALSAS" (date + team codes).

Fees: taker = 0.07 x P x (1-P) per contract (max 1.75¢ at 50¢), maker = 0.0175 x P x (1-P).
Kalshi rounds the fee up to the cent per ORDER, so tiny orders pay slightly more.
"""
from __future__ import annotations

import base64
import re
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from .common import Diagnostics, http, norm_name, norm_team

BASE = "https://api.elections.kalshi.com/trade-api/v2"
API_PATH_PREFIX = "/trade-api/v2"

# Known series (confirmed against the live API). discover_series() adds any others.
SERIES = {
    "nba": {"game": ["KXNBAGAME"], "spread": ["KXNBASPREAD"], "total": ["KXNBATOTAL"],
            "props": {"KXNBAPTS": "PTS", "KXNBAREB": "REB", "KXNBAAST": "AST", "KXNBA3PT": "FG3M"}},
    "nfl": {"game": ["KXNFLGAME"], "spread": ["KXNFLSPREAD"], "total": ["KXNFLTOTAL"],
            "props": {"KXNFLPASSYDS": "passing_yards", "KXNFLRSHYDS": "rushing_yards",
                      "KXNFLRECYDS": "receiving_yards", "KXNFLREC": "receptions"}},
    "ncaab": {"game": ["KXNCAAMBGAME"], "spread": ["KXNCAAMBSPREAD"], "total": ["KXNCAAMBTOTAL"], "props": {}},
    "ncaaf": {"game": ["KXNCAAFGAME"], "spread": ["KXNCAAFSPREAD"], "total": ["KXNCAAFTOTAL"], "props": {}},
}
PREFIX = {"nba": "KXNBA", "nfl": "KXNFL", "ncaab": "KXNCAAMB", "ncaaf": "KXNCAAF"}
# title keywords -> our stat names, for discovering prop series we haven't hard-coded
PROP_KEYWORDS = {
    "nba": [("three", "FG3M"), ("3-point", "FG3M"), ("rebound", "REB"), ("assist", "AST"), ("points", "PTS")],
    "nfl": [("passing yard", "passing_yards"), ("rushing yard", "rushing_yards"),
            ("receiving yard", "receiving_yards"), ("reception", "receptions"),
            ("touchdown", "anytime_td"), ("passing td", "passing_tds")],
}
EXCLUDE_WORDS = ("leader", "career", "season", "mvp", "award", "draft", "quarter", "half", "range", "precise",
                 "combo", "parlay", "first", "series", "playoff", "champion", "week", "total wins")

# Kalshi team codes that differ from ours
CODE_ALIAS = {
    "nfl": {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "LV": "LV", "LVR": "LV", "NOS": "NO", "KCC": "KC",
            "GBP": "GB", "TBB": "TB", "SFO": "SF", "NWE": "NE"},
    "nba": {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "WSH": "WAS", "PHO": "PHX", "BRK": "BKN",
            "UTAH": "UTA", "CHO": "CHA"},
}

_EVENT_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})([A-Z0-9]+)$")
_MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
                                         "NOV", "DEC"], start=1)}


# =============================================================================
# fees & value
# =============================================================================
def taker_fee(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return 0.07 * p * (1 - p)


def maker_fee(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return 0.0175 * p * (1 - p)


def order_fee(price: float, contracts: int, maker: bool = False) -> float:
    """Exact Kalshi fee for an order: rounded UP to the next cent."""
    rate = 0.0175 if maker else 0.07
    return float(np.ceil(rate * contracts * price * (1 - price) * 100 - 1e-9) / 100)


# =============================================================================
# public market data
# =============================================================================
def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def fetch_markets(series_ticker: str, diag: Diagnostics | None = None, status: str = "open",
                  max_pages: int = 6) -> list[dict]:
    diag = diag or Diagnostics()
    out, cursor = [], None
    for _ in range(max_pages):
        params = {"series_ticker": series_ticker, "status": status, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        try:
            r = http().get(f"{BASE}/markets", params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(1.5)
                continue
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            diag.warn(f"Kalshi markets request failed ({series_ticker})", e)
            break
        out.extend(data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor:
            break
    return out


def fetch_market(ticker: str, diag: Diagnostics | None = None) -> dict | None:
    try:
        r = http().get(f"{BASE}/markets/{ticker}", timeout=12)
        r.raise_for_status()
        return r.json().get("market")
    except Exception as e:
        if diag is not None:
            diag.warn(f"Kalshi market lookup failed ({ticker})", e)
        return None


def discover_series(sport: str, diag: Diagnostics | None = None) -> dict:
    """Scan Kalshi's sports series list for per-game spread/total/prop series we don't know about."""
    known = {k: (list(v) if isinstance(v, list) else dict(v)) for k, v in SERIES[sport].items()}
    try:
        r = http().get(f"{BASE}/series", params={"category": "Sports"}, timeout=30)
        r.raise_for_status()
        series = r.json().get("series", [])
    except Exception as e:
        if diag is not None:
            diag.warn("Kalshi series discovery failed; using the built-in list", e)
        return known
    pre = PREFIX[sport]
    for s in series:
        t, title = str(s.get("ticker", "")), str(s.get("title", "")).lower()
        if not t.startswith(pre) or any(w in title for w in EXCLUDE_WORDS):
            continue
        # don't confuse NCAAMB with NCAAWB etc.
        if sport == "ncaab" and "women" in title:
            continue
        if t.endswith("SPREAD") and t not in known["spread"]:
            known["spread"].append(t)
        elif t.endswith("TOTAL") and "team" not in title and t not in known["total"]:
            known["total"].append(t)
        elif sport in PROP_KEYWORDS and t not in known["props"] and t not in known["game"] \
                and "team" not in title and "total" not in title and "spread" not in title:
            for kw, stat in PROP_KEYWORDS[sport]:
                if kw in title:
                    known["props"][t] = stat
                    break
    return known


def parse_event(event_ticker: str) -> tuple[pd.Timestamp | None, str]:
    """'KXNBAGAME-26NOV27LALSAS' -> (2026-11-27, 'LALSAS')."""
    m = _EVENT_RE.search(str(event_ticker))
    if not m:
        return None, ""
    yy, mon, dd, teams = m.groups()
    try:
        return pd.Timestamp(year=2000 + int(yy), month=_MONTHS[mon], day=int(dd)), teams
    except (KeyError, ValueError):
        return None, teams


def _market_rows(raw: list[dict], kind: str, stat: str | None = None) -> list[dict]:
    rows = []
    for m in raw:
        date, teams = parse_event(m.get("event_ticker", ""))
        suffix = str(m.get("ticker", "")).rsplit("-", 1)[-1]
        name = str(m.get("yes_sub_title") or m.get("title") or "")
        if kind == "prop":
            name = name.split(":")[0].strip()
        rows.append({
            "ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"), "kind": kind, "stat": stat,
            "date": date, "game_key": f"{date:%Y%m%d}-{teams}" if date is not None else teams, "teams": teams,
            "code": re.sub(r"\d+$", "", suffix) if kind in ("game", "spread") else "",
            "name": name, "title": m.get("title", ""), "strike": _f(m.get("floor_strike")),
            "yes_bid": _f(m.get("yes_bid_dollars")), "yes_ask": _f(m.get("yes_ask_dollars")),
            "last": _f(m.get("last_price_dollars")), "volume": _f(m.get("volume_fp")),
            "open_interest": _f(m.get("open_interest_fp")), "close_time": m.get("close_time"),
            "status": m.get("status"), "result": m.get("result", ""),
        })
    return rows


def load_markets(sport: str, diag: Diagnostics | None = None, discover: bool = True) -> pd.DataFrame:
    diag = diag or Diagnostics()
    series = discover_series(sport, diag) if discover else SERIES[sport]
    rows = []
    for kind in ("game", "spread", "total"):
        for s in series[kind]:
            rows += _market_rows(fetch_markets(s, diag), kind)
    for s, stat in series["props"].items():
        rows += _market_rows(fetch_markets(s, diag), "prop", stat)
    df = pd.DataFrame(rows)
    if not df.empty:
        diag.note(f"Kalshi {sport}: {len(df)} open contracts "
                  f"({', '.join(f'{k} {n}' for k, n in df['kind'].value_counts().items())})")
    return df


# =============================================================================
# matching Kalshi games to our predictions
# =============================================================================
def _name_score(kalshi_name: str, display: str) -> float | None:
    """How well 'Norfolk St.' matches 'Norfolk State Spartans' (lower = better, None = no match)."""
    k, d = norm_team(kalshi_name), norm_team(display)
    if not k or not d:
        return None
    if d == k:
        return 0.0
    if d.startswith(k + " "):
        return float(len(d.split()) - len(k.split()))   # leftover words (usually the mascot)
    return None


def match_games(markets: pd.DataFrame, preds: pd.DataFrame, sport: str) -> dict:
    """game_key -> {"row": prediction row, "codes": {kalshi code: 'home'/'away'}}."""
    games = markets[markets["kind"] == "game"]
    if games.empty or preds is None or preds.empty:
        return {}
    alias = CODE_ALIAS.get(sport, {})
    out = {}
    for gk, grp in games.groupby("game_key"):
        sides = grp[["code", "name"]].drop_duplicates("code").to_dict("records")
        if len(sides) != 2:
            continue
        best, best_score, best_codes = None, None, None
        for idx, p in preds.iterrows():
            codes, score = {}, 0.0
            for s in sides:
                if sport in ("nba", "nfl"):
                    c = alias.get(s["code"], s["code"])
                    side = "home" if c == p["home"] else ("away" if c == p["away"] else None)
                    sc = 0.0 if side else None
                else:
                    sh = _name_score(s["name"], p.get("home_name") or "")
                    sa = _name_score(s["name"], p.get("away_name") or "")
                    side, sc = (("home", sh) if sh is not None and (sa is None or sh <= sa)
                                else ("away", sa) if sa is not None else (None, None))
                if side is None or side in codes.values():
                    break
                codes[s["code"]] = side
                score += sc
            else:
                if best_score is None or score < best_score:
                    best, best_score, best_codes = idx, score, codes
        if best is not None:
            out[gk] = {"row": preds.loc[best], "codes": best_codes}
    return out


# =============================================================================
# model probability for each contract + value after fees
# =============================================================================
def score_markets(markets: pd.DataFrame, preds: pd.DataFrame, cfg, sport: str,
                  projections: list[dict] | None = None, max_spread: float = 0.25) -> pd.DataFrame:
    """One row per contract with the better side (YES or NO), model probability, and EV after fees."""
    if markets is None or markets.empty:
        return pd.DataFrame()
    gm = match_games(markets, preds, sport)
    proj_idx = {}
    for p in projections or []:
        proj_idx[(p["player_norm"], p["stat"])] = p
    rows = []
    for _, m in markets.iterrows():
        g = gm.get(m["game_key"])
        if g is None:
            continue
        r, codes = g["row"], g["codes"]
        home_name = r.get("home_name") or r["home"]
        away_name = r.get("away_name") or r["away"]
        game = f"{away_name} @ {home_name}"
        p_yes, detail = None, ""
        if m["kind"] == "game":
            side = codes.get(m["code"])
            if side:
                p_yes = r["home_win"] if side == "home" else r["away_win"]
                detail = f"model {r['model_away_pts']:.0f}–{r['model_home_pts']:.0f}"
        elif m["kind"] == "spread" and np.isfinite(m["strike"]):
            side = codes.get(m["code"])
            if side:
                team_margin = r["model_margin"] if side == "home" else -r["model_margin"]
                p_yes = float(norm.sf(m["strike"], team_margin, cfg.margin_sigma))
                detail = f"model margin {team_margin:+.1f}"
        elif m["kind"] == "total" and np.isfinite(m["strike"]):
            p_yes = float(norm.sf(m["strike"], r["model_total"], cfg.total_sigma))
            detail = f"model total {r['model_total']:.1f}"
        elif m["kind"] == "prop" and np.isfinite(m["strike"]):
            pr = proj_idx.get((norm_name(m["name"]), m["stat"]))
            if pr is not None and pr["team"] in (r["home"], r["away"]):
                p_yes = pr["dist"].prob_over(m["strike"])
                detail = f"proj {pr['median']:.1f}"
        if p_yes is None or not np.isfinite(p_yes):
            continue
        bid, ask = m["yes_bid"], m["yes_ask"]
        cands = []
        if np.isfinite(ask) and 0 < ask < 1:
            cands.append(("YES", ask, p_yes))
        if np.isfinite(bid) and 0 < bid < 1:
            cands.append(("NO", 1 - bid, 1 - p_yes))          # buying NO costs 1 - YES bid
        if not cands:
            continue
        evs = [(s, price, prob, prob - price - float(taker_fee(price))) for s, price, prob in cands]
        side, price, prob, ev = max(evs, key=lambda x: x[3])
        width = (ask - bid) if np.isfinite(ask) and np.isfinite(bid) else np.nan
        rows.append({
            "sport": sport, "kind": m["kind"], "stat": m["stat"], "game": game, "contract": m["title"],
            "side": side, "price": price, "model_prob": prob, "edge": prob - price,
            "ev_taker": ev, "ev_maker": prob - price - float(maker_fee(price)),
            "fee": float(taker_fee(price)), "spread_width": width, "volume": m["volume"],
            "detail": detail, "ticker": m["ticker"], "close_time": m["close_time"], "date": m["date"],
            "liquid": bool(np.isfinite(width) and width <= max_spread),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("ev_taker", ascending=False).reset_index(drop=True) if not out.empty else out


# =============================================================================
# optional: read YOUR settled positions (needs a Kalshi API key)
# =============================================================================
def normalize_pem(text: str) -> str:
    """Rebuild a PEM key pasted in any shape (one line, spaces instead of line breaks,
    END marker on the same line as the key, literal \\n) into the strict format crypto libraries need."""
    t = str(text).replace("\\n", "\n").strip().strip('"').strip("'")
    m = re.search(r"-----BEGIN ([A-Z ]+)-----(.*?)-----END \1-----", t, re.S)
    if not m:
        return t
    label, body = m.group(1), re.sub(r"\s+", "", m.group(2))
    lines = [body[i:i + 64] for i in range(0, len(body), 64)]
    return f"-----BEGIN {label}-----\n" + "\n".join(lines) + f"\n-----END {label}-----\n"


def _sign(private_key_pem: str, message: str) -> str:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

    key = serialization.load_pem_private_key(normalize_pem(private_key_pem).encode(), password=None)
    data = message.encode()
    if isinstance(key, ed25519.Ed25519PrivateKey):
        sig = key.sign(data)
    elif isinstance(key, rsa.RSAPrivateKey):
        sig = key.sign(data, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=hashes.SHA256().digest_size),
                       hashes.SHA256())
    else:
        raise ValueError("Unsupported Kalshi key type (expected Ed25519 or RSA)")
    return base64.b64encode(sig).decode()


def auth_headers(key_id: str, private_key_pem: str, method: str, path: str) -> dict:
    ts = str(int(time.time() * 1000))
    path = path.split("?")[0]
    return {"KALSHI-ACCESS-KEY": key_id, "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": _sign(private_key_pem, ts + method.upper() + path)}


def fetch_settlements(key_id: str, private_key_pem: str, diag: Diagnostics | None = None,
                      max_pages: int = 20) -> pd.DataFrame:
    """Your settled positions: cost, fees, revenue per market."""
    diag = diag or Diagnostics()
    path = f"{API_PATH_PREFIX}/portfolio/settlements"
    rows, cursor = [], None
    for _ in range(max_pages):
        params = {"limit": 1000}
        if cursor:
            params["cursor"] = cursor
        try:
            r = http().get(f"{BASE}/portfolio/settlements", params=params, timeout=15,
                           headers=auth_headers(key_id, private_key_pem, "GET", path))
            if r.status_code in (401, 403):
                diag.warn("Kalshi rejected the API key (check KALSHI_KEY_ID / KALSHI_PRIVATE_KEY)")
                break
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            diag.warn("Kalshi settlements request failed", e)
            break
        rows += data.get("settlements", [])
        cursor = data.get("cursor")
        if not cursor:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    num = lambda c: pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)
    out = pd.DataFrame({
        "ticker": df["ticker"], "event_ticker": df.get("event_ticker"), "result": df.get("market_result"),
        "settled": pd.to_datetime(df.get("settled_time"), utc=True, errors="coerce"),
        "yes_count": num("yes_count_fp"), "no_count": num("no_count_fp"),
        "cost": num("yes_total_cost_dollars") + num("no_total_cost_dollars"),
        "fees": num("fee_cost"), "revenue": num("revenue") / 100.0,
    })
    out["profit"] = out["revenue"] - out["cost"] - out["fees"]
    out["sport"] = out["ticker"].map(lambda t: next((s for s, p in PREFIX.items() if str(t).startswith(p)), "other"))
    return out.sort_values("settled", ascending=False).reset_index(drop=True)
