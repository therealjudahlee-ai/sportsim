"""Cached data access for the app.

Big contexts use cache_resource (one shared copy, no pickling on every rerun)
WITH a TTL, so data actually refreshes. Nothing in the app mutates a context;
treat them as read-only. Derived results are keyed by the context's build
timestamp so they refresh whenever the context does.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from engine.common import Diagnostics, now_et
from engine.config import get_config
from engine.market import merge_game_lines
from engine.odds_api import NBA_PROP_MARKETS, NFL_PROP_MARKETS, OddsClient, get_key

HOUR = 3600


def _stamp(ctx: dict) -> dict:
    ctx["built_at"] = now_et().isoformat(timespec="seconds")
    return ctx


@st.cache_resource(ttl=2 * HOUR, show_spinner="Loading NFL stats, schedule, rosters and injuries…")
def nfl_ctx() -> dict:
    from data import nfl
    return _stamp(nfl.build_context(Diagnostics()))


@st.cache_resource(ttl=HOUR, show_spinner="Loading NBA game logs, slate and injuries…")
def nba_ctx(date_str: str) -> dict:
    from data import nba
    return _stamp(nba.build_context(date_str, Diagnostics()))


@st.cache_resource(ttl=2 * HOUR, show_spinner="Loading college basketball box scores and slate…")
def ncaab_ctx(date_str: str) -> dict:
    from data import college
    return _stamp(college.build_ncaab_context(date_str, Diagnostics()))


@st.cache_resource(ttl=3 * HOUR, show_spinner="Loading college football results and schedule…")
def ncaaf_ctx() -> dict:
    from data import college
    return _stamp(college.build_ncaaf_context(Diagnostics()))


@st.cache_resource(ttl=2 * HOUR, show_spinner="Projecting players…")
def projections(sport: str, built_at: str, stats: tuple, date_str: str = "") -> list[dict]:
    if sport == "nfl":
        from data import nfl
        return nfl.all_projections(nfl_ctx(), list(stats))
    from data import nba
    return nba.all_projections(nba_ctx(date_str), list(stats))


# ------------------------------------------------------------------ odds
def odds_key() -> str:
    try:
        return get_key(st.secrets)
    except Exception:  # no secrets.toml at all
        return get_key(None)


@st.cache_data(ttl=30 * 60, show_spinner="Fetching sportsbook lines…")
def game_lines(sport: str) -> tuple[pd.DataFrame, str | None, list]:
    diag = Diagnostics()
    client = OddsClient(odds_key(), diag)
    df = client.game_lines(get_config(sport).odds_api_key)
    return df, client.remaining, diag.warnings


def with_market(preds: pd.DataFrame, sport: str) -> tuple[pd.DataFrame, str]:
    """Overlay live consensus lines when an Odds API key is configured."""
    if preds is None or preds.empty or not odds_key():
        return preds, "schedule feed (ESPN/nflverse)"
    lines, remaining, _ = game_lines(sport)
    merged, n = merge_game_lines(preds, lines, get_config(sport))
    src = f"The Odds API consensus ({n}/{len(preds)} games matched"
    src += f", {remaining} credits left)" if remaining else ")"
    return merged, src


@st.cache_data(ttl=30 * 60, show_spinner="Fetching player prop lines (uses API credits)…")
def prop_lines(sport: str, matchups: tuple, stats: tuple) -> tuple[pd.DataFrame, str | None, list]:
    """matchups: tuple of (our_home_key, our_away_key). Fetches props for matching events only."""
    from engine.odds_api import NBA_TEAMS, NFL_TEAMS
    diag = Diagnostics()
    client = OddsClient(odds_key(), diag)
    cfg = get_config(sport)
    names = NFL_TEAMS if sport == "nfl" else NBA_TEAMS
    events = client._get(f"/sports/{cfg.odds_api_key}/events") or []
    want = set(matchups)
    mm = NFL_PROP_MARKETS if sport == "nfl" else NBA_PROP_MARKETS
    frames = []
    for ev in events:
        key = (names.get(ev.get("home_team")), names.get(ev.get("away_team")))
        if key in want:
            df = client.player_props(cfg.odds_api_key, ev["id"], list(stats), mm)
            if not df.empty:
                frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, client.remaining, diag.warnings


def clear(fn) -> None:
    fn.clear()


@st.cache_data(ttl=30 * 60, show_spinner="Fetching alternate lines (uses API credits)…")
def alt_lines(sport: str, matchups: tuple, stats: tuple) -> tuple[pd.DataFrame, str | None, list]:
    from engine.odds_api import NBA_ALT_MARKETS, NBA_TEAMS, NFL_ALT_MARKETS, NFL_TEAMS, alt_ladder
    diag = Diagnostics()
    client = OddsClient(odds_key(), diag)
    cfg = get_config(sport)
    names = NFL_TEAMS if sport == "nfl" else NBA_TEAMS
    alt_map = NFL_ALT_MARKETS if sport == "nfl" else NBA_ALT_MARKETS
    events = client._get(f"/sports/{cfg.odds_api_key}/events") or []
    frames = []
    for ev in events:
        if (names.get(ev.get("home_team")), names.get(ev.get("away_team"))) in set(matchups):
            df = alt_ladder(client, cfg.odds_api_key, ev["id"], list(stats), alt_map)
            if not df.empty:
                frames.append(df)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), client.remaining, diag.warnings
