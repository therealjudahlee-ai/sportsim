"""Breaking injury news: insider posts on X + fresh ESPN injury reports.

X (Twitter) is optional. Its API is pay-per-use (about half a cent per post READ),
so we only pull posts from a short list of insiders, only the last few hours, and
the app caches results for 2 minutes. Add X_BEARER_TOKEN to secrets to turn it on.

Nothing here changes projections by itself — news is matched to players on the
slate and shown with an "Apply" button, because automatic parsing of tweets is
error-prone ("not ruled out" contains "ruled out").
"""
from __future__ import annotations

import re
from datetime import timedelta

import pandas as pd

from .common import Diagnostics, http, norm_name, now_et

# Edit these lists freely — handles only, no @.
INSIDERS = {
    "nfl": ["AdamSchefter", "RapSheet", "TomPelissero", "JFowlerESPN", "MikeGarafolo", "UnderdogNFL"],
    "nba": ["ShamsCharania", "UnderdogNBA", "ChrisBHaynes", "TheSteinLine", "TheNBACentral"],
}

X_SEARCH = "https://api.x.com/2/tweets/search/recent"

_NEG = re.compile(r"\bnot (?:been |yet )?ruled out\b|\bnot expected to miss\b|\bavoid(?:ed|s)? (?:a )?serious\b")
_ACTIVE = re.compile(r"\b(?:will play|is active|expected to play|plans to play|will return|returning|cleared to "
                     r"play|available (?:tonight|today|sunday|monday|thursday)|upgraded to available|good to go|"
                     r"will start|in the lineup)\b")
_OUT = re.compile(r"\b(?:ruled out|will not play|won'?t play|is out\b|out tonight|out for|inactive|not playing|"
                  r"placed on (?:ir|injured reserve)|season[- ]ending|will miss|to miss|is sidelined|"
                  r"doesn'?t play|downgraded to out|not travel)")
_DOUBT = re.compile(r"\bdoubtful\b")
_QUEST = re.compile(r"\b(?:questionable|game[- ]time decision|gtd|limited)\b")


def classify(text: str) -> str | None:
    t = text.lower()
    if _NEG.search(t):
        return "questionable"
    if _ACTIVE.search(t):
        return "active"
    if _OUT.search(t):
        return "out"
    if _DOUBT.search(t):
        return "doubtful"
    if _QUEST.search(t):
        return "questionable"
    return None


def fetch_x_posts(token: str, handles: list[str], hours: float = 6, max_results: int = 100,
                  diag: Diagnostics | None = None) -> pd.DataFrame:
    """Recent original posts from the given accounts. Billed per post returned."""
    diag = diag or Diagnostics()
    if not token or not handles:
        return pd.DataFrame()
    query = "(" + " OR ".join(f"from:{h}" for h in handles) + ") -is:retweet -is:reply"
    start = (now_et() - timedelta(hours=hours)).astimezone().isoformat(timespec="seconds")
    try:
        r = http().get(X_SEARCH, headers={"Authorization": f"Bearer {token}"}, timeout=12, params={
            "query": query, "max_results": max(10, min(100, max_results)), "start_time": start,
            "tweet.fields": "created_at,author_id"})
        if r.status_code in (401, 403):
            diag.warn(f"X API rejected the token ({r.status_code}) — check X_BEARER_TOKEN and that the app "
                      "has pay-per-use credits")
            return pd.DataFrame()
        if r.status_code == 402:
            diag.warn("X API: out of credits")
            return pd.DataFrame()
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as e:
        diag.warn("X API request failed", e)
        return pd.DataFrame()
    return pd.DataFrame([{"id": d["id"], "text": d.get("text", ""),
                          "time": pd.to_datetime(d.get("created_at"), utc=True),
                          "url": f"https://x.com/i/web/status/{d['id']}"} for d in data])


def match_news(posts: pd.DataFrame, players: pd.DataFrame, source: str = "X") -> pd.DataFrame:
    """Find slate players named in posts and classify the news."""
    cols = ["name_norm", "player", "team", "status", "source", "time", "text", "url"]
    if posts is None or posts.empty or players is None or players.empty:
        return pd.DataFrame(columns=cols)
    names = {r["name_norm"]: (r["name"], r["team"]) for _, r in players.iterrows() if len(r["name_norm"]) > 4}
    rows = []
    for _, p in posts.iterrows():
        text_norm = " " + norm_name(p["text"]) + " "
        hits = [k for k in names if f" {k} " in text_norm]
        if not hits:
            continue
        status = classify(p["text"])
        if status is None:
            continue
        for k in hits:
            rows.append({"name_norm": k, "player": names[k][0], "team": names[k][1], "status": status,
                         "source": source, "time": p.get("time"), "text": p["text"], "url": p.get("url", "")})
    out = pd.DataFrame(rows, columns=cols)
    # newest report per player wins
    return out.sort_values("time", ascending=False).drop_duplicates("name_norm").reset_index(drop=True)


def report_changes(fresh: pd.DataFrame, ctx_out: dict, players: pd.DataFrame) -> pd.DataFrame:
    """Players on the slate whose fresh ESPN status is out/doubtful but the loaded data didn't have them out."""
    from data.espn import status_weight
    cols = ["name_norm", "player", "team", "status", "source", "time", "text", "url"]
    if fresh is None or fresh.empty or players is None or players.empty:
        return pd.DataFrame(columns=cols)
    on_slate = {r["name_norm"]: (r["name"], r["team"]) for _, r in players.iterrows()}
    rows = []
    for _, r in fresh.iterrows():
        k = r["name_norm"]
        if k in on_slate and status_weight(r["status"]) >= 0.75 and k not in ctx_out:
            rows.append({"name_norm": k, "player": on_slate[k][0], "team": on_slate[k][1], "status": r["status"],
                         "source": "ESPN injury report", "time": pd.NaT, "text": r.get("comment", ""), "url": ""})
    return pd.DataFrame(rows, columns=cols)
