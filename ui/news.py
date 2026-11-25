"""Injury center: breaking-news alerts + manual overrides that re-run the projections."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data.espn import status_weight
from engine.news import INSIDERS, match_news, report_changes

from . import loaders


def _key(sport, what):
    return f"{sport}_{what}"


def _apply(sport: str, name: str, status: str):
    """Button callback: push a news item into the override widgets."""
    out_k, act_k = _key(sport, "mark_out"), _key(sport, "mark_active")
    outs, acts = list(st.session_state.get(out_k, [])), list(st.session_state.get(act_k, []))
    if status == "active":
        outs = [n for n in outs if n != name]
        if name not in acts:
            acts.append(name)
    else:
        acts = [n for n in acts if n != name]
        if name not in outs:
            outs.append(name)
    st.session_state[out_k], st.session_state[act_k] = outs, acts


def injury_center(sport: str, base: dict) -> dict:
    """Renders alerts + sidebar overrides; returns the context with overrides applied."""
    mod = __import__(f"data.{sport}", fromlist=["apply_overrides"])
    players = base["players"]
    if players.empty:
        return base
    by_name = {r["name"]: r["name_norm"] for _, r in players.iterrows()}
    injured = [r["name"] for _, r in players.iterrows() if r["status"]]

    # ---------------- news
    alerts = []
    fresh = loaders.fresh_injuries(sport)
    loaded_out = {r["name_norm"] for _, r in players.iterrows() if status_weight(r["status"]) >= 0.75}
    alerts.append(report_changes(fresh, loaded_out, players))
    x_on = bool(loaders.x_token())
    if x_on:
        posts, warns = loaders.x_posts(sport, tuple(INSIDERS[sport]), 6.0)
        for w in warns:
            st.sidebar.warning(w)
        alerts.append(match_news(posts, players, "X"))
    news = pd.concat([a for a in alerts if not a.empty], ignore_index=True) if any(
        not a.empty for a in alerts) else pd.DataFrame()

    outs = set(st.session_state.get(_key(sport, "mark_out"), []))
    acts = set(st.session_state.get(_key(sport, "mark_active"), []))
    if not news.empty:
        def handled(r):
            current = players.set_index("name_norm").loc[r["name_norm"], "status"]
            if r["status"] == "active":
                return r["player"] in acts or not current
            return r["player"] in outs or (status_weight(current) >= status_weight(r["status"]) > 0)
        news = news[~news.apply(handled, axis=1)]

    if not news.empty:
        with st.expander(f"🚨 {len(news)} injury update{'s' if len(news) != 1 else ''} not in the projections yet",
                         expanded=True):
            for i, r in news.iterrows():
                c1, c2 = st.columns([5, 1])
                when = "" if pd.isna(r["time"]) else pd.Timestamp(r["time"]).tz_convert("America/New_York").strftime(
                    " · %-I:%M %p ET")
                link = f" · [post]({r['url']})" if r["url"] else ""
                c1.markdown(f"**{r['player']}** ({r['team']}) → **{r['status'].upper()}** "
                            f"<span style='opacity:.6'>· {r['source']}{when}{link}</span>", unsafe_allow_html=True)
                if r["text"]:
                    c1.caption(str(r["text"])[:240])
                c2.button("Apply", key=_key(sport, f"apply_{i}_{r['name_norm']}"), on_click=_apply,
                          args=(sport, r["player"], r["status"]), use_container_width=True)
            st.caption("Applying marks the player in the sidebar overrides and re-runs every projection on the page. "
                       "Check the post first — automatic reading of news isn't perfect.")
    elif not x_on:
        st.caption("Injury news: ESPN report re-checked every 15 min. Add `X_BEARER_TOKEN` to secrets to also scan "
                   "insiders on X.")

    # ---------------- sidebar overrides
    with st.sidebar:
        st.divider()
        st.markdown("**Injury overrides**")
        names = sorted(by_name)
        st.session_state[_key(sport, "mark_out")] = [n for n in st.session_state.get(_key(sport, "mark_out"), [])
                                                      if n in by_name]
        st.session_state[_key(sport, "mark_active")] = [n for n in st.session_state.get(_key(sport, "mark_active"), [])
                                                         if n in by_name]
        mark_out = st.multiselect("Mark OUT", names, key=_key(sport, "mark_out"),
                                  help="Treat as out even if the injury feeds haven't caught up.")
        mark_act = st.multiselect("Mark ACTIVE (ignore listed injury)", sorted(injured),
                                  key=_key(sport, "mark_active"))
        x_note = "on" if x_on else "off (no X_BEARER_TOKEN)"
        st.caption(f"X insider scan: {x_note}")

    overrides = {by_name[n]: (n, "out") for n in mark_out}
    overrides.update({by_name[n]: (n, "active") for n in mark_act if n in by_name})
    if not overrides:
        return base
    st.info(f"Projections include your overrides: "
            + ", ".join(f"{n} {'OUT' if s == 'out' else 'ACTIVE'}" for n, s in overrides.values()), icon="✏️")
    return mod.apply_overrides(base, overrides)
