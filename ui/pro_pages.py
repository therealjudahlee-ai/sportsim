"""Props / parlays / DFS tabs shared by the NFL and NBA pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from engine import dfs
from engine.edges import build_parlays, score_props
from engine.props import NBA_PROP_STATS

from . import components as ui
from . import loaders

NFL_STATS_BY_POS = {
    "QB": ["passing_yards", "passing_tds", "rushing_yards", "anytime_td", "dk_points", "fd_points"],
    "RB": ["rushing_yards", "receiving_yards", "receptions", "anytime_td", "dk_points", "fd_points"],
    "WR": ["receiving_yards", "receptions", "anytime_td", "rushing_yards", "dk_points", "fd_points"],
    "TE": ["receiving_yards", "receptions", "anytime_td", "dk_points", "fd_points"],
}
NFL_MARKET_STATS = ["passing_yards", "rushing_yards", "receiving_yards", "receptions", "anytime_td", "passing_tds"]
NBA_MARKET_STATS = ["PTS", "REB", "AST", "FG3M", "PRA", "PR", "PA", "RA", "STL", "BLK"]
LABELS = {"passing_yards": "Pass yds", "rushing_yards": "Rush yds", "receiving_yards": "Rec yds",
          "receptions": "Receptions", "anytime_td": "Anytime TD", "passing_tds": "Pass TDs",
          "dk_points": "DK pts", "fd_points": "FD pts", "PTS": "Points", "REB": "Rebounds", "AST": "Assists",
          "FG3M": "3PM", "PRA": "Pts+Reb+Ast", "PR": "Pts+Reb", "PA": "Pts+Ast", "RA": "Reb+Ast",
          "STL": "Steals", "BLK": "Blocks", "DK_PTS": "DK pts", "FD_PTS": "FD pts"}


def _mod(sport):
    if sport == "nfl":
        from data import nfl as m
    else:
        from data import nba as m
    return m


def _player_label(r):
    pos = f" {r['position']}" if "position" in r and r["position"] else ""
    st_ = f" · {r['status'].upper()}" if r.get("status") else ""
    return f"{r['name']} · {r['team']}{pos}{st_}"


# =============================================================================
def props_tab(sport: str, ctx: dict, date_str: str = ""):
    mod = _mod(sport)
    players = ctx["players"]
    if players.empty:
        st.info("No active players found for this slate.")
        return

    st.markdown("#### Check a line")
    st.caption("Pick a player and type the line you're looking at. Probabilities come from the full projected "
               "distribution, not a fixed percentage of the average.")
    c1, c2, c3 = st.columns([2.2, 1.2, 1])
    labels = {_player_label(r): r["player_id"] for _, r in players.iterrows()}
    who = c1.selectbox("Player", list(labels), key=f"{sport}_prop_player")
    pid = labels[who]
    prow = players[players["player_id"] == pid].iloc[0]
    stats = NFL_STATS_BY_POS.get(prow.get("position"), NFL_MARKET_STATS) if sport == "nfl" else NBA_PROP_STATS
    stat = c2.selectbox("Stat", stats, format_func=lambda s: LABELS.get(s, s), key=f"{sport}_prop_stat")
    bump = c3.slider("Usage adjust", 0.80, 1.25, 1.00, 0.01, key=f"{sport}_prop_bump",
                     help="Manual multiplier — e.g. 1.08 if a teammate was just ruled out and the injury "
                          "feed hasn't caught it yet.")
    proj = mod.project_player(ctx, pid, stat, bump=bump)
    if proj is None:
        st.warning("Not enough recent games to project this stat for this player.")
    else:
        default = 0.5 if stat == "anytime_td" else float(np.floor(proj["median"]) + 0.5)
        line_val = st.number_input("Line", value=default, step=0.5, key=f"{sport}_line_{pid}_{stat}")
        ui.prop_detail(proj, line_val)

    st.divider()
    st.markdown("#### Props vs. the sportsbooks")
    if not loaders.odds_key():
        ui.no_odds_key_note()
        _projection_table(sport, ctx, date_str)
        return

    market_stats = NFL_MARKET_STATS if sport == "nfl" else NBA_MARKET_STATS
    c1, c2 = st.columns([3, 1])
    chosen = c1.multiselect("Markets", market_stats, default=market_stats[:4] if sport == "nfl" else market_stats[:3],
                            format_func=lambda s: LABELS.get(s, s), key=f"{sport}_mkts")
    n_games = len(ctx["predictions"])
    c2.write("")
    if c2.button(f"Load lines (~{max(1, n_games) * max(1, len(chosen))} credits)", key=f"{sport}_load_lines",
                 type="primary", use_container_width=True):
        matchups = tuple((r["home"], r["away"]) for _, r in ctx["predictions"].iterrows())
        market, remaining, warns = loaders.prop_lines(sport, matchups, tuple(chosen))
        for w in warns:
            st.warning(w)
        proj = loaders.projections(sport, ctx["built_at"], tuple(chosen), date_str)
        st.session_state[f"{sport}_edges"] = score_props(proj, market)
        st.session_state[f"{sport}_credits"] = remaining

    edges = st.session_state.get(f"{sport}_edges")
    if edges is None:
        st.caption("Press **Load lines** to pull real player-prop odds and compare them to the model.")
        return
    if edges.empty:
        st.warning("No prop lines matched our players (books may not have posted props yet).")
        return
    f1, f2, f3, f4 = st.columns(4)
    min_edge = f1.slider("Min edge (pct pts)", 0.0, 15.0, 3.0, 0.5, key=f"{sport}_minedge") / 100
    stat_f = f2.multiselect("Stats", sorted(edges["stat"].unique()), format_func=lambda s: LABELS.get(s, s),
                            key=f"{sport}_statf")
    game_f = f3.selectbox("Game", ["All"] + sorted(edges["game"].unique()), key=f"{sport}_gamef")
    plus_only = f4.toggle("Plus-money only", key=f"{sport}_plus",
                          help="Replaces the old '2x–3x value finder': real plus-money prices where the model "
                               "still sees an edge.")
    e = edges[edges["edge"] >= min_edge]
    if stat_f:
        e = e[e["stat"].isin(stat_f)]
    if game_f != "All":
        e = e[e["game"] == game_f]
    if plus_only:
        e = e[e["best_price"] >= 100]
    credits = st.session_state.get(f"{sport}_credits")
    st.caption(f"{len(e)} of {len(edges)} props shown"
               + (f" · {credits} API credits left" if credits else "")
               + " · edge = model probability − no-vig market probability")
    ui.edges_table(e, f"{sport}_edges_tbl")


def _projection_table(sport, ctx, date_str):
    stats = NFL_MARKET_STATS if sport == "nfl" else NBA_MARKET_STATS
    stat = st.selectbox("Projection board", stats, format_func=lambda s: LABELS.get(s, s), key=f"{sport}_board")
    projs = loaders.projections(sport, ctx["built_at"], (stat,), date_str)
    if not projs:
        st.info("No projections available.")
        return
    from engine.common import prob_to_american
    rows = []
    for p in projs:
        r = {"Player": p["player"], "Team": p["team"], "Opp": p["opp"], "Median": p["median"],
             "20th pct": p["p20"], "80th pct": p["p80"]}
        if stat == "anytime_td":
            pr = p["dist"].prob_over(0.5)
            r.update({"TD %": pr * 100, "Fair odds": prob_to_american(pr)})
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("TD %" if stat == "anytime_td" else "Median", ascending=False)
    st.dataframe(df, hide_index=True, use_container_width=True, column_config={
        c: st.column_config.NumberColumn(format="%.1f") for c in ("Median", "20th pct", "80th pct")} | {
        "TD %": st.column_config.NumberColumn(format="%.0f%%")})


# =============================================================================
def parlay_tab(sport: str):
    edges = st.session_state.get(f"{sport}_edges")
    st.caption("Parlays are built only from props with a real edge vs. sportsbook prices. Legs in the same game "
               "are simulated together (a QB and his receiver tend to hit together; two receivers on one "
               "team compete for targets).")
    if not loaders.odds_key():
        ui.no_odds_key_note()
        return
    if edges is None or edges.empty:
        st.info("Load prop lines in the **Player props** tab first.")
        return
    c1, c2, c3, c4 = st.columns(4)
    n_legs = c1.selectbox("Legs", [2, 3, 4, 5], index=1, key=f"{sport}_legs")
    n_tix = c2.slider("Tickets", 1, 10, 4, key=f"{sport}_tix")
    min_edge = c3.slider("Min edge per leg (pct pts)", 0.0, 10.0, 2.0, 0.5, key=f"{sport}_pedge") / 100
    game = c4.selectbox("Game", ["Any (cross-game)"] + sorted(edges["game"].unique()), key=f"{sport}_pgame")
    if st.button("Build parlays", type="primary", key=f"{sport}_build"):
        with st.spinner("Searching combinations…"):
            st.session_state[f"{sport}_tickets"] = build_parlays(
                edges, sport, n_legs=n_legs, max_tickets=n_tix, min_edge=min_edge,
                game=None if game.startswith("Any") else game)
    if f"{sport}_tickets" in st.session_state:
        ui.parlay_tickets(st.session_state[f"{sport}_tickets"])


# =============================================================================
@st.cache_data(ttl=2 * 3600, show_spinner="Building DFS player pool…")
def _pool(sport: str, built_at: str, site: str, salaries: pd.DataFrame | None, date_str: str) -> pd.DataFrame:
    if sport == "nfl":
        from data import nfl
        return nfl.dfs_pool(loaders.nfl_ctx(), site, salaries)
    from data import nba
    return nba.dfs_pool(loaders.nba_ctx(date_str), site, salaries)


def dfs_tab(sport: str, ctx: dict, date_str: str = ""):
    c1, c2, c3 = st.columns(3)
    site = c1.radio("Site", ["DK", "FD"], horizontal=True, key=f"{sport}_site",
                    format_func=lambda s: {"DK": "DraftKings", "FD": "FanDuel"}[s])
    objective = c2.selectbox("Optimize for", ["proj", "ceiling", "leverage"], key=f"{sport}_obj",
                             format_func=lambda s: {"proj": "Cash (median)", "ceiling": "GPP (85th pct ceiling)",
                                                    "leverage": "GPP leverage (ceiling, low-owned)"}[s])
    up = c3.file_uploader("Salary CSV from the contest page", type=["csv"], key=f"{sport}_csv")

    salaries = None
    if up is not None:
        try:
            salaries = dfs.parse_salary_csv(pd.read_csv(up))
            st.success(f"Loaded {len(salaries)} salaries from {up.name}")
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")
    elif site == "DK":
        if st.button("Try live DraftKings salaries", key=f"{sport}_livedk"):
            from data.dk import live_salaries
            from engine.common import Diagnostics
            d = Diagnostics()
            st.session_state[f"{sport}_live_sal"] = live_salaries(sport.upper(), d)
            for w in d.warnings:
                st.warning(w)
        salaries = st.session_state.get(f"{sport}_live_sal")

    if salaries is None or salaries.empty:
        if sport == "nba":
            st.info("Upload the DraftKings/FanDuel salary CSV (or load live DK salaries). NBA lineups need the "
                    "site's position eligibility, which only comes with real salaries.")
            return
        st.warning("Using **estimated** salaries. Upload your contest's CSV for exact prices and IDs you can "
                   "bulk-upload.")
    pool = _pool(sport, ctx["built_at"], site, salaries, date_str)
    if pool.empty:
        st.warning("No players in the pool — salary file may be for a different slate.")
        return

    names = pool["name"].tolist()
    l1, l2 = st.columns(2)
    locked = l1.multiselect("Lock", names, key=f"{sport}_lock")
    excluded = l2.multiselect("Exclude", [n for n in names if n not in locked], key=f"{sport}_excl")
    o1, o2, o3, o4 = st.columns(4)
    n_lu = o1.slider("Lineups", 1, 20, 3, key=f"{sport}_nlu")
    min_u = o2.slider("Min unique players", 1, 5, 2, key=f"{sport}_uniq")
    max_e = o3.slider("Max exposure %", 20, 100, 70, key=f"{sport}_exp") / 100
    kwargs = {}
    if sport == "nfl":
        stack = o4.selectbox("Stack", ["qb1", "qb2", "none"], key=f"{sport}_stack",
                             format_func=lambda s: {"qb1": "QB + 1 pass catcher", "qb2": "QB + 2 pass catchers",
                                                    "none": "No stack"}[s])
        t1, t2, t3 = st.columns(3)
        kwargs = dict(stack=stack, bringback=t1.toggle("Bring-back", True, key=f"{sport}_bb"),
                      no_opp_vs_dst=t2.toggle("No players vs. my DST", True, key=f"{sport}_nodst"),
                      max_1_rb_per_team=t3.toggle("Max 1 RB per team", True, key=f"{sport}_1rb"))
    else:
        kwargs = dict(max_per_team=o4.slider("Max per team", 2, 5, 4, key=f"{sport}_mpt"))

    if st.button("Generate lineups", type="primary", key=f"{sport}_gen"):
        with st.spinner("Solving…"):
            st.session_state[f"{sport}_lineups"] = (site, dfs.optimize(
                pool, sport, site, objective=objective, num_lineups=n_lu, min_unique=min_u, max_exposure=max_e,
                locked=locked, excluded=excluded, **kwargs))

    saved = st.session_state.get(f"{sport}_lineups")
    if saved and saved[0] == site:
        lus = saved[1]
        if not lus:
            st.error("No valid lineup with these locks/rules.")
        else:
            cap = dfs.SITES[(sport, site)]["cap"]
            for i, lu in enumerate(lus, 1):
                with st.container(border=True):
                    st.markdown(f"**Lineup {i}** · ${lu['salary'].sum():,} / ${cap:,} · median "
                                f"{lu['proj'].sum():.1f} · ceiling {lu['ceiling'].sum():.1f} · est. own "
                                f"{lu['own_est'].sum():.0f}%")
                    st.dataframe(lu[["slot", "name", "team", "opp", "salary", "proj", "ceiling", "own_est"]],
                                 hide_index=True, use_container_width=True, column_config={
                                     "proj": st.column_config.NumberColumn(format="%.1f"),
                                     "ceiling": st.column_config.NumberColumn(format="%.1f"),
                                     "own_est": st.column_config.NumberColumn("own % (rough)", format="%.0f")})
            expo = pd.concat(lus)["name"].value_counts().rename("lineups").to_frame()
            expo["exposure"] = expo["lineups"] / len(lus) * 100
            with st.expander("Exposure"):
                st.dataframe(expo, use_container_width=True,
                             column_config={"exposure": st.column_config.NumberColumn(format="%.0f%%")})
            if salaries is not None and not salaries.empty:
                st.download_button(f"Download {site} upload CSV", dfs.export_csv(lus, sport, site),
                                   f"{site}_{sport}_lineups.csv", "text/csv", key=f"{sport}_dfsdl")

    with st.expander("Player pool"):
        st.dataframe(pool.drop(columns=["id"]), hide_index=True, use_container_width=True)
