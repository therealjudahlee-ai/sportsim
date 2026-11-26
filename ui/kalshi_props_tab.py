"""Kalshi player props: the model's most likely props + a player-prop combo builder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from engine import combos, kalshi
from engine.config import get_config
from engine.edges import hit_rate

from . import components as ui
from . import loaders
from .pro_pages import LABELS


def _c(x):
    return "—" if x is None or pd.isna(x) else f"{x * 100:.0f}¢"


def _prop_label(r) -> str:
    """'Malik Monk: 20+ points' + side -> readable pick."""
    title = str(r["contract"])
    if r["pick"] == "YES":
        return f"{title} — YES ({_c(r['leg_price'])})"
    return f"{title} — NO, stays under ({_c(r['leg_price'])})"


def kalshi_props_tab(sport: str, ctx: dict, preds: pd.DataFrame):
    st.caption("Kalshi **player props** only, ranked by how likely the model thinks they are to hit. For each "
               "player and stat you get one line — the highest one that still clears your minimum chance, "
               "so it pays as much as possible. Build combos below and compare Kalshi's quote with the max "
               "quote shown.")
    mk, warns = loaders.kalshi_markets(sport)
    for w in warns:
        st.warning(w)
    if mk.empty or not (mk["kind"] == "prop").any():
        st.info("Kalshi has no player-prop markets open for this sport right now.")
        return
    stats = tuple(sorted(set(mk.loc[mk["kind"] == "prop", "stat"].dropna())))
    projs = loaders.projections_for(ctx, stats)
    proj_idx = {(p["player_norm"], p["stat"]): p for p in projs}
    edges = kalshi.score_markets(mk, preds, get_config(sport), sport, projs)
    props = edges[edges["kind"] == "prop"] if not edges.empty else edges
    if props.empty:
        st.info(f"Kalshi lists {int((mk['kind'] == 'prop').sum())} player-prop contracts, but none match players "
                "on this slate yet (props usually post the day of the game).")
        return
    games = {gk: g["row"] for gk, g in kalshi.match_games(mk, preds, sport).items()}

    # ---------------- filters
    c1, c2, c3, c4 = st.columns([2, 1.6, 1.2, 1.2])
    game_names = props.drop_duplicates("game_key").set_index("game")["game_key"].to_dict()
    gsel = c1.selectbox("Game", ["All games"] + sorted(game_names), key=f"{sport}_kp_game")
    stat_sel = c2.multiselect("Stats", list(stats), default=list(stats), format_func=lambda s: LABELS.get(s, s),
                              key=f"{sport}_kp_stats")
    min_prob = c3.slider("Min chance", 50, 95, 65, key=f"{sport}_kp_minp") / 100
    max_price = c4.slider("Max price (¢)", 50, 98, 90, key=f"{sport}_kp_maxp") / 100
    d1, d2 = st.columns(2)
    min_vol = d1.number_input("Min volume", 0, 100000, 0, step=10, key=f"{sport}_kp_vol",
                              help="Many prop rungs have little trading; raise this to stick to active markets.")
    sides = d2.radio("Sides", ["Overs and unders", "Overs only (YES)"], horizontal=True, key=f"{sport}_kp_sides")

    p = props[props["stat"].isin(stat_sel) & (props["volume"].fillna(0) >= min_vol)]
    if gsel != "All games":
        p = p[p["game_key"] == game_names[gsel]]
    opts = combos.leg_options(p)
    if opts.empty:
        st.info("No props match these filters.")
        return
    if sides.startswith("Overs only"):
        opts = opts[opts["pick"] == "YES"]
    board = combos.likely_sides(opts, min_prob, max_price)
    if board is None or board.empty:
        st.info("No props clear that minimum chance and max price — try lowering the minimum chance.")
        return

    rows = []
    for r in board.to_dict("records"):
        pr = proj_idx.get((r["player_norm"], r["stat"]))
        hist = pr.get("hist", []) if pr else []
        l10 = hit_rate(hist, r["strike"], 10)
        if r["pick"] == "NO" and pd.notna(l10):
            l10 = 1 - l10
        rows.append({"Pick": _prop_label(r), "Game": r["game"], "Stat": LABELS.get(r["stat"], r["stat"]),
                     "Model %": r["leg_prob"] * 100, "Price (¢)": r["leg_price"] * 100,
                     "Value (¢)": (r["leg_prob"] - r["leg_price"] - float(kalshi.taker_fee(r["leg_price"]))) * 100,
                     "Proj": pr["median"] if pr else np.nan, "Line": r["strike"],
                     "L10 hit %": l10 * 100 if pd.notna(l10) else np.nan, "Volume": r["volume"]})
    tbl = pd.DataFrame(rows)
    st.markdown(f"#### Most likely props ({len(tbl)})")
    st.dataframe(tbl, hide_index=True, use_container_width=True, column_config={
        "Model %": st.column_config.ProgressColumn(format="%.0f%%", min_value=50, max_value=100),
        "Price (¢)": st.column_config.NumberColumn(format="%.0f"),
        "Value (¢)": st.column_config.NumberColumn(format="%+.1f",
                                                    help="Expected profit per contract after the fee, if the model is right."),
        "Proj": st.column_config.NumberColumn(format="%.1f"), "Line": st.column_config.NumberColumn(format="%.1f"),
        "L10 hit %": st.column_config.NumberColumn(format="%.0f%%"),
        "Volume": st.column_config.NumberColumn(format="%.0f")})

    # ---------------- combo builder
    st.divider()
    st.markdown("#### Player-prop combo")
    by_label = {_prop_label(r): r for r in board.to_dict("records")}
    chosen = st.multiselect("Pick props (any games)", list(by_label), key=f"{sport}_kp_legs")
    legs = [by_label[l] for l in chosen]
    if len({combos.ladder_key(l) for l in legs}) < len(legs):
        st.warning("Two of those are lines on the same player and stat — one already covers the other.")
    elif len(legs) >= 2:
        try:
            sim = combos.simulate_multi(legs, games, get_config(sport), sport, proj_idx)
        except ValueError as e:
            st.error(str(e))
            sim = None
        if sim:
            _combo_metrics(sim)
    elif legs:
        st.caption("Add at least one more prop.")

    st.markdown("**Or let the app build prop combos**")
    s1, s2 = st.columns([1, 3])
    n_legs = s1.selectbox("Props per combo", [2, 3, 4, 5, 6], index=1, key=f"{sport}_kp_n")
    s2.write("")
    scope = "this game" if gsel != "All games" else "all games"
    if s2.button(f"Suggest prop combos ({scope})", key=f"{sport}_kp_go"):
        with st.spinner("Simulating combinations…"):
            sugg_opts = opts if gsel == "All games" else opts[opts["game_key"] == game_names[gsel]]
            st.session_state[f"{sport}_kp_sug"] = ((gsel, n_legs, min_prob, max_price), combos.suggest_props(
                sugg_opts, games, get_config(sport), sport, proj_idx, n_legs=n_legs, min_prob=min_prob,
                max_price=max_price))
    saved = st.session_state.get(f"{sport}_kp_sug")
    if saved and saved[0] == (gsel, n_legs, min_prob, max_price):
        if not saved[1]:
            st.info("Not enough props clear your filters for a combo that size — lower the minimum chance or "
                    "pick fewer props.")
        for i, sg in enumerate(saved[1], 1):
            with st.container(border=True):
                st.markdown(f"**Prop combo {i}**" + (f" · {sg['n_games']} games" if sg["n_games"] > 1 else ""))
                for l in sg["legs"]:
                    st.markdown(f"- {_prop_label(l)} · {l['game']}")
                _combo_metrics(sg, naive=sg["naive_price"])
        if saved[1]:
            st.caption("Ranked by the chance every prop hits. Props in the same game are simulated together "
                       "(teammates' stats move with their team's scoring). Build the same combo in Kalshi, request "
                       "a quote, and take it only at or below the max quote.")


def _combo_metrics(sim: dict, naive: float | None = None):
    a, b, c, d = st.columns(4)
    a.metric("Model hit chance", ui.pct(sim["joint"], 1),
             delta=f"{ui.pct(sim['independent'], 1)} if independent", delta_color="off")
    b.metric("Fair price", _c(sim["fair_price"]), delta=f"pays {1 / max(sim['joint'], 1e-6):.1f}x", delta_color="off")
    c.metric("Max quote to take", _c(sim["max_quote"]), delta=f"{_c(sim['max_quote_maker'])} as limit order",
             delta_color="off", help="Highest Kalshi quote with positive expected value after the fee.")
    if naive is not None:
        d.metric("Legs bought separately", _c(naive), help="Product of the leg prices — a rough reference only.")
    else:
        boost = sim["correlation_boost"]
        d.metric("Correlation effect", "—" if pd.isna(boost) else f"{boost:.2f}x")
