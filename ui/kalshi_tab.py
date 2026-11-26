"""Kalshi edge finder tab (works for every sport; props for NFL/NBA)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from engine import kalshi
from engine.config import get_config

from . import components as ui
from . import loaders

KIND_LABEL = {"game": "Winner", "spread": "Spread", "total": "Total", "prop": "Player prop"}
TIERS = ["Tier 1 · best value", "Tier 2", "Tier 3"]


def _cents(x) -> str:
    return "—" if pd.isna(x) else f"{x * 100:.0f}¢"


def kalshi_tab(sport: str, ctx: dict, preds: pd.DataFrame):
    st.caption("Live Kalshi prices compared with the model. A contract's price is the market's probability "
               "(62¢ = 62%). **EV** is your expected profit per $1 contract *after* Kalshi's fee — taker fee is "
               "0.07 × P × (1−P), up to 1.75¢ at 50¢. No account or API key needed.")
    mk, warns = loaders.kalshi_markets(sport)
    for w in warns:
        st.warning(w)
    if mk.empty:
        st.info("Kalshi has no open markets for this sport right now.")
        return
    projs = []
    if sport in ("nfl", "nba"):
        stats = tuple(sorted(set(mk.loc[mk["kind"] == "prop", "stat"].dropna())))
        if stats:
            projs = loaders.projections_for(ctx, stats)
    edges = kalshi.score_markets(mk, preds, get_config(sport), sport, projs)
    if edges.empty:
        st.info(f"Kalshi lists {len(mk)} {sport.upper()} contracts, but none match games on this slate yet.")
        return

    kinds = [k for k in ("game", "spread", "total", "prop") if k in set(edges["kind"])]
    c1, c2, c3, c4 = st.columns([2, 1.3, 1.3, 1])
    pick = c1.multiselect("Markets", kinds, default=kinds, format_func=KIND_LABEL.get, key=f"{sport}_k_kinds")
    min_ev = c2.slider("Min EV per contract (¢)", -5.0, 20.0, 2.0, 0.5, key=f"{sport}_k_ev") / 100
    max_w = c3.slider("Max bid/ask gap (¢)", 1, 50, 15, key=f"{sport}_k_gap",
                      help="Wide gaps mean thin markets — the price you see may not fill.") / 100
    n_cards = c4.number_input("Top plays", 1, 30, 8, key=f"{sport}_k_n")
    t1, t2 = st.columns(2)
    game = t1.selectbox("Game", ["All"] + sorted(edges["game"].unique()), key=f"{sport}_k_game")
    one_per = t2.toggle("Best contract per game/player", True, key=f"{sport}_k_one")

    e = edges[edges["kind"].isin(pick) & (edges["ev_taker"] >= min_ev)
              & (edges["spread_width"].fillna(1) <= max_w)]
    if game != "All":
        e = e[e["game"] == game]
    if one_per:
        e = e.assign(_k=e["game"] + "|" + e["kind"] + "|" + e["contract"].str.split(":").str[0]) \
             .drop_duplicates("_k").drop(columns="_k")
    st.caption(f"{len(e)} of {len(edges)} matched contracts shown · prices refresh every 90 s · "
               "your fill can differ from the displayed ask")
    if e.empty:
        st.info("Nothing clears your filters. Try a lower minimum EV or a wider bid/ask gap.")
    for i, (_, r) in enumerate(e.head(int(n_cards)).iterrows()):
        with st.container(border=True):
            st.caption(f"**{TIERS[i] if i < 3 else f'Tier {i + 1}'}** · {KIND_LABEL[r['kind']]} · {r['game']}")
            st.markdown(f"#### Buy **{r['side']}** · {r['contract']}")
            a, b, c, d = st.columns(4)
            a.metric("Price", _cents(r["price"]), delta=f"fee {r['fee'] * 100:.1f}¢", delta_color="off")
            b.metric("Model probability", ui.pct(r["model_prob"], 1), delta=f"{r['edge'] * 100:+.1f} pts vs price",
                     delta_color="off")
            c.metric("EV per contract", f"{r['ev_taker'] * 100:+.1f}¢",
                     delta=f"{r['ev_maker'] * 100:+.1f}¢ as limit order", delta_color="off",
                     help="Taker = buying at the ask right now. Limit (maker) orders pay a quarter of the fee "
                          "if someone fills them.")
            d.metric("Market", f"gap {_cents(r['spread_width'])}", delta=f"vol {r['volume']:,.0f}", delta_color="off")
            st.caption(f"{r['detail']} · `{r['ticker']}`")

    with st.expander("All matched contracts"):
        show = edges.copy()
        show["Market"] = show["kind"].map(KIND_LABEL)
        for col in ("price", "model_prob", "edge", "ev_taker", "ev_maker", "spread_width"):
            show[col] = show[col] * 100
        st.dataframe(show[["Market", "game", "contract", "side", "price", "model_prob", "edge", "ev_taker",
                           "ev_maker", "spread_width", "volume", "detail", "ticker"]],
                     hide_index=True, use_container_width=True, column_config={
                         "price": st.column_config.NumberColumn("Price (¢)", format="%.0f"),
                         "model_prob": st.column_config.NumberColumn("Model %", format="%.1f"),
                         "edge": st.column_config.NumberColumn("Edge (pts)", format="%+.1f"),
                         "ev_taker": st.column_config.NumberColumn("EV taker (¢)", format="%+.1f"),
                         "ev_maker": st.column_config.NumberColumn("EV limit (¢)", format="%+.1f"),
                         "spread_width": st.column_config.NumberColumn("Gap (¢)", format="%.0f"),
                         "volume": st.column_config.NumberColumn("Volume", format="%.0f")})
        st.download_button("Download CSV", edges.to_csv(index=False).encode(), f"kalshi_{sport}.csv", "text/csv",
                           key=f"{sport}_k_dl")
