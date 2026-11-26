"""Kalshi edge finder tab (works for every sport; props for NFL/NBA)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from engine import combos, kalshi
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
    rank = st.radio("Show me", ["Most likely to hit", "Best value vs. Kalshi's price"], horizontal=True,
                    key=f"{sport}_k_rank",
                    help="Most likely: the legs the model is most confident in, whatever the price. "
                         "Best value: where Kalshi's price looks too low compared with the model.")
    likely = rank.startswith("Most likely")
    c1, c2, c3, c4 = st.columns([2, 1.3, 1.3, 1])
    pick = c1.multiselect("Markets", kinds, default=kinds, format_func=KIND_LABEL.get, key=f"{sport}_k_kinds")
    if likely:
        min_prob = c2.slider("Min model probability", 50, 95, 65, key=f"{sport}_k_minp") / 100
        max_price = c2.slider("Max price (¢)", 50, 99, 90, key=f"{sport}_k_maxp",
                              help="Very high prices pay almost nothing — a 95¢ contract wins 5¢.") / 100
        min_ev = -1.0
    else:
        min_ev = c2.slider("Min EV per contract (¢)", -5.0, 20.0, 2.0, 0.5, key=f"{sport}_k_ev") / 100
    max_w = c3.slider("Max bid/ask gap (¢)", 1, 50, 15, key=f"{sport}_k_gap",
                      help="Wide gaps mean thin markets — the price you see may not fill.") / 100
    n_cards = c4.number_input("Top plays", 1, 30, 8, key=f"{sport}_k_n")
    t1, t2, t3, t4 = st.columns([2, 1, 1.2, 1.2])
    game = t1.selectbox("Game", ["All"] + sorted(edges["game"].unique()), key=f"{sport}_k_game")
    min_vol = t2.number_input("Min volume", 0, 100000, 25, step=25, key=f"{sport}_k_vol",
                              help="Contracts nobody has traded have meaningless prices.")
    one_per = t3.toggle("Best contract per game/player", True, key=f"{sport}_k_one")
    show_sus = t4.toggle("Show huge disagreements", False, key=f"{sport}_k_sus",
                         help="When the model and Kalshi disagree by 20+ points, the model is usually missing "
                              "something (injury, stale data, a market it misread). Hidden by default.")

    base = edges[edges["kind"].isin(pick) & (edges["spread_width"].fillna(1) <= max_w)
                 & (edges["volume"].fillna(0) >= min_vol)]
    if not show_sus:
        base = base[~base["suspect"]]
    if likely:
        # the model's more likely side of each contract, priced at what you'd pay for that side
        rows = []
        for _, r in base.iterrows():
            yes_ok = pd.notna(r["yes_ask"]) and 0 < r["yes_ask"] < 1
            no_ok = pd.notna(r["yes_bid"]) and 0 < r["yes_bid"] < 1
            if r["p_yes"] >= 0.5 and yes_ok:
                side, price, prob = "YES", r["yes_ask"], r["p_yes"]
            elif r["p_yes"] < 0.5 and no_ok:
                side, price, prob = "NO", 1 - r["yes_bid"], 1 - r["p_yes"]
            else:
                continue
            fee = float(kalshi.taker_fee(price))
            rows.append({**r.to_dict(), "side": side, "price": price, "model_prob": prob, "edge": prob - price,
                         "fee": fee, "ev_taker": prob - price - fee,
                         "ev_maker": prob - price - float(kalshi.maker_fee(price))})
        e = pd.DataFrame(rows, columns=list(base.columns))
        if not e.empty:
            e = e[(e["model_prob"] >= min_prob) & (e["price"] <= max_price)].sort_values("model_prob", ascending=False)
    else:
        e = base[base["ev_taker"] >= min_ev]
    if game != "All":
        e = e[e["game"] == game]
    if one_per:
        e = e.assign(_k=e["game"] + "|" + e["kind"] + "|" + e["contract"].str.split(":").str[0]) \
             .drop_duplicates("_k").drop(columns="_k")
    n_sus = int(edges["suspect"].sum())
    st.caption(f"{len(e)} of {len(edges)} matched contracts shown · {n_sus} hidden as huge disagreements · "
               "prices refresh every 90 s · your fill can differ from the displayed ask")
    if e.empty:
        st.info("Nothing clears your filters. Try a lower minimum EV or a wider bid/ask gap.")
    for i, (_, r) in enumerate(e.head(int(n_cards)).iterrows()):
        with st.container(border=True):
            tier = (["Tier 1 · most likely", "Tier 2", "Tier 3"] if likely else TIERS)
            st.caption(f"**{tier[i] if i < 3 else f'Tier {i + 1}'}** · {KIND_LABEL[r['kind']]} · {r['game']}")
            st.markdown(f"#### Buy **{r['side']}** · {r['contract']}")
            a, b, c, d = st.columns(4)
            a.metric("Price", _cents(r["price"]), delta=f"fee {r['fee'] * 100:.1f}¢", delta_color="off")
            b.metric("Model probability", ui.pct(r["model_prob"], 1), delta=f"{r['edge'] * 100:+.1f} pts vs price",
                     delta_color="off")
            c.metric("EV per contract", f"{r['ev_taker'] * 100:+.1f}¢",
                     delta=("priced fairly or better" if r["ev_taker"] >= 0 else "Kalshi's price is a bit high")
                     if likely else f"{r['ev_maker'] * 100:+.1f}¢ as limit order", delta_color="off",
                     help="Taker = buying at the ask right now. Limit (maker) orders pay a quarter of the fee "
                          "if someone fills them.")
            d.metric("Market", f"gap {_cents(r['spread_width'])}", delta=f"vol {r['volume']:,.0f}", delta_color="off")
            warn = " · ⚠️ huge disagreement with the market — check news before trusting" if r["suspect"] else ""
            st.caption(f"{r['detail']} · `{r['ticker']}`{warn}")

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
    st.divider()
    combo_builder(sport, edges, mk, preds, projs)


# =============================================================================
def combo_builder(sport: str, edges: pd.DataFrame, markets: pd.DataFrame, preds: pd.DataFrame, projs: list):
    st.markdown("#### Same-game combo builder")
    st.caption("Pick legs from one game and the app simulates the whole game 40,000 times to price the combo — "
               "legs that tend to happen together (a team winning, its QB throwing for yards, the over) are "
               "counted together instead of multiplied. Kalshi prices combos by quote, so build the same combo in "
               "Kalshi, request a quote, and take it only at or below the **max quote** shown here.")
    if edges.empty:
        return
    cfg = get_config(sport)
    gm = kalshi.match_games(markets, preds, sport)
    proj_idx = {(p["player_norm"], p["stat"]): p for p in projs or []}
    games = edges.drop_duplicates("game_key")[["game_key", "game"]].sort_values("game")
    labels = dict(zip(games["game"], games["game_key"]))
    c1, c2 = st.columns([3, 1])
    gname = c1.selectbox("Game", list(labels), key=f"{sport}_cb_game")
    gk = labels[gname]
    game = gm.get(gk, {}).get("row")
    if game is None:
        st.info("Couldn't match that game to a prediction.")
        return
    opts = combos.leg_options(edges, gk)
    if opts.empty:
        st.info("No tradable contracts for this game.")
        return
    opts = opts.assign(label=opts["label"] + opts["suspect"].map({True: " ⚠️", False: ""}))
    by_label = {r["label"]: r for r in opts.to_dict("records")}
    chosen = st.multiselect("Your legs", list(by_label), key=f"{sport}_cb_legs",
                            help="⚠️ = the model disagrees with Kalshi by 20+ points on that leg; usually the "
                                 "model is missing something.")
    legs = [by_label[l] for l in chosen]
    if len({l["ticker"] for l in legs}) < len(legs):
        st.error("You picked both YES and NO of the same contract — that combo can never win.")
    elif len(legs) >= 2:
        try:
            sim = combos.simulate(legs, game, cfg, sport, proj_idx)
        except ValueError as e:
            st.error(str(e))
            sim = None
        if sim:
            a, b, c, d = st.columns(4)
            a.metric("Model hit chance", ui.pct(sim["joint"], 1),
                     delta=f"{ui.pct(sim['independent'], 1)} if legs were independent", delta_color="off")
            b.metric("Fair price", _cents(sim["fair_price"]), delta=f"pays {1 / max(sim['joint'], 1e-6):.1f}x",
                     delta_color="off")
            c.metric("Max quote to take", _cents(sim["max_quote"]),
                     delta=f"{_cents(sim['max_quote_maker'])} as a limit order", delta_color="off",
                     help="Highest Kalshi quote where the combo still has positive expected value after the "
                          "taker fee, if the model is right.")
            boost = sim["correlation_boost"]
            d.metric("Correlation effect", "—" if pd.isna(boost) else f"{boost:.2f}x",
                     help="Above 1 = these legs tend to hit together, so the combo is more likely than "
                          "multiplying the legs suggests (and Kalshi's quote should reflect that too).")
            st.dataframe(pd.DataFrame([{"Leg": l["label"], "Model %": p * 100, "Leg price (¢)": l["leg_price"] * 100}
                                       for l, p in zip(legs, sim["leg_probs"])]),
                         hide_index=True, use_container_width=True, column_config={
                             "Model %": st.column_config.NumberColumn(format="%.1f"),
                             "Leg price (¢)": st.column_config.NumberColumn(format="%.0f")})
    elif legs:
        st.caption("Add at least one more leg.")

    st.markdown("**Or let the app suggest combos for this game**")
    s1, s2, s3, s4 = st.columns([1, 1.6, 1.4, 1.4])
    n_legs = s1.selectbox("Legs", [2, 3, 4], key=f"{sport}_cb_n")
    mode = s2.radio("Build from", ["Most likely legs", "Best-value legs"], key=f"{sport}_cb_mode")
    leg_min = s3.slider("Min chance per leg", 50, 90, 65, key=f"{sport}_cb_minp",
                        disabled=mode != "Most likely legs") / 100
    s4.write("")
    if s4.button("Suggest combos", key=f"{sport}_cb_go"):
        with st.spinner("Simulating combinations…"):
            st.session_state[f"{sport}_cb_sug"] = (gk, n_legs, mode, combos.suggest(
                opts, game, cfg, sport, proj_idx, n_legs=n_legs, min_prob=leg_min,
                mode="likely" if mode == "Most likely legs" else "value"))
    saved = st.session_state.get(f"{sport}_cb_sug")
    if saved and saved[0] == gk and saved[1] == n_legs and saved[2] == mode:
        sug = saved[3]
        if not sug:
            st.info("No combos found for this game with those settings — try a lower minimum chance per leg "
                    "or fewer legs.")
        for i, sgg in enumerate(sug, 1):
            with st.container(border=True):
                st.markdown(f"**Combo {i}** · " + " + ".join(l["label"] for l in sgg["legs"]))
                a, b, c, d = st.columns(4)
                a.metric("Model hit chance", ui.pct(sgg["joint"], 1))
                b.metric("Fair price", _cents(sgg["fair_price"]))
                c.metric("Max quote to take", _cents(sgg["max_quote"]))
                d.metric("Legs bought separately", _cents(sgg["naive_price"]),
                         help="Product of the leg prices — a rough reference. Kalshi's combo quote accounts for "
                              "correlation, so the real quote is the only true test.")
        if sug:
            st.caption(("Most-likely combos are ranked by the chance every leg hits; likely legs cost more, so "
                        "check that the Kalshi quote is at or below the max quote. " if mode == "Most likely legs" else
                        "Value combos use only legs where the model beats Kalshi's price on its own. ")
                       + "Two rungs of the same ladder are never stacked. If Kalshi's combo builder doesn't offer a "
                         "NO side, use the opposite contract instead (e.g. the other team's \"wins\" market).")
