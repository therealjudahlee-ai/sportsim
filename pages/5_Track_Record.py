import altair as alt
import pandas as pd
import streamlit as st

from engine import kalshi_track, tracking
from engine.config import DEFAULTS
from ui import loaders

st.set_page_config(page_title="Track record · SportSim", page_icon="📈", layout="wide")
st.markdown("## Track record")
st.caption("Every pre-game prediction is logged automatically and graded once the game is final. Picks are graded "
           "against the line at the time of the FIRST prediction; closing-line value (CLV) shows whether the market "
           "later moved toward our side.")

top = st.tabs(["Game picks", "Kalshi edges", "My Kalshi account"])

# =============================================================================
with top[0]:
    g = tracking.graded()
    if g.empty:
        st.info("Nothing graded yet. The GitHub Action in `.github/workflows/refresh.yml` logs predictions twice a "
                "day and commits them to `predictions/`; results show up here after the first games finish.")
    else:
        sports = sorted(g["sport"].unique())
        tabs = st.tabs([DEFAULTS[s].label if s in DEFAULTS else s for s in sports])
        for tab, sport in zip(tabs, sports):
            with tab:
                gs = g[g["sport"] == sport]
                min_edge = st.slider("Only count spread picks where the model disagreed with the line by at least (pts)",
                                     0.0, 8.0, 0.0, 0.5, key=f"tr_edge_{sport}")
                s = tracking.summary(gs, min_edge)
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Games graded", s["games"])
                c2.metric("Straight-up", f"{s['su_accuracy']:.1%}")
                c3.metric("Against the spread", s["ats_record"],
                          delta=None if pd.isna(s["ats_pct"]) else f"{s['ats_pct']:.1%} (52.4% breaks even at -110)",
                          delta_color="off")
                c4.metric("Brier score", f"{s['brier']:.3f}",
                          delta=None if pd.isna(s["market_brier"]) else f"market {s['market_brier']:.3f} (lower is better)",
                          delta_color="off")
                c5.metric("Avg CLV", "—" if pd.isna(s["avg_clv_pts"]) else f"{s['avg_clv_pts']:+.2f} pts",
                          help="Positive = the line moved toward our pick after we logged it.")

                cal = tracking.calibration(gs)
                left, right = st.columns([1, 1])
                with left:
                    st.markdown("**Calibration** — when we say 70%, do favorites win 70%?")
                    if not cal.empty:
                        pts = alt.Chart(cal).mark_circle().encode(
                            x=alt.X("predicted:Q", scale=alt.Scale(domain=[0.5, 1]), title="Predicted win %"),
                            y=alt.Y("actual:Q", scale=alt.Scale(domain=[0, 1]), title="Actual win %"),
                            size=alt.Size("n:Q", legend=None), tooltip=["bin", "predicted", "actual", "n"])
                        diag = alt.Chart(pd.DataFrame({"x": [0.5, 1], "y": [0.5, 1]})).mark_line(
                            strokeDash=[4, 4], color="gray").encode(x="x", y="y")
                        st.altair_chart((diag + pts).properties(height=280), use_container_width=True)
                with right:
                    st.markdown("**ATS by size of disagreement**")
                    ats = gs[gs["ats_result"].isin([0.0, 1.0])].copy()
                    if not ats.empty:
                        ats["edge_bucket"] = pd.cut(ats["ats_edge"], [0, 1, 2, 3, 5, 100],
                                                    labels=["0–1", "1–2", "2–3", "3–5", "5+"])
                        tb = ats.groupby("edge_bucket", observed=True)["ats_result"].agg(["mean", "size"]).reset_index()
                        tb.columns = ["Edge (pts)", "Cover %", "Picks"]
                        tb["Cover %"] *= 100
                        st.dataframe(tb, hide_index=True, use_container_width=True,
                                     column_config={"Cover %": st.column_config.NumberColumn(format="%.1f%%")})

                st.markdown("**Recent graded games**")
                recent = gs.sort_values("date", ascending=False).head(50)
                st.dataframe(pd.DataFrame({
                    "Date": recent["date"].astype(str).str[:10],
                    "Game": recent["away_name"] + " @ " + recent["home_name"],
                    "Model margin": recent["model_margin"].round(1), "Line": recent["home_spread"],
                    "Final": recent["away_pts"].astype(int).astype(str) + "–" + recent["home_pts"].astype(int).astype(str),
                    "Win %": (recent["home_win"] * 100).round(0), "ATS": recent["ats_result"].map(
                        {1.0: "✅", 0.0: "❌", 0.5: "push"}).fillna("—"),
                }), hide_index=True, use_container_width=True)

# =============================================================================
with top[1]:
    st.caption("Every refresh logs the Kalshi contracts the model flags (EV ≥ 2¢ after fees, reasonably liquid) at "
               "the price shown at that moment, then grades them when Kalshi settles the market. Profit is per $1 "
               "contract after the taker fee — i.e. what you'd have made buying every flagged contract once.")
    kg = kalshi_track.graded()
    if kg.empty:
        st.info("No settled Kalshi contracts yet. They'll appear a day or two after the first flagged games finish.")
    else:
        ksports = ["all"] + sorted(kg["sport"].unique())
        pick = st.radio("Sport", ksports, horizontal=True, key="k_tr_sport")
        kk = kg if pick == "all" else kg[kg["sport"] == pick]
        s = kalshi_track.summary(kk)
        a, b, c, d, e = st.columns(5)
        a.metric("Contracts graded", s["contracts"])
        b.metric("Won", f"{s['win_rate']:.1%}", delta=f"model expected {s['avg_model_prob']:.1%}", delta_color="off")
        c.metric("Profit / contract", f"{s['profit_per_contract'] * 100:+.1f}¢",
                 delta=f"total {s['total_profit']:+.2f} per 1-contract stakes", delta_color="off")
        d.metric("ROI", f"{s['roi']:+.1%}")
        e.metric("Avg CLV", "—" if pd.isna(s["avg_clv"]) else f"{s['avg_clv'] * 100:+.1f}¢",
                 help="Last traded price before close minus the price we logged. Positive = market moved our way.")
        by = kk.groupby("kind").agg(contracts=("won", "size"), won=("won", "mean"), profit=("profit", "sum"),
                                    clv=("clv", "mean")).reset_index()
        by["won"] *= 100
        by["clv"] *= 100
        st.dataframe(by, hide_index=True, use_container_width=True, column_config={
            "won": st.column_config.NumberColumn("Won %", format="%.1f%%"),
            "profit": st.column_config.NumberColumn("Profit ($/contract total)", format="%+.2f"),
            "clv": st.column_config.NumberColumn("Avg CLV (¢)", format="%+.1f")})
        show = kk.sort_values("logged_at", ascending=False).head(60)
        st.dataframe(pd.DataFrame({
            "Logged": show["logged_at"].astype(str).str[:16], "Game": show["game"], "Contract": show["contract"],
            "Side": show["side"], "Price (¢)": (show["price"] * 100).round(0),
            "Model %": (show["model_prob"] * 100).round(1), "Result": show["won"].map({True: "✅", False: "❌"}),
            "Profit (¢)": (show["profit"] * 100).round(1)}), hide_index=True, use_container_width=True)

# =============================================================================
with top[2]:
    kid, pem = loaders.kalshi_keys()
    if not (kid and pem):
        st.info("Optional: see your own Kalshi results here (record, profit after fees, by sport). Create an API key "
                "in Kalshi → Account → Profile Settings → **API Keys**, then add `KALSHI_KEY_ID` and "
                "`KALSHI_PRIVATE_KEY` to your Streamlit secrets. The app only *reads* your settlements — it never "
                "places orders — but a Kalshi key may be able to trade, so keep it in secrets and never share it.")
    else:
        mine, warns = loaders.kalshi_settlements()
        for w in warns:
            st.warning(w)
        if mine.empty:
            st.info("No settled positions found on your Kalshi account yet.")
        else:
            staked = float((mine["cost"] + mine["fees"]).sum())
            a, b, c, d = st.columns(4)
            a.metric("Settled positions", len(mine))
            b.metric("Profit after fees", f"${mine['profit'].sum():,.2f}")
            c.metric("ROI", f"{mine['profit'].sum() / staked:+.1%}" if staked else "—")
            d.metric("Fees paid", f"${mine['fees'].sum():,.2f}")
            by = mine.groupby("sport").agg(positions=("profit", "size"), profit=("profit", "sum"),
                                           staked=("cost", "sum")).reset_index()
            st.dataframe(by, hide_index=True, use_container_width=True, column_config={
                "profit": st.column_config.NumberColumn("Profit ($)", format="%+.2f"),
                "staked": st.column_config.NumberColumn("Cost ($)", format="%.2f")})
            st.dataframe(mine.head(100)[["settled", "ticker", "result", "yes_count", "no_count", "cost", "fees",
                                         "revenue", "profit"]], hide_index=True, use_container_width=True)
