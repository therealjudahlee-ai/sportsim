import altair as alt
import pandas as pd
import streamlit as st

from engine import tracking
from engine.config import DEFAULTS

st.set_page_config(page_title="Track record · SportSim", page_icon="📈", layout="wide")
st.markdown("## Track record")
st.caption("Every pre-game prediction is logged automatically and graded once the game is final. Picks are graded "
           "against the line at the time of the FIRST prediction; closing-line value (CLV) shows whether the market "
           "later moved toward our side.")

g = tracking.graded()
if g.empty:
    st.info("Nothing graded yet. The GitHub Action in `.github/workflows/refresh.yml` logs predictions twice a "
            "day and commits them to `predictions/`; results show up here after the first games finish. "
            "You can also run `python -m jobs.refresh` yourself.")
    st.stop()

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
