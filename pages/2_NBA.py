import streamlit as st

from engine.common import today_et
from engine.config import get_config
from ui import components as ui
from ui import loaders, pro_pages

st.set_page_config(page_title="NBA · SportSim", page_icon="🏀", layout="wide")
cfg = get_config("nba")

date_str = st.sidebar.date_input("Slate date", value=today_et(), key="nba_date").isoformat()
try:
    ctx = loaders.nba_ctx(date_str)
except Exception as e:
    st.error(f"Couldn't load NBA data: {type(e).__name__}: {e}")
    if st.button("Retry"):
        loaders.nba_ctx.clear()
        st.rerun()
    st.stop()

preds, src = loaders.with_market(ctx["predictions"], "nba")
ui.page_header(f"NBA · {date_str}", f"{len(preds)} games · {ctx['season'] - 1}-{str(ctx['season'])[-2:]} season")
ui.data_status(ctx, on_refresh=loaders.nba_ctx.clear)

tabs = st.tabs(["Games", "Player props", "2x–3x Value", "Parlays", "DFS", "Team ratings"])
with tabs[0]:
    ui.games_board(preds, cfg, src, key="nba_games", default_cards=15)
with tabs[1]:
    pro_pages.props_tab("nba", ctx, date_str)
with tabs[2]:
    pro_pages.value_tab("nba", ctx, date_str)
with tabs[3]:
    pro_pages.parlay_tab("nba")
with tabs[4]:
    pro_pages.dfs_tab("nba", ctx, date_str)
with tabs[5]:
    ui.ratings_table(ctx["ratings"], uses_poss=True)
    if ctx["team_adj"]:
        st.caption("Injury adjustments today (points): " +
                   ", ".join(f"{t} {v:+.1f}" for t, v in sorted(ctx["team_adj"].items(), key=lambda x: x[1])))
