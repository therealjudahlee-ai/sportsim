import streamlit as st

from engine.config import get_config
from ui import components as ui
from ui import loaders, pro_pages

st.set_page_config(page_title="NFL · SportSim", page_icon="🏈", layout="wide")
cfg = get_config("nfl")

try:
    ctx = loaders.nfl_ctx()
except Exception as e:
    st.error(f"Couldn't load NFL data: {type(e).__name__}: {e}")
    if st.button("Retry"):
        loaders.nfl_ctx.clear()
        st.rerun()
    st.stop()

preds, src = loaders.with_market(ctx["predictions"], "nfl")
ui.page_header(f"NFL · {ctx['season']} Week {ctx['week']}",
               f"{len(preds)} games · injuries from nflverse + ESPN")
ui.data_status(ctx, on_refresh=loaders.nfl_ctx.clear)

tabs = st.tabs(["Games", "Player props", "2x–3x Value", "Parlays", "DFS", "Team ratings"])
with tabs[0]:
    ui.games_board(preds, cfg, src, key="nfl_games", default_cards=16)
with tabs[1]:
    pro_pages.props_tab("nfl", ctx)
with tabs[2]:
    pro_pages.value_tab("nfl", ctx)
with tabs[3]:
    pro_pages.parlay_tab("nfl")
with tabs[4]:
    pro_pages.dfs_tab("nfl", ctx)
with tabs[5]:
    ui.ratings_table(ctx["ratings"])
    if ctx["team_adj"]:
        st.caption("Injury adjustments this week (points): " +
                   ", ".join(f"{t} {v:+.1f}" for t, v in sorted(ctx["team_adj"].items(), key=lambda x: x[1])))
