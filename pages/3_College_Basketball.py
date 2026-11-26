import streamlit as st

from engine.common import today_et
from engine.config import get_config
from ui import components as ui
from ui.kalshi_tab import kalshi_tab
from ui import loaders

st.set_page_config(page_title="College Basketball · SportSim", page_icon="🎓", layout="wide")
cfg = get_config("ncaab")

date_str = st.sidebar.date_input("Slate date", value=today_et(), key="cbb_date").isoformat()
try:
    ctx = loaders.ncaab_ctx(date_str)
except Exception as e:
    st.error(f"Couldn't load college basketball data: {type(e).__name__}: {e}")
    if st.button("Retry"):
        loaders.ncaab_ctx.clear()
        st.rerun()
    st.stop()

preds, src = loaders.with_market(ctx["predictions"], "ncaab")
n_known = int(preds["known"].sum()) if not preds.empty else 0
ui.page_header(f"Men's College Basketball · {date_str}",
               f"{len(preds)} D-I games ({n_known} with both teams rated) · tempo-free, opponent-adjusted")
ui.data_status(ctx, on_refresh=loaders.ncaab_ctx.clear)

tabs = st.tabs(["Games", "Kalshi", "Team ratings"])
with tabs[0]:
    ui.games_board(preds, cfg, src, key="cbb_games", default_cards=12)
with tabs[1]:
    kalshi_tab("ncaab", ctx, preds)
with tabs[2]:
    ui.ratings_table(ctx["ratings"], names=ctx["names"], uses_poss=True)
