import streamlit as st

from engine.config import get_config
from ui import components as ui
from ui import loaders

st.set_page_config(page_title="College Football · SportSim", page_icon="🏟️", layout="wide")
cfg = get_config("ncaaf")

try:
    ctx = loaders.ncaaf_ctx()
except Exception as e:
    st.error(f"Couldn't load college football data: {type(e).__name__}: {e}")
    if st.button("Retry"):
        loaders.ncaaf_ctx.clear()
        st.rerun()
    st.stop()

preds, src = loaders.with_market(ctx["predictions"], "ncaaf")
ui.page_header(f"College Football · {ctx['season']}", f"{len(preds)} upcoming FBS games")
ui.data_status(ctx, on_refresh=loaders.ncaaf_ctx.clear)

tabs = st.tabs(["Games", "Team ratings"])
with tabs[0]:
    ui.games_board(preds, cfg, src, key="cfb_games", default_cards=12)
with tabs[1]:
    ui.ratings_table(ctx["ratings"], names=ctx["names"])
