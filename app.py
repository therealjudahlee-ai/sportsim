"""SportSim — home page.  Run:  streamlit run app.py"""
import streamlit as st

from engine import tracking

st.set_page_config(page_title="SportSim", page_icon="📊", layout="wide")

st.markdown("# SportSim")
st.caption("Opponent-adjusted game projections, player props and DFS for the NFL, NBA, college basketball and "
           "college football — compared against real market prices.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.page_link("pages/1_NFL.py", label="NFL", icon="🏈")
c2.page_link("pages/2_NBA.py", label="NBA", icon="🏀")
c3.page_link("pages/3_College_Basketball.py", label="College Basketball", icon="🎓")
c4.page_link("pages/4_College_Football.py", label="College Football", icon="🏟️")
c5.page_link("pages/5_Track_Record.py", label="Track record", icon="📈")

st.divider()
g = tracking.graded()
if not g.empty:
    st.markdown("#### Live track record")
    cols = st.columns(max(1, g["sport"].nunique()))
    for col, (sport, grp) in zip(cols, g.groupby("sport")):
        s = tracking.summary(grp)
        col.metric(sport.upper(), f"{s['ats_record']} ATS",
                   delta=f"{s['su_accuracy']:.0%} straight-up · {s['games']} games", delta_color="off")
    st.divider()

st.markdown("""
#### How the numbers are made
- **Team ratings** — every team gets an offense and defense rating from a recency-weighted regression over
  *all* games, adjusted for opponent strength and home court. Basketball is rated per 100 possessions with a
  separate pace model. Early in the season, ratings start from last year's (shrunk toward average) and the
  new season's games take over as they come in.
- **Game probabilities** — projected margin and total, with the spread of outcomes fit on past seasons, give
  exact win / cover / over probabilities. The model never blends in the betting line, so the gap between the
  model and the market is an honest edge.
- **Player props** — projected volume × efficiency × matchup × game script, with a distribution that fits the
  stat (skewed for yards and points, discrete for counts). Compared against real sportsbook lines with the
  vig removed.
- **Accountability** — every pre-game prediction is logged automatically and graded after the game. The
  *Track record* page shows the results, including whether the market moved toward our side (closing-line value).
""")
st.caption("For entertainment and research. Projections are estimates, not guarantees. If you bet, bet "
           "responsibly and only where it's legal for you.")
