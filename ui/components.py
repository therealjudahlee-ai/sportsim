"""Reusable UI pieces."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from engine.config import SportConfig
from engine.gamesim import margin_density


# ----------------------------------------------------------------------------- formatting
def pct(x, d=0):
    return "—" if x is None or pd.isna(x) else f"{x:.{d}%}"


def line(team, pts):
    if pts is None or pd.isna(pts):
        return "—"
    if abs(pts) < 0.25:
        return f"{team} PK"
    return f"{team} {pts:+.1f}"


def short(name: str, n: int = 22) -> str:
    name = str(name)
    return name if len(name) <= n else name[: n - 1] + "…"


# ----------------------------------------------------------------------------- page chrome
def page_header(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def data_status(ctx: dict, on_refresh=None, extra: str = ""):
    with st.sidebar:
        st.divider()
        st.caption(f"Data built {ctx.get('built_at', '?')} ET")
        if extra:
            st.caption(extra)
        warns = ctx.get("diag", {}).get("warnings", [])
        info = ctx.get("diag", {}).get("info", [])
        with st.expander(f"Data status ({len(warns)} warning{'s' if len(warns) != 1 else ''})", expanded=False):
            for w in warns:
                st.warning(w, icon="⚠️")
            for i in info:
                st.caption(i)
        if on_refresh and st.button("Reload data", use_container_width=True):
            on_refresh()
            st.rerun()


# ----------------------------------------------------------------------------- games
SORTS = {
    "Biggest spread edge": lambda d: -d["spread_edge_pts"].abs().fillna(-1),
    "Biggest moneyline edge": lambda d: -d["win_edge"].abs().fillna(-1),
    "Most confident winner": lambda d: -d["pick_prob"],
    "Biggest total edge": lambda d: -d["total_edge_pts"].abs().fillna(-1),
}


def games_board(preds: pd.DataFrame, cfg: SportConfig, line_source: str, key: str, default_cards: int = 8):
    if preds is None or preds.empty:
        st.info("No games on this slate.")
        return
    has_mkt = preds["home_spread"].notna().any()
    c1, c2, c3 = st.columns([2, 1.2, 1])
    sort = c1.selectbox("Rank games by", list(SORTS) if has_mkt else ["Most confident winner"], key=f"{key}_sort")
    min_conf = c2.slider("Min win probability", 50, 95, 50, 1, key=f"{key}_minp") / 100
    n_cards = c3.number_input("Cards", 1, 60, default_cards, key=f"{key}_n")
    show_unknown = st.toggle("Include games with unrated teams", value=False, key=f"{key}_unk")

    df = preds.copy()
    if not show_unknown:
        df = df[df["known"]]
    df = df[df["pick_prob"] >= min_conf]
    df = df.assign(_k=SORTS[sort](df)).sort_values("_k")
    st.caption(f"{len(df)} games · lines from {line_source} · probabilities are the model's own (no market blending)")

    for _, r in df.head(int(n_cards)).iterrows():
        game_card(r, cfg)

    with st.expander("Full slate table"):
        tbl = pd.DataFrame({
            "Matchup": df["away_name"].fillna(df["away"]) + " @ " + df["home_name"].fillna(df["home"]),
            "Proj score": [f"{a:.1f}–{h:.1f}" for a, h in zip(df["model_away_pts"], df["model_home_pts"])],
            "Pick": df["pick_name"], "Win %": df["pick_prob"] * 100, "Mkt win %": df["market_pick_prob"] * 100,
            "Model line": [line("Home", -m) for m in df["model_margin"]],
            "Market line": [line("Home", s) for s in df["home_spread"]],
            "ATS lean": [("Home" if a == "home" else "Away" if a == "away" else "—") for a in df["ats_pick"]],
            "ATS %": df["ats_prob"] * 100, "Model total": df["model_total"], "Total": df["total_line"],
            "O/U %": df["total_prob"] * 100, "Notes": df["notes"],
        })
        st.dataframe(tbl, hide_index=True, use_container_width=True, column_config={
            "Win %": st.column_config.ProgressColumn(format="%.0f%%", min_value=50, max_value=100),
            "Mkt win %": st.column_config.NumberColumn(format="%.0f%%"),
            "ATS %": st.column_config.NumberColumn(format="%.0f%%"),
            "O/U %": st.column_config.NumberColumn(format="%.0f%%"),
            "Model total": st.column_config.NumberColumn(format="%.1f"),
        })
        st.download_button("Download CSV", df.drop(columns="_k").to_csv(index=False).encode(),
                           f"{cfg.key}_slate.csv", "text/csv", key=f"{key}_dl")


def game_card(r: pd.Series, cfg: SportConfig):
    home, away = r.get("home_name") or r["home"], r.get("away_name") or r["away"]
    with st.container(border=True):
        when = ""
        if "date" in r and pd.notna(r["date"]):
            try:
                when = pd.Timestamp(r["date"]).tz_convert("America/New_York").strftime("%a %-I:%M %p ET")
            except (TypeError, ValueError):
                when = pd.Timestamp(r["date"]).strftime("%a %b %-d")
        st.markdown(f"**{away} @ {home}**  \n<span style='opacity:.6'>{when}"
                    f"{' · neutral site' if r.get('neutral') else ''}</span>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Projected score", f"{r['model_away_pts']:.0f}–{r['model_home_pts']:.0f}",
                  help=f"{away} {r['model_away_pts']:.1f}, {home} {r['model_home_pts']:.1f}")
        edge = r.get("win_edge")
        c2.metric(f"Win: {short(r['pick_name'], 16)}", pct(r["pick_prob"]),
                  delta=None if pd.isna(edge) else f"{edge:+.1%} vs market",
                  help=f"Fair moneyline {r['pick_fair_ml']}. Market (no-vig): {pct(r.get('market_pick_prob'))}")
        if pd.notna(r.get("home_spread")):
            side = home if r["ats_pick"] == "home" else away
            mkt = r["home_spread"] if r["ats_pick"] == "home" else -r["home_spread"]
            c3.metric("Spread lean", f"{short(side, 14)} {mkt:+.1f}", delta=f"{pct(r['ats_prob'])} to cover",
                      delta_color="off",
                      help=f"Model line {line(home, -r['model_margin'])} vs market {line(home, r['home_spread'])} "
                           f"({r['spread_edge_pts']:+.1f} pts)")
        else:
            c3.metric("Model line", line(short(home, 14), -r["model_margin"]))
        if pd.notna(r.get("total_line")):
            c4.metric("Total lean", f"{r['total_pick'].title()} {r['total_line']:.1f}",
                      delta=f"{pct(r['total_prob'])} · model {r['model_total']:.1f}", delta_color="off")
        else:
            c4.metric("Model total", f"{r['model_total']:.1f}")
        bits = [b for b in [r.get("notes"), r.get("injury_notes")] if isinstance(b, str) and b]
        if r.get("blowout", 0) >= 0.35:
            bits.append(f"blowout risk {r['blowout']:.0%}")
        if not r.get("known", True):
            bits.append("⚠️ a team has no rating history — treat as low confidence")
        if bits:
            st.caption(" · ".join(bits))
        with st.expander("Outcome distribution"):
            st.altair_chart(margin_chart(r, cfg), use_container_width=True)


def margin_chart(r: pd.Series, cfg: SportConfig):
    home = short(r.get("home_name") or r["home"], 18)
    d = margin_density(r["model_margin"], cfg)
    d["winner"] = np.where(d["margin"] > 0, home, short(r.get("away_name") or r["away"], 18))
    area = alt.Chart(d).mark_area(opacity=0.55).encode(
        x=alt.X("margin:Q", title=f"Final margin ({home} perspective)"),
        y=alt.Y("density:Q", axis=None),
        color=alt.Color("winner:N", legend=alt.Legend(title=None, orient="top")),
    )
    rules = [alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="gray").encode(x="x:Q")]
    if pd.notna(r.get("home_spread")):
        rules.append(alt.Chart(pd.DataFrame({"x": [-r["home_spread"]], "lbl": ["market"]}))
                     .mark_rule(strokeDash=[5, 4], color="#e45756").encode(x="x:Q", tooltip=["lbl"]))
    return alt.layer(area, *rules).properties(height=180)


def ratings_table(ratings, names: dict | None = None, top: int | None = None, uses_poss=False):
    t = ratings.table()
    if t.empty:
        st.info("No ratings yet.")
        return
    if names:
        t["team"] = t["team"].map(lambda k: names.get(k, k))
    t.insert(0, "rank", range(1, len(t) + 1))
    unit = "per 100 poss." if uses_poss else "points"
    st.caption(f"Opponent-adjusted, recency-weighted ratings ({unit} vs an average team). "
               f"Home edge ≈ {ratings.hfa_points():.1f} pts.")
    cols = ["rank", "team", "net", "off", "def", "games"] + (["pace"] if uses_poss else [])
    if "net_epa" in t.columns:
        cols = ["rank", "team", "net", "net_epa", "off_epa", "def_epa", "off", "def", "games"]
        st.caption("NFL: 'net' blends score-based and play-by-play (EPA per play) ratings. EPA columns are per "
                   "play; off/def are points per game.")
    st.dataframe(t[cols].head(top) if top else t[cols], hide_index=True, use_container_width=True,
                 column_config={"net_epa": st.column_config.NumberColumn("net EPA/play", format="%+.3f"),
                                "off_epa": st.column_config.NumberColumn("off EPA/play", format="%+.3f"),
                                "def_epa": st.column_config.NumberColumn("def EPA/play (− good)", format="%+.3f"),
                                "net": st.column_config.NumberColumn(format="%+.1f"),
                                "off": st.column_config.NumberColumn("off (+ good)", format="%+.1f"),
                                "def": st.column_config.NumberColumn("def (− good)", format="%+.1f"),
                                "pace": st.column_config.NumberColumn(format="%.1f")})


# ----------------------------------------------------------------------------- props
def prop_detail(proj: dict, line_val: float):
    d = proj["dist"]
    p_over, p_under = d.prob_over(line_val), d.prob_under(line_val)
    from engine.common import prob_to_american
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projection (median)", f"{proj['median']:.1f}", help=f"Mean {proj['mean']:.2f}")
    c2.metric(f"Over {line_val:g}", pct(p_over, 1), delta=f"fair {prob_to_american(p_over)}", delta_color="off")
    c3.metric(f"Under {line_val:g}", pct(p_under, 1), delta=f"fair {prob_to_american(p_under)}", delta_color="off")
    c4.metric("Range (20th–80th)", f"{proj['p20']:.0f}–{proj['p80']:.0f}", delta=f"{proj['games_used']} games",
              delta_color="off")
    hist = pd.Series(proj.get("hist", []), dtype=float)
    if len(hist):
        rates = {f"Last {n}": (hist.tail(n) > line_val).mean() for n in (5, 10, 20) if len(hist) >= n}
        st.caption("Hit rate vs this line — " + " · ".join(f"{k}: {v:.0%}" for k, v in rates.items()) +
                   f" · recent: {', '.join(f'{x:g}' for x in hist.tail(8))}")
    notes = list(proj.get("notes", []))
    if proj.get("def_mult") and abs(proj["def_mult"] - 1) >= 0.03:
        notes.append(f"matchup x{proj['def_mult']:.2f}")
    if proj.get("status"):
        notes.append(f"status: {proj['status']}")
    if notes:
        st.caption(" · ".join(notes))
    s = pd.DataFrame({"value": d.sample(20_000, seed=3)})
    s["side"] = np.where(s["value"] > line_val, "over", "under")
    chart = alt.Chart(s).mark_bar(opacity=0.8).encode(
        x=alt.X("value:Q", bin=alt.Bin(maxbins=40), title=proj["stat"]),
        y=alt.Y("count()", axis=None), color=alt.Color("side:N", legend=alt.Legend(orient="top", title=None)))
    rule = alt.Chart(pd.DataFrame({"x": [line_val]})).mark_rule(color="black").encode(x="x:Q")
    st.altair_chart((chart + rule).properties(height=200), use_container_width=True)


def edges_table(edges: pd.DataFrame, key: str):
    if edges is None or edges.empty:
        st.info("No props matched to projections yet.")
        return
    show = edges.copy()
    for c in ("prob", "market_prob", "edge"):
        show[c] = show[c] * 100
    show["Pick"] = show["side"].str.title() + " " + show["line"].map(lambda x: f"{x:g}")
    show["Price"] = show["best_price"].map(lambda x: "—" if pd.isna(x) else (f"+{x:.0f}" if x > 0 else f"{x:.0f}"))
    cols = ["player", "team", "game", "stat", "Pick", "proj_median", "prob", "market_prob", "edge", "ev", "Price",
            "book", "n_books"]
    st.dataframe(show[cols], hide_index=True, use_container_width=True, column_config={
        "proj_median": st.column_config.NumberColumn("Proj", format="%.1f"),
        "prob": st.column_config.NumberColumn("Model %", format="%.1f%%"),
        "market_prob": st.column_config.NumberColumn("Market % (no-vig)", format="%.1f%%"),
        "edge": st.column_config.NumberColumn("Edge", format="%+.1f%%"),
        "ev": st.column_config.NumberColumn("EV / $1", format="%+.3f"),
        "n_books": st.column_config.NumberColumn("Books"),
    })
    st.download_button("Download CSV", edges.to_csv(index=False).encode(), f"{key}_edges.csv", "text/csv",
                       key=f"{key}_dl")


def parlay_tickets(tickets: list[dict]):
    if not tickets:
        st.info("No qualifying parlays — lower the minimum edge or legs.")
        return
    for i, t in enumerate(tickets, 1):
        with st.container(border=True):
            tag = "same game" if t["same_game"] else f"{len({l['game'] for l in t['legs']})} games"
            st.markdown(f"**Ticket {i}** · {len(t['legs'])} legs · {tag}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model hit prob", pct(t["model_prob"], 1))
            c2.metric("Market (indep.)", pct(t["market_prob"], 1))
            c3.metric("Fair payout", f"{t['fair_payout']:.2f}x")
            c4.metric("EV / $1 at best prices", "—" if pd.isna(t["ev"]) else f"{t['ev']:+.2f}",
                      help="Assumes the book pays the straight product of each leg's best price. "
                           "Same-game parlays are usually priced lower than that.")
            st.dataframe(pd.DataFrame([{
                "Player": l["player"], "Team": l["team"], "Pick": f"{l['side'].title()} {l['line']:g} {l['stat']}",
                "Model %": l["prob"] * 100, "Market %": l["market_prob"] * 100, "Price": l["best_price"],
                "Book": l["book"],
            } for l in t["legs"]]), hide_index=True, use_container_width=True,
                column_config={"Model %": st.column_config.NumberColumn(format="%.1f%%"),
                               "Market %": st.column_config.NumberColumn(format="%.1f%%")})


def no_odds_key_note():
    st.info(
        "**Real prop lines are off.** Add an API key from [The Odds API](https://the-odds-api.com) "
        "(free tier available) to `.streamlit/secrets.toml` as `ODDS_API_KEY = \"...\"`, or in your "
        "Streamlit Cloud app's *Settings → Secrets*. Until then you can still check any line you're "
        "looking at with the single-player tool, or browse projections below.", icon="🔑")
