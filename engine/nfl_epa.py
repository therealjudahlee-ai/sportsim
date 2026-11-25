"""NFL efficiency ratings from play-by-play (EPA = expected points added).

Final scores are noisy: a fumble on the 1, a pick-six or a short field can swing a
game by 14 points without saying much about how good either team is. EPA per play
measures how much each snap moved the expected score, so it's a far more stable
signal of team quality — and it's already in points, which makes it easy to blend
with the score-based ratings.

The blended margin = epa_weight * (EPA-based margin) + (1 - epa_weight) * (score-based
margin). epa_weight is tuned by the backtest. Totals still come from the score model.

Also measures quarterbacks in EPA per dropback, so "starter out" injuries are priced
by the actual gap between that starter and his backup instead of a flat number.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .config import SportConfig
from .ratings import RatingsModel, TeamRatings

PBP_COLS = ["game_id", "season", "week", "season_type", "game_date", "home_team", "away_team", "posteam",
            "defteam", "play_type", "epa", "wp", "qb_dropback", "passer_player_id", "passer_player_name"]

QB_REPLACEMENT_EPA = -0.12   # EPA/dropback of a typical emergency backup
QB_SHRINK_DROPBACKS = 150.0  # pull small samples toward replacement level
DROPBACKS_PER_GAME = 36.0
QB_OUT_CAP = 9.0             # max points for any single QB absence


def load_pbp_lean(nfl_module, seasons: list[int]) -> pd.DataFrame:
    """Load play-by-play one season at a time, keeping only the needed columns
    (full pbp is ~370 columns; this keeps peak memory low enough for Streamlit Cloud)."""
    import time
    frames = []
    for s in seasons:
        for attempt in range(3):          # nflverse files can 404 briefly while being re-uploaded
            try:
                raw = nfl_module.load_pbp([s])
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(20)
        frames.append(raw.select([c for c in PBP_COLS if c in raw.columns]).to_pandas())
        del raw
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PBP_COLS)


def team_game_epa(pbp: pd.DataFrame, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-game offensive EPA/play for each team, in the games-schema the ratings model uses
    (home_pts/away_pts hold EPA/play x 100). Garbage time (win prob < 5% or > 95%) is dropped."""
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna() & pbp["posteam"].notna()]
    if "season_type" in p.columns:
        p = p[p["season_type"].isin(["REG", "POST"])]
    if "wp" in p.columns:
        p = p[p["wp"].isna() | p["wp"].between(0.05, 0.95)]
    if p.empty:
        return pd.DataFrame()
    g = (p.groupby(["game_id", "posteam"])
         .agg(epa=("epa", "mean"), plays=("epa", "size"), season=("season", "first"),
              date=("game_date", "first"), home=("home_team", "first"), away=("away_team", "first"))
         .reset_index())
    h = g[g["posteam"] == g["home"]].set_index("game_id")
    a = g[g["posteam"] == g["away"]].set_index("game_id")
    j = h.join(a[["epa", "plays"]], rsuffix="_a", how="inner")
    out = pd.DataFrame({
        "game_id": j.index.astype(str), "date": pd.to_datetime(j["date"]), "season": j["season"].astype(int),
        "home": j["home"], "away": j["away"],
        "home_pts": j["epa"] * 100.0, "away_pts": j["epa_a"] * 100.0,
        "plays_home": j["plays"], "plays_away": j["plays_a"], "neutral": False,
    }).reset_index(drop=True)
    if games is not None and "neutral" in games.columns:
        nmap = dict(zip(games["game_id"].astype(str), games["neutral"]))
        out["neutral"] = out["game_id"].map(nmap).fillna(False).astype(bool)
    return out


def qb_epa_values(pbp: pd.DataFrame, last_n: int = 500) -> dict:
    """{passer_player_id: shrunk EPA per dropback over his last `last_n` dropbacks}."""
    if "qb_dropback" not in pbp.columns or "passer_player_id" not in pbp.columns:
        return {}
    d = pbp[(pbp["qb_dropback"] == 1) & pbp["epa"].notna() & pbp["passer_player_id"].notna()]
    if d.empty:
        return {}
    d = d.sort_values(["season", "week"])
    out = {}
    for pid, grp in d.groupby("passer_player_id"):
        e = grp["epa"].tail(last_n)
        out[str(pid)] = float((e.sum() + QB_SHRINK_DROPBACKS * QB_REPLACEMENT_EPA) / (len(e) + QB_SHRINK_DROPBACKS))
    return out


def qb_out_points(qb_vals: dict, starter_id, backup_id, default: float) -> tuple[float, str]:
    """Points of margin lost when the starter sits. Returns (points, explanation)."""
    s = qb_vals.get(str(starter_id)) if starter_id else None
    if s is None:
        return default, "flat estimate (no play-by-play for starter)"
    b = qb_vals.get(str(backup_id), QB_REPLACEMENT_EPA) if backup_id else QB_REPLACEMENT_EPA
    pts = float(np.clip((s - b) * DROPBACKS_PER_GAME, 0.0, QB_OUT_CAP))
    return pts, f"starter {s:+.2f} vs backup {b:+.2f} EPA/dropback"


@dataclass
class BlendedRatings:
    """Quacks like TeamRatings (project / known / hfa_points / table)."""
    pts: TeamRatings
    epa: TeamRatings | None
    weight: float
    plays_per_game: float

    @property
    def uses_poss(self):
        return False

    def known(self, team) -> bool:
        return self.pts.known(team)

    def hfa_points(self) -> float:
        return self.pts.hfa_points()

    def project(self, home, away, neutral=False) -> dict:
        p = self.pts.project(home, away, neutral)
        if self.epa is None or self.weight <= 0:
            return p
        e = self.epa.project(home, away, neutral)
        epa_margin = e["margin"] / 100.0 * self.plays_per_game   # EPA/play gap x plays = points
        margin = self.weight * epa_margin + (1 - self.weight) * p["margin"]
        total = p["total"]
        return {**p, "home_pts": total / 2 + margin / 2, "away_pts": total / 2 - margin / 2,
                "margin": margin, "epa_margin": epa_margin, "score_margin": p["margin"],
                "home_known": p["home_known"] and e["home_known"], "away_known": p["away_known"] and e["away_known"]}

    def table(self) -> pd.DataFrame:
        t = self.pts.table()
        if self.epa is None or t.empty:
            return t
        t = t.rename(columns={"off": "off_pts", "def": "def_pts", "net": "net_pts"})
        t["off_epa"] = t["team"].map(lambda x: self.epa.off.get(x, np.nan) / 100)
        t["def_epa"] = t["team"].map(lambda x: self.epa.deff.get(x, np.nan) / 100)
        t["net_epa"] = t["off_epa"] - t["def_epa"]
        # blended net rating in points vs an average team
        t["net"] = self.weight * t["net_epa"] * self.plays_per_game + (1 - self.weight) * t["net_pts"]
        t["off"], t["def"] = t["off_pts"], t["def_pts"]
        return t.sort_values("net", ascending=False).reset_index(drop=True)


class NFLModel:
    """Season-chained score ratings + EPA ratings, blended."""

    def __init__(self, games: pd.DataFrame, egames: pd.DataFrame | None, cfg: SportConfig):
        self.cfg = cfg
        self.games = games
        self.pts = RatingsModel(games, cfg)
        self.epa = RatingsModel(egames, replace(cfg, uses_possessions=False)) \
            if egames is not None and not egames.empty else None
        if self.epa is not None:
            self.plays_per_game = float(pd.concat([egames["plays_home"], egames["plays_away"]]).mean())
        else:
            self.plays_per_game = 60.0

    def ratings_as_of(self, as_of, season: int) -> BlendedRatings:
        e = self.epa.ratings_as_of(as_of, season) if self.epa is not None else None
        return BlendedRatings(self.pts.ratings_as_of(as_of, season), e, self.cfg.epa_weight, self.plays_per_game)
