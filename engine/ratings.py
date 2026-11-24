"""Opponent-adjusted team ratings via recency-weighted ridge regression.

Points model (football):        pts = mu + OFF[team] + DEF[opp] + HFA * h
Efficiency model (basketball):  pts/100 poss = same form,  and
                                poss = mu_p + PACE[home] + PACE[away]

h = +0.5 for the home team, -0.5 for the away team, 0 at neutral sites,
so HFA is the full home edge in margin.

Early season: instead of shrinking team ratings toward 0 (average), we shrink
them toward last season's final ratings * prior_regress. With few games the
prior dominates; as games pile up the data takes over. This replaces the old
"tail(22) games across two seasons" approach, which mostly used last year's
rosters in November.

Games DataFrame schema (produced by every data loader):
    game_id, date (datetime64), season (int), home, away (team keys),
    home_pts, away_pts, neutral (bool), [poss]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .common import recency_weights
from .config import SportConfig

OTHER = "__other__"


def _ridge_solve(idx, val, y, w, n_feat, alpha, prior):
    """Weighted ridge with a non-zero prior, built from sparse rows.

    idx/val: (m, k) arrays — each row has k non-zero features.
    Minimizes sum w (y - Xb)^2 + sum alpha (b - prior)^2.
    """
    A = np.zeros((n_feat, n_feat))
    rhs = np.zeros(n_feat)
    k = idx.shape[1]
    for a in range(k):
        wa = w * val[:, a]
        np.add.at(rhs, idx[:, a], wa * y)
        for b in range(k):
            np.add.at(A, (idx[:, a], idx[:, b]), wa * val[:, b])
    A[np.diag_indices(n_feat)] += alpha
    rhs += alpha * prior
    return np.linalg.solve(A, rhs)


@dataclass
class TeamRatings:
    mu: float
    hfa: float                        # in model units (pts, or pts/100 for basketball)
    off: dict = field(default_factory=dict)
    deff: dict = field(default_factory=dict)
    pace_mu: float = 0.0
    pace: dict = field(default_factory=dict)
    uses_poss: bool = False
    n_games: dict = field(default_factory=dict)
    as_of: pd.Timestamp | None = None
    season: int | None = None

    def known(self, team) -> bool:
        return team in self.off

    def _get(self, d, team):
        if team in d:
            return d[team]
        return d.get(OTHER, 0.0)

    def project(self, home, away, neutral=False) -> dict:
        h = 0.0 if neutral else 0.5
        e_home = self.mu + self._get(self.off, home) + self._get(self.deff, away) + self.hfa * h
        e_away = self.mu + self._get(self.off, away) + self._get(self.deff, home) - self.hfa * h
        if self.uses_poss:
            poss = self.pace_mu + self._get(self.pace, home) + self._get(self.pace, away)
            hp, ap = e_home * poss / 100.0, e_away * poss / 100.0
        else:
            poss = np.nan
            hp, ap = e_home, e_away
        return {
            "home_pts": float(hp), "away_pts": float(ap),
            "margin": float(hp - ap), "total": float(hp + ap), "poss": float(poss),
            "home_known": self.known(home), "away_known": self.known(away),
        }

    def hfa_points(self) -> float:
        """Home edge in points of margin."""
        if self.uses_poss:
            return self.hfa * self.pace_mu / 100.0
        return self.hfa

    def table(self) -> pd.DataFrame:
        rows = []
        for t in self.off:
            if t == OTHER:
                continue
            rows.append({
                "team": t,
                "off": self.off[t],
                "def": self.deff[t],
                "net": self.off[t] - self.deff[t],
                "pace": self.pace_mu + self.pace.get(t, 0.0) if self.uses_poss else np.nan,
                "games": self.n_games.get(t, 0),
            })
        df = pd.DataFrame(rows)
        return df.sort_values("net", ascending=False).reset_index(drop=True) if not df.empty else df


def fit_ratings(games: pd.DataFrame, as_of, cfg: SportConfig, prior: TeamRatings | None = None,
                season: int | None = None) -> TeamRatings:
    """Fit ratings on games strictly before `as_of` (and in `season`, if given)."""
    as_of = pd.Timestamp(as_of)
    g = games[(games["date"] < as_of)].dropna(subset=["home_pts", "away_pts"])
    if season is not None:
        g = g[g["season"] == season]
    if cfg.uses_possessions:
        g = g[g["poss"].notna() & (g["poss"] > 0)] if "poss" in g.columns else g.iloc[0:0]

    if g.empty:
        if prior is not None:
            return _carry_prior(prior, cfg, as_of, season)
        return TeamRatings(mu=0.0, hfa=cfg.default_hfa, uses_poss=cfg.uses_possessions, as_of=as_of, season=season)

    # Pool rarely-seen teams (e.g. non-D1 opponents) so ridge doesn't call them average
    counts = pd.concat([g["home"], g["away"]]).value_counts()
    keep = set(counts[counts >= cfg.min_team_games].index)
    home = g["home"].where(g["home"].isin(keep), OTHER).to_numpy()
    away = g["away"].where(g["away"].isin(keep), OTHER).to_numpy()
    teams = sorted(set(home) | set(away))
    ti = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    w_game = recency_weights((as_of - g["date"]).dt.days.to_numpy(), cfg.half_life_days)
    neutral = g["neutral"].fillna(False).astype(bool).to_numpy()
    hcode = np.where(neutral, 0.0, 0.5)

    hp = g["home_pts"].to_numpy(float)
    ap = g["away_pts"].to_numpy(float)
    if cfg.uses_possessions:
        poss = g["poss"].to_numpy(float)
        y_home, y_away = hp / poss * 100.0, ap / poss * 100.0
    else:
        poss = None
        y_home, y_away = hp, ap

    hi = np.array([ti[t] for t in home])
    ai = np.array([ti[t] for t in away])
    m = len(g)
    # features: 0 = intercept, 1 = hfa, 2..2+n = off, 2+n..2+2n = def
    n_feat = 2 + 2 * n
    idx = np.empty((2 * m, 4), dtype=int)
    val = np.ones((2 * m, 4))
    idx[:m] = np.column_stack([np.zeros(m, int), np.ones(m, int), 2 + hi, 2 + n + ai])
    idx[m:] = np.column_stack([np.zeros(m, int), np.ones(m, int), 2 + ai, 2 + n + hi])
    val[:m, 1] = hcode
    val[m:, 1] = -hcode
    y = np.concatenate([y_home, y_away])
    w = np.concatenate([w_game, w_game])

    league_mean = float(np.average(y, weights=w))
    avg_poss = float(np.average(poss, weights=w_game)) if poss is not None else None
    hfa_prior = cfg.default_hfa * (100.0 / avg_poss if avg_poss else 1.0)

    alpha = np.full(n_feat, cfg.ridge_alpha)
    alpha[0] = 1e-6
    alpha[1] = 3.0  # HFA: weak pull toward the default, mostly data-driven
    b0 = np.zeros(n_feat)
    b0[0] = league_mean
    b0[1] = hfa_prior
    if prior is not None:
        for t, i in ti.items():
            b0[2 + i] = cfg.prior_regress * prior.off.get(t, prior.off.get(OTHER, 0.0))
            b0[2 + n + i] = cfg.prior_regress * prior.deff.get(t, prior.deff.get(OTHER, 0.0))

    beta = _ridge_solve(idx, val, y, w, n_feat, alpha, b0)
    off = beta[2:2 + n]
    deff = beta[2 + n:]
    mu = beta[0] + off.mean() + deff.mean()
    off = off - off.mean()
    deff = deff - deff.mean()

    out = TeamRatings(
        mu=float(mu), hfa=float(beta[1]),
        off={t: float(off[i]) for t, i in ti.items()},
        deff={t: float(deff[i]) for t, i in ti.items()},
        uses_poss=cfg.uses_possessions,
        n_games={t: int(counts.get(t, 0)) for t in teams},
        as_of=as_of, season=season,
    )

    if cfg.uses_possessions:
        tidx = np.column_stack([np.zeros(m, int), 1 + hi, 1 + ai])
        tval = np.ones((m, 3))
        p_alpha = np.full(1 + n, cfg.ridge_alpha)
        p_alpha[0] = 1e-6
        p0 = np.zeros(1 + n)
        p0[0] = float(np.average(poss, weights=w_game))
        if prior is not None and prior.pace:
            for t, i in ti.items():
                p0[1 + i] = cfg.prior_regress * prior.pace.get(t, 0.0)
        pb = _ridge_solve(tidx, tval, poss, w_game, 1 + n, p_alpha, p0)
        pace = pb[1:]
        out.pace_mu = float(pb[0] + 2 * pace.mean())
        pace = pace - pace.mean()
        out.pace = {t: float(pace[i]) for t, i in ti.items()}
    return out


def _carry_prior(prior: TeamRatings, cfg: SportConfig, as_of, season) -> TeamRatings:
    r = cfg.prior_regress
    return TeamRatings(
        mu=prior.mu, hfa=prior.hfa,
        off={t: r * v for t, v in prior.off.items()},
        deff={t: r * v for t, v in prior.deff.items()},
        pace_mu=prior.pace_mu, pace={t: r * v for t, v in prior.pace.items()},
        uses_poss=prior.uses_poss, n_games={}, as_of=pd.Timestamp(as_of), season=season,
    )


class RatingsModel:
    """Season-chained ratings: last season (fit on its full data) is the prior
    for this season. Caches the prior fits so walk-forward backtests are fast."""

    def __init__(self, games: pd.DataFrame, cfg: SportConfig):
        self.games = games.sort_values("date").reset_index(drop=True)
        self.cfg = cfg
        self._season_end: dict[int, TeamRatings] = {}

    def _season_final(self, season: int) -> TeamRatings | None:
        if season in self._season_end:
            return self._season_end[season]
        sg = self.games[self.games["season"] == season]
        if sg.empty:
            self._season_end[season] = None
            return None
        prior = self._season_final(season - 1) if (season - 1) in set(self.games["season"]) else None
        end = sg["date"].max() + pd.Timedelta(days=1)
        fit = fit_ratings(self.games, end, self.cfg, prior=prior, season=season)
        self._season_end[season] = fit
        return fit

    def ratings_as_of(self, as_of, season: int) -> TeamRatings:
        prior = self._season_final(season - 1)
        return fit_ratings(self.games, as_of, self.cfg, prior=prior, season=season)
