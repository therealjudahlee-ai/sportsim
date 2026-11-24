"""Player prop projections.

Old approach: Normal(volume) x Normal(efficiency), 1M draws per prop. Problems:
symmetric (props are right-skewed), can't represent counts, slow.

New approach: project the MEAN (volume x efficiency x matchup x game script),
estimate the SPREAD from the player's own history shrunk toward a position
default, then use the right distribution family:
  * yards / points / combos -> Gamma (right-skewed, non-negative)
  * receptions, rebounds, assists, threes, steals, blocks -> Negative binomial
  * touchdowns -> Poisson
P(over) comes straight from the CDF: exact and instant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as st


# ----------------------------------------------------------------------------
# Distributions
# ----------------------------------------------------------------------------
@dataclass
class StatDist:
    family: str     # "gamma" | "nbinom" | "poisson"
    mean: float
    var: float

    def _frozen(self):
        m = max(self.mean, 1e-6)
        if self.family == "gamma":
            v = max(self.var, 1e-6)
            return st.gamma(a=m * m / v, scale=v / m)
        if self.family == "nbinom" and self.var > m * 1.02:
            p = m / self.var
            n = m * p / (1 - p)
            return st.nbinom(n, p)
        return st.poisson(m)

    @property
    def discrete(self) -> bool:
        return self.family in ("nbinom", "poisson")

    def prob_over(self, line: float) -> float:
        d = self._frozen()
        if self.discrete:
            return float(d.sf(np.floor(line)))          # P(X >= floor(line)+1)
        return float(d.sf(line))

    def prob_under(self, line: float) -> float:
        d = self._frozen()
        if self.discrete:
            return float(d.cdf(np.ceil(line) - 1))      # P(X <= ceil(line)-1)
        return float(d.cdf(line))

    def quantile(self, q: float) -> float:
        return float(self._frozen().ppf(q))

    def sample(self, n: int = 20_000, seed: int | None = None) -> np.ndarray:
        return self._frozen().rvs(size=n, random_state=seed)


def _shrink(x, n, prior, k):
    """Bayesian-style shrink of a sample estimate x (from n obs) toward prior."""
    if x is None or not np.isfinite(x):
        return prior
    return (n * x + k * prior) / (n + k)


def _ewm_last(s: pd.Series, span: float) -> float:
    s = s.dropna()
    return float(s.ewm(span=span).mean().iloc[-1]) if len(s) else np.nan


def shrink_matchup(raw_mult: float, n_games: int, k: float = 8.0) -> float:
    """Opponent multipliers from ~10 games are mostly noise — pull toward 1.0."""
    if raw_mult is None or not np.isfinite(raw_mult):
        return 1.0
    return float(1.0 + (raw_mult - 1.0) * n_games / (n_games + k))


# ----------------------------------------------------------------------------
# NFL
# ----------------------------------------------------------------------------
NFL_VOLUME = {
    "passing_yards": ("attempts", 10, 150.0),   # (volume col, min volume for a "healthy" game, shrink K)
    "rushing_yards": ("carries", 4, 60.0),
    "receiving_yards": ("targets", 2, 30.0),
    "receptions": ("targets", 2, 30.0),
}
NFL_CV = {"passing_yards": 0.28, "rushing_yards": 0.55, "receiving_yards": 0.65,
          "dk_points": 0.55, "fd_points": 0.55, "passing_tds": 0.75}
NFL_COUNT = {"receptions": 1.15, "passing_tds": 1.0, "rushing_tds": 1.0, "receiving_tds": 1.0}
NFL_PROP_STATS = ["passing_yards", "rushing_yards", "receiving_yards", "receptions",
                  "anytime_td", "passing_tds", "dk_points", "fd_points"]


def nfl_env_factors(stat: str, team_proj_pts: float | None, team_margin: float | None,
                    league_team_pts: float, wind_outdoors: float = 0.0) -> tuple[float, float, list]:
    """Returns (volume multiplier, efficiency multiplier, notes)."""
    vol, eff, notes = 1.0, 1.0, []
    if team_proj_pts is not None and np.isfinite(team_proj_pts):
        diff = team_proj_pts - league_team_pts
        if stat in ("anytime_td", "passing_tds", "rushing_tds", "receiving_tds"):
            eff *= 1.0 + 0.03 * diff
        else:
            eff *= 1.0 + 0.008 * diff
    if team_margin is not None and np.isfinite(team_margin):
        if stat in ("passing_yards", "receiving_yards", "receptions"):
            vol *= 1.0 - 0.004 * team_margin      # trailing teams throw more
        elif stat == "rushing_yards":
            vol *= 1.0 + 0.006 * team_margin      # leading teams run more
    if wind_outdoors >= 15 and stat in ("passing_yards", "receiving_yards", "receptions", "passing_tds"):
        eff *= 0.93
        notes.append(f"wind {wind_outdoors:.0f} mph")
    return float(np.clip(vol, 0.85, 1.15)), float(np.clip(eff, 0.75, 1.3)), notes


def project_nfl(logs: pd.DataFrame, stat: str, position: str, league: dict,
                def_mult: float = 1.0, env: tuple = (1.0, 1.0, []), bump: float = 1.0) -> dict | None:
    """logs: this player's game logs sorted oldest -> newest.
    league: position baselines from the context (eff per volume, TD rates)."""
    recent = logs.tail(16)
    if recent.empty or stat not in recent.columns and stat != "anytime_td":
        return None
    vol_m, eff_m, notes = env
    notes = list(notes)

    if stat == "anytime_td":
        td = recent["anytime_td"].fillna(0)
        rate = 0.6 * _ewm_last(td, 6) + 0.4 * td.mean()
        base = league.get(("td_rate", position), 0.25)
        lam = _shrink(rate, len(td), base, 6.0) * def_mult * eff_m * bump
        lam = float(np.clip(lam, 0.02, 2.0))
        dist = StatDist("poisson", lam, lam)
        return _pack(dist, len(td), notes)

    if stat in NFL_VOLUME:
        vcol, vmin, k = NFL_VOLUME[stat]
        healthy = recent[recent[vcol].fillna(0) >= vmin].tail(12)
        if len(healthy) < 3:
            return None
        vol = _ewm_last(healthy[vcol], 5)
        num_col = "receptions" if stat == "receptions" else stat
        w = np.exp(np.linspace(-1.0, 0.0, len(healthy)))
        tot_stat = float(np.sum(w * healthy[num_col].fillna(0)))
        tot_vol = float(np.sum(w * healthy[vcol].fillna(0)))
        lg_eff = league.get(("eff", position, stat), tot_stat / max(tot_vol, 1))
        eff = (tot_stat + k * lg_eff) / (tot_vol + k)
        mean = vol * vol_m * eff * eff_m * def_mult * bump
        n = len(healthy)
        hist = healthy[num_col].fillna(0)
    else:
        hist = recent[stat].fillna(0).tail(12)
        if len(hist) < 3:
            return None
        mean = _ewm_last(hist, 5) * vol_m * eff_m * def_mult * bump
        n = len(hist)

    mean = max(float(mean), 0.05)
    if stat in NFL_COUNT:
        disp_hist = hist.var() / max(hist.mean(), 1e-6) if n >= 3 else np.nan
        disp = max(_shrink(disp_hist, n, NFL_COUNT[stat], 8.0), 1.0)
        dist = StatDist("nbinom", mean, mean * disp)
    else:
        cv_hist = hist.std() / max(hist.mean(), 1e-6) if n >= 3 else np.nan
        cv = _shrink(cv_hist, n, NFL_CV.get(stat, 0.5), 8.0)
        dist = StatDist("gamma", mean, (cv * mean) ** 2)
    return _pack(dist, n, notes)


# ----------------------------------------------------------------------------
# NBA (and works for any minutes-based sport)
# ----------------------------------------------------------------------------
NBA_PROP_STATS = ["PTS", "REB", "AST", "PRA", "PR", "PA", "RA", "FG3M", "STL", "BLK", "DK_PTS"]
NBA_CV = {"PTS": 0.36, "PRA": 0.28, "PR": 0.31, "PA": 0.32, "RA": 0.36, "DK_PTS": 0.30}
NBA_COUNT = {"REB": 1.25, "AST": 1.35, "FG3M": 1.3, "STL": 1.1, "BLK": 1.45}


def project_nba(logs: pd.DataFrame, stat: str, league_rate: dict, def_mult: float = 1.0,
                pace_mult: float = 1.0, fatigue_mult: float = 1.0, minutes_mult: float = 1.0,
                bump: float = 1.0, home: bool | None = None) -> dict | None:
    recent = logs.tail(25)
    healthy = recent[recent["MIN"] >= 10].tail(15)
    if len(healthy) < 4 or stat not in healthy.columns:
        return None
    mins = _ewm_last(healthy["MIN"], 5) * minutes_mult
    w = np.exp(np.linspace(-1.2, 0.0, len(healthy)))
    tot = float(np.sum(w * healthy[stat]))
    tmin = float(np.sum(w * healthy["MIN"]))
    lg = league_rate.get(stat, tot / max(tmin, 1))
    rate = (tot + 120.0 * lg) / (tmin + 120.0)   # shrink per-minute rate with ~120 min of league avg
    notes = []
    if home is not None and "IS_HOME" in healthy.columns:
        split = healthy[healthy["IS_HOME"] == home]
        if len(split) >= 5:
            srate = split[stat].sum() / max(split["MIN"].sum(), 1)
            rate = 0.85 * rate + 0.15 * srate
    mean = max(mins * rate * def_mult * pace_mult * fatigue_mult * bump, 0.05)
    n = len(healthy)
    hist = healthy[stat]
    if stat in NBA_COUNT:
        disp_hist = hist.var() / max(hist.mean(), 1e-6)
        disp = max(_shrink(disp_hist, n, NBA_COUNT[stat], 10.0), 1.0)
        dist = StatDist("nbinom", mean, mean * disp)
    else:
        cv_hist = hist.std() / max(hist.mean(), 1e-6)
        cv = _shrink(cv_hist, n, NBA_CV.get(stat, 0.35), 10.0)
        dist = StatDist("gamma", mean, (cv * mean) ** 2)
    out = _pack(dist, n, notes)
    out["minutes"] = mins
    return out


def _pack(dist: StatDist, n: int, notes: list) -> dict:
    return {
        "dist": dist,
        "mean": dist.mean,
        "median": dist.quantile(0.5),
        "p20": dist.quantile(0.2),
        "p80": dist.quantile(0.8),
        "games_used": int(n),
        "notes": notes,
    }


# ----------------------------------------------------------------------------
# Correlated parlays (Gaussian copula)
# ----------------------------------------------------------------------------
_PASSY = {"passing_yards", "passing_tds"}
_RECV = {"receiving_yards", "receptions"}


def leg_corr(a: dict, b: dict, sport: str) -> float:
    """Rough correlations between two prop outcomes (on the OVER side)."""
    if a["player"] == b["player"]:
        c = 0.55
    elif a.get("game") != b.get("game"):
        c = 0.0
    elif a.get("team") == b.get("team"):
        if sport == "nfl":
            sa, sb = a["stat"], b["stat"]
            if (sa in _PASSY and sb in _RECV) or (sb in _PASSY and sa in _RECV):
                c = 0.35
            elif sa in _RECV and sb in _RECV:
                c = -0.10
            elif {sa, sb} & {"rushing_yards"} and {sa, sb} & _PASSY:
                c = -0.08
            elif sa == "anytime_td" or sb == "anytime_td":
                c = 0.05
            else:
                c = 0.0
        else:
            c = -0.06 if a["stat"] == b["stat"] == "PTS" else 0.0
    else:  # same game, opposing teams: shootouts lift both
        c = 0.10 if sport == "nfl" else 0.05
    flip = (a.get("side", "over") == "over") != (b.get("side", "over") == "over")
    return -c if flip else c


def _nearest_psd(R):
    vals, vecs = np.linalg.eigh(R)
    R2 = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
    d = np.sqrt(np.diag(R2))
    return R2 / np.outer(d, d)


def parlay_prob(legs: list[dict], sport: str, n: int = 200_000, seed: int = 7) -> float:
    """Joint probability that every leg hits. Each leg needs 'prob' (its own
    hit probability) plus player/team/game/stat/side for correlations."""
    k = len(legs)
    if k == 0:
        return 0.0
    if k == 1:
        return float(legs[0]["prob"])
    R = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            R[i, j] = R[j, i] = leg_corr(legs[i], legs[j], sport)
    if not np.allclose(R, np.eye(k)):
        R = _nearest_psd(R)
    else:
        return float(np.prod([l["prob"] for l in legs]))
    z = np.random.default_rng(seed).multivariate_normal(np.zeros(k), R, size=n, method="cholesky")
    thresh = st.norm.ppf([l["prob"] for l in legs])
    return float(np.mean(np.all(z < thresh, axis=1)))


# ----------------------------------------------------------------------------
# Which props actually exist? Books only hang a line when a player gets real
# volume in that stat (nobody posts Jared Goff over 1.5 rushing yards). These
# are the smallest projected MEANS at which lines are commonly offered.
# Stats not listed (DK/FD fantasy points) are always allowed.
# ----------------------------------------------------------------------------
MARKET_MIN = {
    "nfl": {
        "passing_yards": 150.0,   # starting QBs only
        "passing_tds": 0.8,
        "rushing_yards": 12.0,    # RBs + genuinely mobile QBs (Allen, Hurts, Jackson…)
        "receiving_yards": 12.0,
        "receptions": 1.8,
        "anytime_td": 0.08,       # expected TDs (~8% chance) — long shots still get priced
    },
    "nba": {
        "PTS": 7.0, "REB": 3.0, "AST": 2.0, "FG3M": 0.8, "PRA": 12.0, "PR": 10.0, "PA": 9.0,
        "RA": 5.0, "STL": 0.7, "BLK": 0.6,
    },
}
# position gates on top of the volume floors
NFL_POS_MARKETS = {
    "QB": {"passing_yards", "passing_tds", "rushing_yards", "anytime_td"},
    "RB": {"rushing_yards", "receiving_yards", "receptions", "anytime_td"},
    "WR": {"receiving_yards", "receptions", "anytime_td", "rushing_yards"},
    "TE": {"receiving_yards", "receptions", "anytime_td"},
}


def market_offered(sport: str, stat: str, proj: dict | None, position: str | None = None) -> bool:
    if proj is None:
        return False
    if sport == "nfl" and position in NFL_POS_MARKETS and stat in MARKET_MIN["nfl"] \
            and stat not in NFL_POS_MARKETS[position]:
        return False
    floor = MARKET_MIN.get(sport, {}).get(stat)
    return floor is None or proj["mean"] >= floor
