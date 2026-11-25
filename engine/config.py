"""Per-sport model constants.

Every number here is a starting default. Run `python -m jobs.backtest <sport>`
to fit the tunable ones on past seasons; results are written to
engine/tuned_params.json and automatically override these defaults.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

TUNED_PATH = Path(__file__).with_name("tuned_params.json")


@dataclass(frozen=True)
class SportConfig:
    key: str
    label: str
    # --- ratings model (tuned by backtest) ---
    half_life_days: float      # recency weighting of past games
    ridge_alpha: float         # shrinkage toward the prior
    prior_regress: float       # fraction of last season's rating carried into this season
    margin_sigma: float        # SD of actual margin around projected margin
    total_sigma: float         # SD of actual total around projected total
    # --- fixed / heuristic ---
    default_hfa: float         # used only if HFA can't be fit from data
    blowout_margin: float      # |margin| counted as a blowout
    min_team_games: int        # teams with fewer games are pooled as "other"
    uses_possessions: bool     # basketball: fit efficiency + tempo separately
    rest_penalty_b2b: float = 0.0   # points lost on 0 days rest (basketball)
    odds_api_key: str = ""     # The Odds API sport key
    league_avg_total: float = 0.0
    epa_weight: float = 0.0    # NFL only: share of the margin from play-by-play efficiency ratings


DEFAULTS: dict[str, SportConfig] = {
    "nfl": SportConfig(
        key="nfl", label="NFL",
        half_life_days=120, ridge_alpha=6.0, prior_regress=0.55,
        margin_sigma=13.5, total_sigma=13.5,
        default_hfa=1.8, blowout_margin=14, min_team_games=1,
        uses_possessions=False, odds_api_key="americanfootball_nfl",
        league_avg_total=45.0, epa_weight=0.6,
    ),
    "nba": SportConfig(
        key="nba", label="NBA",
        half_life_days=45, ridge_alpha=8.0, prior_regress=0.70,
        margin_sigma=12.0, total_sigma=18.0,
        default_hfa=2.3, blowout_margin=15, min_team_games=1,
        uses_possessions=True, rest_penalty_b2b=1.5,
        odds_api_key="basketball_nba", league_avg_total=228.0,
    ),
    "ncaab": SportConfig(
        key="ncaab", label="Men's College Basketball",
        half_life_days=40, ridge_alpha=5.0, prior_regress=0.60,
        margin_sigma=11.0, total_sigma=15.0,
        default_hfa=3.2, blowout_margin=15, min_team_games=4,
        uses_possessions=True, rest_penalty_b2b=1.0,
        odds_api_key="basketball_ncaab", league_avg_total=145.0,
    ),
    "ncaaf": SportConfig(
        key="ncaaf", label="College Football",
        half_life_days=90, ridge_alpha=4.0, prior_regress=0.60,
        margin_sigma=15.5, total_sigma=17.0,
        default_hfa=2.8, blowout_margin=17, min_team_games=3,
        uses_possessions=False, odds_api_key="americanfootball_ncaaf",
        league_avg_total=54.0,
    ),
}

TUNABLE = ("half_life_days", "ridge_alpha", "prior_regress", "margin_sigma", "total_sigma", "epa_weight")


def get_config(sport: str) -> SportConfig:
    """Defaults merged with any backtest-tuned values."""
    cfg = DEFAULTS[sport]
    if TUNED_PATH.exists():
        try:
            tuned = json.loads(TUNED_PATH.read_text()).get(sport, {})
            cfg = replace(cfg, **{k: float(v) for k, v in tuned.items() if k in TUNABLE})
        except Exception:  # a bad tuned file should never break the app
            pass
    return cfg


def save_tuned(sport: str, params: dict) -> None:
    data = json.loads(TUNED_PATH.read_text()) if TUNED_PATH.exists() else {}
    data[sport] = {k: params[k] for k in TUNABLE if k in params}
    TUNED_PATH.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Heuristic adjustments (not fit by the backtest — no historical injury data).
# Kept in one place so they're easy to find and tweak.
# ---------------------------------------------------------------------------
INJURY = {
    # NFL: points of margin lost when the starting QB is out
    "nfl_qb1_out": 4.5,
    # NFL: per other key skill player out (>= 12 DK pts/game), capped
    "nfl_skill_out": 0.6,
    "nfl_skill_cap": 2.0,
    # NBA/NCAAB: points lost per DK point/game above replacement, capped per player
    "nba_per_dkpt_over_repl": 0.15,
    "nba_replacement_dkpts": 18.0,
    "nba_player_cap": 5.0,
    # how much an injury designation counts
    "status_weight": {"out": 1.0, "doubtful": 0.75, "questionable": 0.3, "day-to-day": 0.3},
}

# Wind above this (mph, outdoors) trims projected NFL totals
NFL_WIND_MPH = 15.0
NFL_WIND_TOTAL_MULT = 0.95
