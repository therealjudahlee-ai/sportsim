"""Shared helpers: odds math, name normalization, time, HTTP, diagnostics."""
from __future__ import annotations

import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import requests

log = logging.getLogger("sportsim")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

ET = ZoneInfo("America/New_York")


# ----------------------------------------------------------------------------
# Time — always US/Eastern. Streamlit Cloud servers run on UTC, so a bare
# datetime.now() flips to "tomorrow" around 6-7pm, right at tip-off.
# ----------------------------------------------------------------------------
def now_et() -> datetime:
    return datetime.now(ET)


def today_et() -> datetime.date:
    return now_et().date()


# ----------------------------------------------------------------------------
# Odds math
# ----------------------------------------------------------------------------
def prob_to_american(p: float) -> str:
    p = float(np.clip(p, 0.005, 0.995))
    if p >= 0.5:
        return f"{-100.0 * p / (1.0 - p):.0f}"
    return f"+{100.0 * (1.0 - p) / p:.0f}"


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / -odds)


def american_to_prob(odds: float) -> float:
    """Implied probability INCLUDING the vig."""
    return 1.0 / american_to_decimal(odds)


def devig_pair(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Remove the vig from a two-way market (multiplicative method)."""
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    s = pa + pb
    return pa / s, pb / s


def expected_value(p: float, american_odds: float) -> float:
    """EV per $1 staked at the given price."""
    return p * american_to_decimal(american_odds) - 1.0


def fair_decimal(p: float) -> float:
    return 1.0 / max(float(p), 1e-6)


# ----------------------------------------------------------------------------
# Names
# ----------------------------------------------------------------------------
_SUFFIXES = (" jr", " sr", " ii", " iii", " iv", " v")


def norm_name(name) -> str:
    """'C.J. Stroud' == 'CJ Stroud'; 'Jaren Jackson Jr.' == 'Jaren Jackson'."""
    s = re.sub(r"[^a-z0-9\s]", "", str(name).lower())
    s = re.sub(r"\s+", " ", s).strip()
    for suf in _SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
    return s


def norm_team(name) -> str:
    s = str(name).lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\bst\b", "state", s)
    return re.sub(r"\s+", " ", s).strip()


# ----------------------------------------------------------------------------
# Diagnostics — replaces `except Exception: pass`. Every loader collects
# warnings here and the app shows them, so silent failures become visible.
# ----------------------------------------------------------------------------
class Diagnostics:
    def __init__(self):
        self.warnings: list[str] = []
        self.info: list[str] = []

    def warn(self, msg: str, exc: Exception | None = None):
        full = f"{msg}: {type(exc).__name__}: {exc}" if exc else msg
        log.warning(full)
        self.warnings.append(full)

    def note(self, msg: str):
        log.info(msg)
        self.info.append(msg)

    def to_dict(self):
        return {"warnings": list(self.warnings), "info": list(self.info)}


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
_session: requests.Session | None = None


def http() -> requests.Session:
    global _session
    if _session is None:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        s = requests.Session()
        retry = Retry(total=2, backoff_factor=0.6, status_forcelist=(429, 500, 502, 503, 504))
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.headers["User-Agent"] = "Mozilla/5.0 (sportsim)"
        _session = s
    return _session


def get_json(url: str, diag: Diagnostics | None = None, what: str = "", timeout: float = 10, **params):
    try:
        r = http().get(url, params=params or None, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001 — we log it, not swallow it
        if diag is not None:
            diag.warn(f"Fetch failed ({what or url})", e)
        return None


def recency_weights(days_ago: np.ndarray, half_life_days: float) -> np.ndarray:
    days_ago = np.clip(np.asarray(days_ago, dtype=float), 0, None)
    return 0.5 ** (days_ago / max(half_life_days, 1e-6))


def pick_col(df, *candidates, required=False):
    """Return the first column name that exists (data sources rename things)."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of {candidates} in columns: {list(df.columns)[:40]}")
    return None
