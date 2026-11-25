# SportSim

Game projections, player props and DFS for the **NFL, NBA, men's college basketball and college
football**, compared against real sportsbook prices and graded publicly.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional, but needed for the props-vs-books, parlay and live-line features: get a free key at
[the-odds-api.com](https://the-odds-api.com), then copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and paste it in. On Streamlit Cloud, put it in **Settings → Secrets**.

## First things to do

1. **Run the tests**: `python -m pytest -q` (all offline; about 15 seconds).
2. **Tune the model on past seasons** (needs internet; each sport takes a few minutes):
   ```bash
   python -m jobs.backtest nba --tune
   python -m jobs.backtest nfl --tune
   python -m jobs.backtest ncaab --tune
   python -m jobs.backtest ncaaf --tune
   ```
   This writes `engine/tuned_params.json`. Commit it, and the app uses the tuned values automatically.
   Without `--tune` it just reports how accurate the current settings are, including against the
   closing spread where historical lines exist (NFL).
3. **Turn on the track record**: push to GitHub, add `ODDS_API_KEY` under *repo Settings → Secrets → Actions*
   (optional), and the workflow in `.github/workflows/refresh.yml` will log predictions twice a day
   and grade them. The **Track record** page reads `predictions/*.csv`.

## Breaking injury news

- The ESPN injury report is re-checked every 15 minutes; anything new shows up as an alert at the top of the
  NFL/NBA pages with an **Apply** button.
- Optional: add `X_BEARER_TOKEN` to scan insiders on X (Schefter, Rapoport, Shams, etc. — edit the list in
  `engine/news.py`). X's API is pay-per-use, roughly half a cent per post read; a few insiders is ~$1–2/day.
- The sidebar's **Injury overrides** let you mark anyone OUT or ACTIVE from any source; every projection,
  prop, parlay and DFS pool on the page re-runs with it.

## Layout

```
app.py                  home page
pages/                  one page per sport + track record
ui/                     Streamlit components, cached loaders, props/parlay/DFS tabs
engine/                 pure Python, no Streamlit (can be reused by an API or another front end)
  ratings.py            opponent-adjusted ridge ratings (+ pace for basketball), preseason prior
  nfl_epa.py            NFL play-by-play (EPA) ratings blended with scores; QB injury pricing
  gamesim.py            exact win / cover / total probabilities
  props.py              prop distributions (gamma / neg. binomial / Poisson) + correlated parlays
  edges.py              model vs. no-vig market, parlay search
  odds_api.py           The Odds API client (incl. alternate-line ladders)
  news.py               X insider feed + injury-news matching
  dfs.py                slot-assignment DFS optimizer (DK/FD, NFL/NBA)
  backtest.py           walk-forward backtest + tuning
  tracking.py           prediction log, grading, calibration, CLV
  config.py             every model constant, in one place
data/                   loaders: nfl.py, nba.py, college.py, espn.py, dk.py
jobs/                   backtest.py (CLI), refresh.py (scheduled logger)
tests/                  offline tests with fake data sources
```

## Known limits (be honest with users about these)

- **Injury impacts**: NFL starting QBs are priced from measured EPA per dropback vs. their backup. Other
  positions and NBA players still use rules of thumb (`engine/config.py → INJURY`).
- **Ownership projections are rough guesses** from value and team total. Swap in a real ownership source for GPPs.
- **Football margins use a normal curve**, which ignores key numbers (3, 7). That's fine for win
  probability but slightly off for spreads that sit on a key number.
- **Data sources are unofficial** (ESPN's site API, stats.nba.com, DraftKings' lobby) or community-maintained
  (nflverse, sportsdataverse). Fine for personal use. Before charging money, move to a licensed data
  provider, and check the rules for selling betting-related tools where your users live.
