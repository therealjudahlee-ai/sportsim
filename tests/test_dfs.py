import numpy as np
import pandas as pd

from engine.dfs import optimize, export_csv, parse_salary_csv, SITES


def _nba_pool(seed=0, n=60):
    rng = np.random.default_rng(seed)
    pos_choices = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SF/PF", "PF/C", "C"]
    teams = ["BOS", "NYK", "LAL", "DEN", "MIA", "PHX"]
    rows = []
    for i in range(n):
        t = teams[i % 6]
        o = teams[(i % 6) ^ 1]
        proj = rng.uniform(15, 55)
        rows.append({"name": f"P{i}", "id": str(1000 + i), "positions": rng.choice(pos_choices),
                     "team": t, "opp": o, "game": "-".join(sorted([t, o])),
                     "salary": int(round((3000 + proj * 160 + rng.normal(0, 600)) / 100) * 100),
                     "proj": proj, "ceiling": proj * 1.3, "leverage": proj})
    return pd.DataFrame(rows)


def _legal(lu, sport, site):
    spec = SITES[(sport, site)]
    if lu["salary"].sum() > spec["cap"]:
        return False
    for (slot, elig), (_, r) in zip(spec["slots"], lu.iterrows()):
        if r["slot"] != slot or not (set(r["positions"]) & elig):
            return False
    return lu["name"].is_unique and len(lu) == len(spec["slots"])


def test_nba_lineups_are_legal_and_unique():
    pool = _nba_pool()
    for site in ["DK", "FD"]:
        lus = optimize(pool, "nba", site, num_lineups=5, min_unique=2, max_exposure=0.8)
        assert len(lus) == 5
        assert all(_legal(lu, "nba", site) for lu in lus)
        for i in range(len(lus)):
            for j in range(i + 1, len(lus)):
                assert len(set(lus[i]["name"]) & set(lus[j]["name"])) <= len(lus[0]) - 2
    csv = export_csv(lus, "nba", "FD").decode().splitlines()
    assert csv[0] == "PG,PG,SG,SG,SF,SF,PF,PF,C" and len(csv) == 6


def _nfl_pool(seed=1):
    rng = np.random.default_rng(seed)
    games = [("KC", "BUF"), ("PHI", "DAL"), ("SF", "SEA"), ("DET", "GB")]
    rows = []
    i = 0
    for a, b in games:
        for t, o in ((a, b), (b, a)):
            for pos, cnt in (("QB", 1), ("RB", 2), ("WR", 3), ("TE", 1), ("DST", 1)):
                for _ in range(cnt):
                    proj = rng.uniform(5, 25) if pos != "DST" else rng.uniform(4, 10)
                    rows.append({"name": f"{t}_{pos}{i}", "id": str(i), "positions": pos, "team": t, "opp": o,
                                 "game": f"{a}@{b}", "salary": int(round((3000 + proj * 220) / 100) * 100),
                                 "proj": proj, "ceiling": proj * 1.5, "leverage": proj})
                    i += 1
    return pd.DataFrame(rows)


def test_nfl_stacks_and_rules():
    pool = _nfl_pool()
    lus = optimize(pool, "nfl", "DK", objective="ceiling", num_lineups=4, stack="qb2", bringback=True,
                   no_opp_vs_dst=True, max_1_rb_per_team=True, locked=[pool.iloc[0]["name"]])
    assert len(lus) == 4
    for lu in lus:
        assert _legal(lu, "nfl", "DK")
        assert pool.iloc[0]["name"] in set(lu["name"])
        qb = lu[lu["positions"].apply(lambda p: "QB" in p)].iloc[0]
        pcs = lu[(lu["team"] == qb["team"]) & lu["positions"].apply(lambda p: bool({"WR", "TE"} & set(p)))]
        assert len(pcs) >= 2
        assert ((lu["opp"] == qb["team"]) & ~lu["positions"].apply(lambda p: "DST" in p)).any()
        dst = lu[lu["positions"].apply(lambda p: "DST" in p)].iloc[0]
        assert not ((lu["opp"] == dst["team"]) & ~lu["positions"].apply(lambda p: "DST" in p)).any()
        rb = lu[lu["positions"].apply(lambda p: "RB" in p)]
        assert rb["team"].value_counts().max() <= 1


def test_parse_dk_and_fd_csv():
    dk = pd.DataFrame({"Position": ["PG/SG", "DST"], "Name + ID": ["A (1)", "Chiefs (2)"], "Name": ["A", "Chiefs"],
                       "ID": [1, 2], "Roster Position": ["PG/G/UTIL", "DST"], "Salary": [6000, 3000],
                       "Game Info": ["KC@BUF 11/24/2026 08:20PM ET"] * 2, "TeamAbbrev": ["BUF", "KC"]})
    p = parse_salary_csv(dk)
    assert p.iloc[0]["positions"] == ["PG", "SG"] and p.iloc[0]["opp"] == "KC"
    assert p.iloc[1]["name_norm"] == "kc dst"
    fd = pd.DataFrame({"Id": ["1-1"], "Position": ["D"], "First Name": ["Kansas City"], "Nickname": ["Kansas City Chiefs"],
                       "Last Name": ["Chiefs"], "Salary": [4000], "Team": ["KC"], "Opponent": ["BUF"]})
    q = parse_salary_csv(fd)
    assert q.iloc[0]["positions"] == ["DST"] and q.iloc[0]["site"] == "FD"
