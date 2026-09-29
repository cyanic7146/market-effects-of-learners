import csv

import numpy as np

from market_quality import bootstrap_ci


populations = ["momentum", "informed", "random", "rl_homogeneous", "rl_heterogeneous"]


def load(path):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["depth"], row["population"], int(row["copies"]))
            if key not in rows:
                rows[key] = {}
            rows[key][int(row["seed"])] = row
    return rows


# who pays for the bloc, the background agents, the maker or nobody
for path, label in [("background_returns.csv", "reference market"), ("background_returns_thin.csv", "thin market")]:
    rows = load(path)
    for depth in ["scaling", "fixed"]:
        base = rows[(depth, "none", 0)]
        level = np.mean([float(r["background_return"]) for r in base.values()])
        maker_level = np.mean([float(r["maker_pnl"]) for r in base.values()])
        print(f"\n{label}, depth {depth}: background return with no bloc {level:+.3f}%, market maker {maker_level:+.3f}%")
        print("  arm background shift, pts bloc own return maker shift, pts")
        for population in populations:
            treated = rows[(depth, population, 16)]
            seeds = sorted(set(base) & set(treated))
            background = np.array([float(treated[s]["background_return"]) - float(base[s]["background_return"]) for s in seeds])
            maker = np.array([float(treated[s]["maker_pnl"]) - float(base[s]["maker_pnl"]) for s in seeds])
            bloc = np.array([float(treated[s]["bloc_return"]) for s in seeds])
            b, blo, bhi = bootstrap_ci(background)
            m, mlo, mhi = bootstrap_ci(maker)
            r, rlo, rhi = bootstrap_ci(bloc)
            print(f"  {population} {b:+.3f} [{blo:+.3f}, {bhi:+.3f}] {r:+.2f}% [{rlo:+.2f}, {rhi:+.2f}] {m:+.3f} [{mlo:+.3f}, {mhi:+.3f}]")
