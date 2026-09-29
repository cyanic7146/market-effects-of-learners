import csv
import os

import numpy as np

from market_quality import bootstrap_ci


sweep_csv = "sweep.csv"
# sweep_csv = "sweep_thin.csv"
measures = ["excess_kurtosis", "mean_abs_gap"]
BLOC = ["bloc_flow_corr", "bloc_mean_flow", "bloc_peak_flow", "bloc_demand_pct", "bloc_longest_run"]

rows = {}
bloc = {}

with open(sweep_csv) as f:
    for row in csv.DictReader(f):
        key = (row["depth"], row["population"], int(row["copies"]))
        if key not in rows:
            rows[key] = {}
            bloc[key] = {column: [] for column in BLOC}
        rows[key][int(row["seed"])] = {m: float(row[m]) for m in measures}
        for column in BLOC:
            bloc[key][column].append(float(row[column]))

depths = sorted({k[0] for k in rows})
populations = ["momentum", "informed", "random", "rl_homogeneous", "rl_heterogeneous"]
copy_counts = sorted({k[2] for k in rows if k[2] > 0})


# same seed with and without the bloc so only the bloc is different
def paired_shift(depth, population, copies, measure):
    baseline = rows[(depth, "none", 0)]
    treated = rows[(depth, population, copies)]
    seeds = sorted(set(baseline) & set(treated))
    return np.array([treated[s][measure] - baseline[s][measure] for s in seeds])


for depth in depths:
    print(f"\n\ndepth {depth}, paired shift against the same seeds with no extra agents")
    base = rows[(depth, "none", 0)]
    for measure in measures:
        level = np.mean([v[measure] for v in base.values()])
        print(f"\n {measure} (N=0 level {level:.4f})")
        print("    arm N shift 95% ci as % of level flow corr")
        for population in populations:
            for copies in copy_counts:
                differences = paired_shift(depth, population, copies, measure)
                shift, low, high = bootstrap_ci(differences)
                if low > 0 or high < 0:
                    flat = ""
                else:
                    flat = "   flat"
                corr = np.mean(bloc[(depth, population, copies)]["bloc_flow_corr"])
                print(f"    {population} {copies} {shift:+.5f} [{low:+.5f}, {high:+.5f}] {100*shift/level:+.2f}% {corr:.3f}{flat}")


def ranking(scores):
    return " > ".join(sorted(scores, key=scores.get, reverse=True))


for depth in depths:
    print(f"\n\ndepth {depth}, what the bloc actually does")
    print("  arm N shares/step peak shares peak/mean demand % inv longest run flow corr")
    for population in populations:
        for copies in copy_counts:
            stats = bloc[(depth, population, copies)]
            mean_flow = np.mean(stats["bloc_mean_flow"])
            peak_flow = np.mean(stats["bloc_peak_flow"])
            if mean_flow > 0:
                peak_to_mean = peak_flow / mean_flow
            else:
                peak_to_mean = 0.0
            print(f"  {population} {copies} {mean_flow:.2f} {peak_flow:.1f} {peak_to_mean:.2f} {np.mean(stats['bloc_demand_pct']):.1f}% {np.mean(stats['bloc_longest_run']):.1f} {np.mean(stats['bloc_flow_corr']):.3f}")

# rank everything at the biggest N
top = max(copy_counts)
for depth in depths:
    effect = {}
    for p in populations:
        effect[p] = abs(np.mean(paired_shift(depth, p, top, "excess_kurtosis")))
    print(f"\n\ndepth {depth}, N={top} rankings, largest first")
    print(f"  kurtosis effect size {ranking(effect)}")
    for column, label in [("bloc_demand_pct", "demand % of inv"), ("bloc_longest_run", "longest run    "), ("bloc_peak_flow", "peak shares    "), ("bloc_mean_flow", "shares per step"), ("bloc_flow_corr", "flow corr      ")]:
        scores = {}
        for p in populations:
            scores[p] = np.mean(bloc[(depth, p, top)][column])
        print(f"  {label} {ranking(scores)}")
