import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from market_quality import bootstrap_ci


out_dir = "FIGURES"
populations = ["momentum", "informed", "random", "rl_homogeneous", "rl_heterogeneous"]
labels = {"momentum": "momentum", "informed": "informed", "random": "random", "rl_homogeneous": "RL homogeneous", "rl_heterogeneous": "RL heterogeneous"}
colours = {"momentum": "#c1440e", "informed": "#1b6ca8", "random": "#888888", "rl_homogeneous": "#2a9d4a", "rl_heterogeneous": "#7b52ab"}

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)


def load_sweep(path):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["depth"], row["population"], int(row["copies"]))
            if key not in rows:
                rows[key] = {}
            rows[key][int(row["seed"])] = row
    return rows


def paired_shift(rows, depth, population, copies, measure):
    baseline = rows[(depth, "none", 0)]
    treated = rows[(depth, population, copies)]
    seeds = sorted(set(baseline) & set(treated))
    return np.array([float(treated[s][measure]) - float(baseline[s][measure]) for s in seeds])


def dose_response(rows, measure, label, filename, log_scale):
    depths = ["scaling", "fixed"]
    counts = sorted({k[2] for k in rows if k[2] > 0})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    for axis, depth in zip(axes, depths):
        for population in populations:
            means, lows, highs = [], [], []
            for copies in counts:
                shift, low, high = bootstrap_ci(paired_shift(rows, depth, population, copies, measure))
                means.append(shift)
                lows.append(low)
                highs.append(high)
            axis.plot(counts, means, marker="o", markersize=4, color=colours[population], label=population)
            axis.fill_between(counts, lows, highs, color=colours[population], alpha=0.15, linewidth=0)

        axis.axhline(0, color="black", linewidth=0.8, linestyle=":")
        axis.set_title(f"depth {depth}")
        axis.set_xlabel("bloc size N")
        axis.set_xscale("log", base=2)
        axis.set_xticks(counts)
        axis.set_xticklabels(counts)
        if log_scale:
            axis.set_yscale("symlog", linthresh=0.1)

    axes[0].set_ylabel(label)
    axes[0].legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(f"{out_dir}/{filename}", dpi=160)
    plt.close(figure)


def threshold(paths, filename):
    figure, axis = plt.subplots(figsize=(7, 4.4))
    markers = {"spaced": "o", "paired": "s", "run": "^"}

    for path, divisor, colour in paths:
        cells = {}
        fills = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                active = int(row["active_steps"])
                if not active or row["depth"] != "fixed":
                    continue
                quantity = round(int(row["total_volume"]) / (16 * active))
                if row["shape"] == "run":
                    run = int(row["run_length"])
                else:
                    run = 1
                key = (row["shape"], 16 * quantity * run)
                if key not in cells:
                    cells[key] = []
                    fills[key] = []
                cells[key].append(float(row["excess_kurtosis"]))
                fills[key].append(float(row["delivered_volume"]) / int(row["scheduled_volume"]))

        inventory = int((2000 + 350 * 25) / divisor)
        for (shape, cumulative), values in sorted(cells.items(), key=lambda kv: kv[0][1]):
            delivered = np.mean(fills[(shape, cumulative)]) >= 0.9
            axis.scatter(100 * cumulative / inventory, np.mean(values), marker=markers[shape],
                         facecolors=colour if delivered else "none", edgecolors=colour, s=26, alpha=0.85)
        axis.scatter([], [], color=colour, s=26, label=f"depth /{divisor:g}")
    axis.scatter([], [], facecolors="none", edgecolors="black", s=26, label="under 90% delivered")

    axis.set_xlabel("cumulative one-directional demand, % of market maker inventory")
    axis.set_title("the two depths do not collapse onto one curve", fontsize=9)
    axis.set_ylabel("excess kurtosis")
    axis.set_yscale("log")
    axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(f"{out_dir}/{filename}", dpi=160)
    plt.close(figure)


def recorded(path, filename):
    stages = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["arm"], row["stage"])
            if key not in stages:
                stages[key] = []
            stages[key].append((float(row["excess_kurtosis"]), float(row["bloc_volume"])))

    arms = ["momentum", "rl_homogeneous"]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))

    baseline = np.mean([v[0] for v in stages[("none", "fixed")]])
    shifts, errors, volumes = [], [], []
    for arm in arms:
        values = []
        for v in stages[(arm, "replayed_at_fixed")]:
            values.append(v[0])
        mean, low, high = bootstrap_ci(np.array(values) - baseline)
        shifts.append(mean)
        errors.append([mean - low, high - mean])
        volumes.append(np.mean([v[1] for v in stages[(arm, "replayed_at_fixed")]]))

    axes[0].bar(arms, shifts, yerr=np.array(errors).T, color=[colours[a] for a in arms], width=0.55, capsize=4)
    axes[0].set_ylabel("kurtosis shift vs no bloc")
    axes[0].set_title("damage from recorded flow")

    axes[1].bar(arms, volumes, color=[colours[a] for a in arms], width=0.55)
    axes[1].set_ylabel("bloc volume, shares")
    axes[1].set_title("volume traded")

    for axis in axes:
        axis.tick_params(axis="x", labelsize=8)
    figure.tight_layout()
    figure.savefig(f"{out_dir}/{filename}", dpi=160)
    plt.close(figure)


def workflow(filename):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    figure, axis = plt.subplots(figsize=(11, 5.2))
    axis.set_xlim(0, 11); axis.set_ylim(0, 5.2); axis.axis("off")

    def box(x, y, w, h, text, colour):
        axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                      facecolor=colour, edgecolor="#555555", linewidth=0.9))
        axis.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        axis.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                       color="#555555", linewidth=0.9))

    top = 3.6
    box(0.2, top, 2.3, 1.3, "market of [1]\n24 background agents,\nmarket maker counterparty,\nhidden OU fundamental", "#e8e4dc")
    box(2.9, top, 2.3, 1.3, "add one bloc\nmomentum, informed, random,\nRL homogeneous, RL heterogeneous\nN = 1, 2, 4, 8, 16", "#dbe7f3")
    box(5.6, top, 2.3, 1.3, "four conditions\nreference or thin market\n× depth scaling or fixed\n1,000 paired seeds per cell", "#dbe7f3")
    box(8.3, top, 2.5, 1.3, "outcomes per episode\nmispricing, excess kurtosis,\nbackground and bloc returns,\nfive bloc flow statistics", "#e3efdd")
    arrow(2.5, top + 0.65, 2.9, top + 0.65); arrow(5.2, top + 0.65, 5.6, top + 0.65); arrow(7.9, top + 0.65, 8.3, top + 0.65)

    mid = 1.85
    box(0.2, mid, 3.5, 1.15, "scripted flow\n16 agents on one fixed schedule, correlation pinned at 1\nvolume, concentration and run length varied", "#f3e6d8")
    box(4.0, mid, 3.3, 1.15, "recorded flow\neach copy's own orders recorded at scaling depth,\nreplayed unchanged at fixed depth", "#f3e6d8")
    box(7.6, mid, 3.2, 1.15, "robustness\nintra-step impact and spread on,\ndepth ÷2, ÷4, ÷6", "#ece8f3")
    arrow(3.7, mid + 0.575, 4.0, mid + 0.575)
    arrow(9.55, top, 9.2, mid + 1.15)
    arrow(1.9, top, 1.9, mid + 1.15)

    axis.text(0.2, 0.95, "every number is a mean per-seed shift against the same seeds with no bloc, with a 95% bootstrap interval",
              fontsize=8.6, color="#333333")
    axis.text(0.2, 0.55, "top row: the population sweep. bottom row: the mechanism and robustness experiments, all in the thin market at fixed depth",
              fontsize=8.6, color="#333333")
    figure.tight_layout()
    figure.savefig(f"{out_dir}/{filename}", dpi=170)
    plt.close(figure)


workflow("workflow.png")

sweep = load_sweep("sweep.csv")
dose_response(sweep, "excess_kurtosis", "kurtosis shift vs N=0", "kurtosis_normal.png", False)

thin = load_sweep("sweep_thin.csv")
dose_response(thin, "excess_kurtosis", "kurtosis shift vs N=0", "kurtosis_thin.png", True)

threshold([("flow_replay_d2.csv", 2, "#1b6ca8"), ("flow_replay.csv", 4, "#c1440e")], "threshold.png")
recorded("recorded_replay.csv", "recorded_replay.png")

print(f"figures written to {out_dir}/")


def two_markets(deep_rows, thin_rows, filename):
    counts = sorted({k[2] for k in deep_rows if k[2] > 0})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, rows, depth, title, log_scale in [(axes[0], deep_rows, "scaling", "deep market", False), (axes[1], thin_rows, "fixed", "thin market", True)]:
        for population in populations:
            means, lows, highs = [], [], []
            for copies in counts:
                shift, low, high = bootstrap_ci(paired_shift(rows, depth, population, copies, "excess_kurtosis"))
                means.append(shift)
                lows.append(low)
                highs.append(high)
            axis.plot(counts, means, marker="o", markersize=4, color=colours[population], label=labels[population])
            axis.fill_between(counts, lows, highs, color=colours[population], alpha=0.15, linewidth=0)
        axis.axhline(0, color="black", linewidth=0.8, linestyle=":")
        axis.set_title(title)
        axis.set_xlabel("bloc size N")
        axis.set_xscale("log", base=2)
        axis.set_xticks(counts)
        axis.set_xticklabels(counts)
        if log_scale:
            axis.set_yscale("symlog", linthresh=0.1)
    axes[0].set_ylabel("excess kurtosis shift vs no bloc")
    axes[0].legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(f"{out_dir}/{filename}", dpi=160)
    plt.close(figure)


two_markets(sweep, thin, "two_markets.png")
print("two_markets.png written")


def example_episode(path, filename):
    rows = list(csv.DictReader(open(path)))
    steps = []
    for r in rows:
        steps.append(int(r["step"]))
    figure, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    axes[0].plot(steps, [float(r["fundamental"]) for r in rows], color="black", linewidth=1, linestyle="--", label="fundamental")
    axes[0].plot(steps, [float(r["price_no_bloc"]) for r in rows], color="#888888", linewidth=1, label="price, no bloc")
    axes[0].plot(steps, [float(r["price_rl_bloc"]) for r in rows], color=colours["rl_homogeneous"], linewidth=1.2, label="price, 16 RL copies")
    axes[0].set_ylabel("price")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].plot(steps, [int(r["inventory_no_bloc"]) for r in rows], color="#888888", linewidth=1, label="no bloc")
    axes[1].plot(steps, [int(r["inventory_rl_bloc"]) for r in rows], color=colours["rl_homogeneous"], linewidth=1.2, label="16 RL copies")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle=":")
    axes[1].set_ylabel("market maker inventory, shares")
    axes[1].set_xlabel("step")
    axes[1].legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(f"{out_dir}/{filename}", dpi=160)
    plt.close(figure)


example_episode("example_episode.csv", "example_episode.png")
print("example_episode.png written")
