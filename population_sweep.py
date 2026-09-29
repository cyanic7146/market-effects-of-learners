import csv
import os
import random

import numpy as np

from values import VALUES
from PRICE_CALCULATIONS.cycles import OrnsteinUhlenbeckFundamental
from market import Market
from run_experiment import make_agents
from AGENTS.momentum_agent import MomentumAgent
from AGENTS.informed_agent import InformedAgent
from AGENTS.random_agent import RandomAgent
from AGENTS.rl_agent import RLAgent
from market_quality import excess_kurtosis, mean_abs_gap, returns_from, bloc_flow_correlation, bloc_flow_size, bloc_run_demand, bootstrap_ci


num_seeds = 1000
seeds = list(range(1, num_seeds + 1))
copy_counts = [1, 2, 4, 8, 16]
append = False
depth_agents = 25
output = "sweep.csv"
depth_divisor = 1.0
spread = False
VALUES["market_spread_enabled"] = spread

calm_model = "MODELS/best_model_calm_31obs_checkpoint/best_model"
calm_normalizer = "MODELS/vec_normalize_calm.pkl"


def momentum_copy(i):
    return MomentumAgent(f"Copy{i + 1}", lookback=8, trade_fraction=0.25, hard_cap=100)


def informed_copy(i):
    return InformedAgent(f"Copy{i + 1}", trade_fraction=0.25, hard_cap=100)


def random_copy(i):
    return RandomAgent(f"Copy{i + 1}", trade_fraction=0.25, hard_cap=100)


def rl_copy(i):
    return RLAgent(f"Copy{i + 1}")


def rl_mixed_copy(i):
    if i % 2 == 0:
        return RLAgent(f"Copy{i + 1}")
    return RLAgent(f"Copy{i + 1}", model_path=calm_model, normalizer_path=calm_normalizer)


populations = [
    ("momentum", momentum_copy),
    ("informed", informed_copy),
    ("random", random_copy),
    ("rl_homogeneous", rl_copy),
    ("rl_heterogeneous", rl_mixed_copy),
]


def measure(population, copy_count, fixed_depth, writer):
    if copy_count:
        extra = [populations_by_name[population](i) for i in range(copy_count)]
    else:
        extra = []
    names = []
    for agent in extra:
        names.append(agent.name)
    market = Market(make_agents() + extra, OrnsteinUhlenbeckFundamental())
    market.quiet = True

    if fixed_depth:
        market.market_starting_cash = VALUES["market_base_cash"] + VALUES["market_cash_per_agent"] * depth_agents
        market.market_starting_inventory = VALUES["market_base_inventory"] + VALUES["market_inventory_per_agent"] * depth_agents

    market.market_starting_cash = market.market_starting_cash / depth_divisor
    market.market_starting_inventory = int(market.market_starting_inventory / depth_divisor)

    kurtoses = []
    for seed in seeds:
        env_rng = random.Random(seed)
        market.rng = env_rng
        market.fundamental_process.rng = env_rng
        random.seed(seed)
        market.reset()
        for i in range(VALUES["num_steps"]):
            market.step()

        returns = returns_from(market.history)
        kurtosis = excess_kurtosis(returns)
        kurtoses.append(kurtosis)
        mean_flow, peak_flow = bloc_flow_size(market.history, names)
        demand_pct, longest_run = bloc_run_demand(market.history, names, market.market_starting_inventory)

        writer.writerow({
            "population": population,
            "copies": copy_count,
            "depth": "fixed" if fixed_depth else "scaling",
            "depth_divisor": depth_divisor,
            "spread": int(spread),
            "seed": seed,
            "excess_kurtosis": f"{kurtosis:.6f}",
            "mean_abs_gap": f"{mean_abs_gap(market.history):.6f}",
            "bloc_flow_corr": f"{bloc_flow_correlation(market.history, names):.6f}",
            "bloc_mean_flow": f"{mean_flow:.6f}",
            "bloc_peak_flow": f"{peak_flow:.6f}",
            "bloc_demand_pct": f"{demand_pct:.4f}",
            "bloc_longest_run": longest_run,
        })

    return kurtoses


populations_by_name = dict(populations)

print(f"population sweep over {len(seeds)} seeds, {VALUES['num_steps']} steps")
print(f"depth divisor {depth_divisor}, market maker spread {spread}")

fields = ["population", "copies", "depth", "depth_divisor", "spread", "seed", "excess_kurtosis", "mean_abs_gap", "bloc_flow_corr", "bloc_mean_flow", "bloc_peak_flow", "bloc_demand_pct", "bloc_longest_run"]
output_file = open(output, "a" if append else "w", newline="")
writer = csv.DictWriter(output_file, fieldnames=fields)
if not append:
    writer.writeheader()

for fixed_depth in [False, True]:
    print(f"\nfixed_depth={fixed_depth}")

    baseline = None
    if not append:
        baseline = measure("none", 0, fixed_depth, writer)
        mean, low, high = bootstrap_ci(baseline)
        print(f"  none N= 0 kurtosis={mean:.3f} 95% ci=[{low:.3f}, {high:.3f}]")

    for population, _ in populations:
        for copy_count in copy_counts:
            kurtoses = measure(population, copy_count, fixed_depth, writer)
            mean, low, high = bootstrap_ci(kurtoses)
            if baseline is None:
                print(f"  {population} N={copy_count} kurtosis={mean:.3f} 95% ci=[{low:.3f}, {high:.3f}]")
                continue
            differences = np.array(kurtoses) - np.array(baseline)
            shift, shift_low, shift_high = bootstrap_ci(differences)
            if shift_low > 0 or shift_high < 0:
                separated = "yes"
            else:
                separated = "no"
            print(f"  {population} N={copy_count} kurtosis={mean:.3f} 95% ci=[{low:.3f}, {high:.3f}] paired shift vs N=0: {shift:+.3f} [{shift_low:+.3f}, {shift_high:+.3f}] separated={separated}")

output_file.close()
print(f"\nper seed rows written to {output}")
