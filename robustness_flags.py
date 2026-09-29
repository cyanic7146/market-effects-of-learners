import csv
import os
import random

import numpy as np

from values import VALUES
from PRICE_CALCULATIONS.cycles import OrnsteinUhlenbeckFundamental
from market import Market
from run_experiment import make_agents
from AGENTS.rl_agent import RLAgent
from market_quality import excess_kurtosis, returns_from, bootstrap_ci


num_seeds = 500
seeds = list(range(1, num_seeds + 1))
output = "robustness_flags.csv"
depth_agents = 25
depth_divisor = 4.0
copy_counts = [0, 8, 16]

# the reference market had both of these off
flag_sets = [
    ("reference", False, False),
    ("intra_step", True, False),
    ("spread", False, True),
    ("both", True, True),
]


def measure(copy_count, intra_step, spread, writer):
    VALUES["intra_step_impact"] = intra_step
    VALUES["market_spread_enabled"] = spread

    # the rl copies never trained with these on
    extra = [RLAgent(f"Copy{i + 1}") for i in range(copy_count)]
    market = Market(make_agents() + extra, OrnsteinUhlenbeckFundamental())
    market.quiet = True

    market.market_starting_cash = (VALUES["market_base_cash"] + VALUES["market_cash_per_agent"] * depth_agents) / depth_divisor
    market.market_starting_inventory = int((VALUES["market_base_inventory"] + VALUES["market_inventory_per_agent"] * depth_agents) / depth_divisor)

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

        writer.writerow({
            "flags": f"intra={int(intra_step)} spread={int(spread)}",
            "copies": copy_count,
            "seed": seed,
            "excess_kurtosis": f"{kurtosis:.6f}",
        })

    return kurtoses


print(f"market flag robustness over {len(seeds)} seeds, depth fixed at {depth_agents} agents, divisor {depth_divisor}")
print("rl homogeneous only")

fields = ["flags", "copies", "seed", "excess_kurtosis"]
output_file = open(output, "w", newline="")
writer = csv.DictWriter(output_file, fieldnames=fields)
writer.writeheader()

for label, intra_step, spread in flag_sets:
    print(f"\n{label}, intra step impact {intra_step}, spread {spread}")
    baseline = measure(0, intra_step, spread, writer)
    mean, low, high = bootstrap_ci(baseline)
    print(f"  N= 0 kurtosis={mean:.3f} 95% ci=[{low:.3f}, {high:.3f}]")

    for copy_count in copy_counts[1:]:
        kurtoses = measure(copy_count, intra_step, spread, writer)
        mean, low, high = bootstrap_ci(kurtoses)
        shift, shift_low, shift_high = bootstrap_ci(np.array(kurtoses) - np.array(baseline))
        if shift_low > 0 or shift_high < 0:
            separated = "yes"
        else:
            separated = "no"
        print(f"  N={copy_count} kurtosis={mean:.3f} 95% ci=[{low:.3f}, {high:.3f}] shift={shift:+.3f} [{shift_low:+.3f}, {shift_high:+.3f}] sep={separated}")

output_file.close()
print(f"\nper seed rows written to {output}")
