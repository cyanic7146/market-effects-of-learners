import csv
import os
import random

import numpy as np

from values import VALUES
from PRICE_CALCULATIONS.cycles import OrnsteinUhlenbeckFundamental
from market import Market
from run_experiment import make_agents
from AGENTS.scripted_agent import ScriptedAgent
from market_quality import excess_kurtosis, returns_from, bloc_flow_size, bootstrap_ci


num_seeds = 500
seeds = list(range(1, num_seeds + 1))
output = "flow_replay.csv"
# output = "flow_replay_d2.csv"
# output = "flow_replay_d6.csv"
bloc_size = 16
depth_agents = 25
depth_divisor = 4.0
# depth_divisor = 2.0
# depth_divisor = 6.0
spread = True
VALUES["market_spread_enabled"] = spread

total_volumes = [4800, 9600, 14400]
ACTIVE_STEPS = [300, 100, 30, 10] # 300 is a trickle, 10 is a burst
shapes = ["spaced", "paired"]
run_active_steps = 60
run_quantities = [10, 20]
run_lengths = [1, 2, 4, 10, 20]
script_cash = 200000.0 # enough that cash never cuts the schedule short


# every scripted agent gets the same schedule, sign flips so position goes back to 0
def bloc_schedule(total_volume, active_steps, shape, run_length, num_steps):
    quantity = int(round(total_volume / (bloc_size * active_steps)))
    schedule = [0] * num_steps

    #same direction for run_length steps in a row
    if shape == "run":
        runs = active_steps // run_length
        spacing = num_steps // runs
        sign = 1
        for i in range(runs):
            start = i * spacing
            for j in range(run_length):
                schedule[start + j] = sign * quantity
            sign = -sign
    elif shape == "spaced":
        spacing = num_steps // active_steps
        sign = 1
        for i in range(active_steps):
            schedule[i * spacing] = sign * quantity
            sign = -sign
    else:
        pairs = active_steps // 2
        spacing = num_steps // pairs
        for i in range(pairs):
            start = i * spacing
            schedule[start] = quantity
            schedule[start + 1] = -quantity

    return schedule, bloc_size * quantity * active_steps


def measure(total_volume, active_steps, shape, fixed_depth, writer, run_length=1):
    if active_steps:
        schedule, scheduled = bloc_schedule(total_volume, active_steps, shape, run_length, VALUES["num_steps"])
        extra = [ScriptedAgent(f"Script{i + 1}", schedule, cash=script_cash) for i in range(bloc_size)]
    else:
        schedule, scheduled = [], 0
        extra = []

    names = [agent.name for agent in extra]
    market = Market(make_agents() + extra, OrnsteinUhlenbeckFundamental())
    market.quiet = True

    if fixed_depth:
        market.market_starting_cash = VALUES["market_base_cash"] + VALUES["market_cash_per_agent"] * depth_agents
        market.market_starting_inventory = VALUES["market_base_inventory"] + VALUES["market_inventory_per_agent"] * depth_agents

    market.market_starting_cash = market.market_starting_cash / depth_divisor
    market.market_starting_inventory = int(market.market_starting_inventory / depth_divisor)

    kurtoses = []
    delivered = []
    for seed in seeds:
        env_rng = random.Random(seed)
        market.rng = env_rng
        market.fundamental_process.rng = env_rng
        random.seed(seed)
        market.reset(VALUES["initial_cash"])
        # reset script cash each episode
        for agent in extra:
            agent.cash = script_cash
        for i in range(VALUES["num_steps"]):
            market.step()

        returns = returns_from(market.history)
        kurtosis = excess_kurtosis(returns)
        kurtoses.append(kurtosis)
        mean_flow, peak_flow = bloc_flow_size(market.history, names)
        delivered.append(mean_flow * VALUES["num_steps"])

        writer.writerow({
            "total_volume": total_volume,
            "active_steps": active_steps,
            "shape": shape,
            "run_length": run_length,
            "depth": "fixed" if fixed_depth else "scaling",
            "seed": seed,
            "scheduled_volume": scheduled,
            "delivered_volume": f"{mean_flow * VALUES['num_steps']:.2f}",
            "excess_kurtosis": f"{kurtosis:.6f}",
            "bloc_mean_flow": f"{mean_flow:.6f}",
            "bloc_peak_flow": f"{peak_flow:.6f}",
        })

    return kurtoses, scheduled, float(np.mean(delivered))


print(f"flow replay over {len(seeds)} seeds, {VALUES['num_steps']} steps")
print(f"depth divisor {depth_divisor}, market maker spread {spread}")
print(f"{bloc_size} scripted agents, one shared schedule")

fields = ["total_volume", "active_steps", "shape", "run_length", "depth", "seed", "scheduled_volume", "delivered_volume", "excess_kurtosis", "bloc_mean_flow", "bloc_peak_flow"]
output_file = open(output, "w", newline="")
writer = csv.DictWriter(output_file, fieldnames=fields)
writer.writeheader()

for fixed_depth in [False, True]:
    print(f"\nfixed_depth={fixed_depth}")

    baseline, _, _ = measure(0, 0, "none", fixed_depth, writer)
    mean, low, high = bootstrap_ci(baseline)
    print(f"  no bloc kurtosis={mean:.3f} 95% ci=[{low:.3f}, {high:.3f}]")

    for total_volume in total_volumes:
        for shape in shapes:
            print("")
            for active_steps in ACTIVE_STEPS:
                kurtoses, scheduled, delivered = measure(total_volume, active_steps, shape, fixed_depth, writer)
                mean, low, high = bootstrap_ci(kurtoses)
                differences = np.array(kurtoses) - np.array(baseline)
                shift, shift_low, shift_high = bootstrap_ci(differences)
                if shift_low > 0 or shift_high < 0:
                    separated = "yes"
                else:
                    separated = "no"
                if scheduled:
                    fill = 100 * delivered / scheduled
                else:
                    fill = 0.0
                print(f"  vol={scheduled} {shape} over {active_steps} steps kurtosis={mean:.3f} 95% ci=[{low:.3f}, {high:.3f}] shift={shift:+.3f} [{shift_low:+.3f}, {shift_high:+.3f}] sep={separated} filled={fill:.1f}%")

    # runs: same peak, different run lengths
    for quantity in run_quantities:
        print("")
        for run_length in run_lengths:
            volume = bloc_size * quantity * run_active_steps
            kurtoses, scheduled, delivered = measure(volume, run_active_steps, "run", fixed_depth, writer, run_length)
            mean, low, high = bootstrap_ci(kurtoses)
            differences = np.array(kurtoses) - np.array(baseline)
            shift, shift_low, shift_high = bootstrap_ci(differences)
            if shift_low > 0 or shift_high < 0:
                separated = "yes"
            else:
                separated = "no"
            if scheduled:
                fill = 100 * delivered / scheduled
            else:
                fill = 0.0
            print(f"  peak={bloc_size * quantity} vol={scheduled} runs of {run_length} kurtosis={mean:.3f} 95% ci=[{low:.3f}, {high:.3f}] shift={shift:+.3f} [{shift_low:+.3f}, {shift_high:+.3f}] sep={separated} filled={fill:.1f}%")

output_file.close()
print(f"\nper seed rows written to {output}")
