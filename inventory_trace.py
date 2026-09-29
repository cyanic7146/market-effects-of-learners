import csv
import os
import random

import numpy as np

from values import VALUES
from PRICE_CALCULATIONS.cycles import OrnsteinUhlenbeckFundamental
from market import Market
from run_experiment import make_agents
from AGENTS.momentum_agent import MomentumAgent
from AGENTS.rl_agent import RLAgent
from AGENTS.scripted_agent import ScriptedAgent
from market_quality import excess_kurtosis, returns_from, bloc_flows, flow_runs, bootstrap_ci


num_seeds = 300
seeds = list(range(1, num_seeds + 1))
output = "inventory_trace.csv"
bloc_size = 16
depth_agents = 25
depth_divisor = 4.0
VALUES["market_spread_enabled"] = True
script_cash = 200000.0
# 16 agents x 22 shares x 4 steps, about the rl blocs biggest run
run_quantity = 22
RUN_LENGTH = 4
run_start = 100 # arbitrary
BROKEN = 10.0 # excess kurtosis above this = broken


def build(extra, fixed_depth):
    market = Market(make_agents() + extra, OrnsteinUhlenbeckFundamental())
    market.quiet = True
    if fixed_depth:
        market.market_starting_cash = VALUES["market_base_cash"] + VALUES["market_cash_per_agent"] * depth_agents
        market.market_starting_inventory = VALUES["market_base_inventory"] + VALUES["market_inventory_per_agent"] * depth_agents
    market.market_starting_cash = market.market_starting_cash / depth_divisor
    market.market_starting_inventory = int(market.market_starting_inventory / depth_divisor)
    return market


def run(market, seed, script_agents=()):
    env_rng = random.Random(seed)
    market.rng = env_rng
    market.fundamental_process.rng = env_rng
    random.seed(seed)
    market.reset()
    for agent in script_agents:
        agent.cash = script_cash
    for i in range(VALUES["num_steps"]):
        market.step()


def single_run(buy_start, sell_start):
    schedule = [0] * VALUES["num_steps"]
    for j in range(RUN_LENGTH):
        schedule[buy_start + j] = run_quantity
        if sell_start is not None:
            schedule[sell_start + j] = -run_quantity
    return schedule


# runs on a clock, direction flips every run
def alternating(quantity, run_length, runs):
    schedule = [0] * VALUES["num_steps"]
    spacing = VALUES["num_steps"] // runs
    sign = 1
    for i in range(runs):
        for j in range(run_length):
            if i * spacing + j < VALUES["num_steps"]:
                schedule[i * spacing + j] = sign * quantity
        sign = -sign
    return schedule


# from the schedule, not from what actually filled
def schedule_stats(scripts):
    if not scripts:
        return 0, 0, 0, 0.0, 0
    total = np.array([agent.schedule for agent in scripts]).sum(axis=0)
    runs = flow_runs(total)
    bought = int(total[total > 0].sum())
    sold = int(-total[total < 0].sum())
    if runs:
        mean_run = float(np.mean(runs))
    else:
        mean_run = 0.0
    if runs:
        biggest_run = int(max(runs))
    else:
        biggest_run = 0
    return bought, sold, len(runs), mean_run, biggest_run


def record_row(writer, cell, seed, market, scripts):
    inventory = np.array([entry["market_inventory"] for entry in market.history])
    kurtosis = excess_kurtosis(returns_from(market.history))
    bought, sold, runs, mean_run, biggest_run = schedule_stats(scripts)
    writer.writerow({
        "cell": cell,
        "seed": seed,
        "excess_kurtosis": f"{kurtosis:.6f}",
        "min_inventory": int(inventory.min()),
        "emptied": int(inventory.min() == 0),
        "bloc_bought": bought,
        "bloc_sold": sold,
        "bloc_runs": runs,
        "bloc_mean_run": f"{mean_run:.1f}",
        "bloc_biggest_run": biggest_run,
    })
    return kurtosis, inventory.min() == 0, bought - sold


print(f"inventory trace over {len(seeds)} seeds, thin market at fixed depth, maker starts with {int((VALUES['market_base_inventory'] + VALUES['market_inventory_per_agent'] * depth_agents) / depth_divisor)} shares")

fields = ["cell", "seed", "excess_kurtosis", "min_inventory", "emptied", "bloc_bought", "bloc_sold", "bloc_runs", "bloc_mean_run", "bloc_biggest_run"]
output_file = open(output, "w", newline="")
writer = csv.DictWriter(output_file, fieldnames=fields)
writer.writeheader()

results = {}

baseline_market = build([], True)
kurtoses, emptied, kept = [], [], []
for seed in seeds:
    run(baseline_market, seed)
    k, e, n = record_row(writer, "no bloc", seed, baseline_market, [])
    kurtoses.append(k)
    emptied.append(e)
    kept.append(n)
baseline = np.array(kurtoses)
results["no bloc"] = (baseline, np.array(emptied), np.array(kept))

for arm, factory in [("momentum replayed", lambda i: MomentumAgent(f"Copy{i + 1}", lookback=8, trade_fraction=0.25, hard_cap=100)), ("RL replayed", lambda i: RLAgent(f"Copy{i + 1}"))]:
    live = [factory(i) for i in range(bloc_size)]
    names = [agent.name for agent in live]
    record_market = build(live, False)
    scripts = [ScriptedAgent(f"Script{i + 1}", [0] * VALUES["num_steps"], cash=script_cash) for i in range(bloc_size)]
    replay_market = build(scripts, True)
    kurtoses, emptied, kept = [], [], []
    for seed in seeds:
        run(record_market, seed)
        flows = bloc_flows(record_market.history, names)
        for i, agent in enumerate(scripts):
            agent.schedule = [int(q) for q in flows[names[i]]]
        run(replay_market, seed, scripts)
        k, e, n = record_row(writer, arm, seed, replay_market, scripts)
        kurtoses.append(k)
        emptied.append(e)
        kept.append(n)
    results[arm] = (np.array(kurtoses), np.array(emptied), np.array(kept))

# one rl sized run given back after 30, 100 or never, then clock runs at 4 sizes
cells = [
    ("one run, sold back after 30 steps", single_run(run_start, run_start + 30)),
    ("one run, sold back after 100 steps", single_run(run_start, run_start + 100)),
    ("one run, never sold back", single_run(run_start, None)),
    ("sixteen-step runs on a clock, 768 shares", alternating(3, 16, 18)),
    ("sixteen-step runs on a clock, 1,024 shares", alternating(4, 16, 18)),
    ("sixteen-step runs on a clock, 1,280 shares", alternating(5, 16, 18)),
    ("sixteen-step runs on a clock, 1,536 shares", alternating(6, 16, 18)),
]
for cell, schedule in cells:
    scripts = [ScriptedAgent(f"Script{i + 1}", schedule, cash=script_cash) for i in range(bloc_size)]
    market = build(scripts, True)
    kurtoses, emptied, kept = [], [], []
    for seed in seeds:
        run(market, seed, scripts)
        k, e, n = record_row(writer, cell, seed, market, scripts)
        kurtoses.append(k)
        emptied.append(e)
        kept.append(n)
    results[cell] = (np.array(kurtoses), np.array(emptied), np.array(kept))

output_file.close()

print("\n cell maker emptied market broke shares kept kurtosis shift")
for cell, (kurtoses, emptied, kept) in results.items():
    shift, low, high = bootstrap_ci(kurtoses - baseline)
    print(f"  {cell} {100 * emptied.mean():.0f}% {100 * (kurtoses > BROKEN).mean():.0f}% {kept.mean():.0f} {shift:+.1f} [{low:+.1f}, {high:+.1f}]")
print(f"\nper seed rows written to {output}")
