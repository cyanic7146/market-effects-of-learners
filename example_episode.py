import csv
import os
import random

import numpy as np

from values import VALUES
from PRICE_CALCULATIONS.cycles import OrnsteinUhlenbeckFundamental
from market import Market
from run_experiment import make_agents
from AGENTS.rl_agent import RLAgent
from market_quality import excess_kurtosis, returns_from


output = "example_episode.csv"
bloc_size = 16
depth_agents = 25
depth_divisor = 4.0
VALUES["market_spread_enabled"] = True
BROKEN = 10.0


def build(extra):
    market = Market(make_agents() + extra, OrnsteinUhlenbeckFundamental())
    market.quiet = True
    market.market_starting_cash = (VALUES["market_base_cash"] + VALUES["market_cash_per_agent"] * depth_agents) / depth_divisor
    market.market_starting_inventory = int((VALUES["market_base_inventory"] + VALUES["market_inventory_per_agent"] * depth_agents) / depth_divisor)
    return market


def run(market, seed):
    env_rng = random.Random(seed)
    market.rng = env_rng
    market.fundamental_process.rng = env_rng
    random.seed(seed)
    market.reset()
    for i in range(VALUES["num_steps"]):
        market.step()
    return list(market.history)


broken = {}
for row in csv.DictReader(open("sweep_thin.csv")):
    if row["depth"] == "fixed" and row["population"] == "rl_homogeneous" and int(row["copies"]) == bloc_size and float(row["excess_kurtosis"]) > BROKEN:
        broken[int(row["seed"])] = float(row["excess_kurtosis"])
median = float(np.median(list(broken.values())))
candidates = sorted(broken, key=lambda s: abs(broken[s] - median))

bloc_market = build([RLAgent(f"Copy{i + 1}") for i in range(bloc_size)])
none_market = build([])
for seed in candidates:
    with_bloc = run(bloc_market, seed)
    kurtosis = excess_kurtosis(returns_from(with_bloc))
    inventory = []
    for entry in with_bloc:
        inventory.append(entry["market_inventory"])
    if kurtosis > BROKEN and min(inventory) == 0:
        break
without = run(none_market, seed)

fundamental_gap = max(abs(a["fundamental_value"] - b["fundamental_value"]) for a, b in zip(without, with_bloc))
prices = []
for entry in with_bloc:
    prices.append(entry["market_price"])
print(f"seed {seed}: kurtosis with bloc {kurtosis:.1f} (sweep says {broken[seed]:.1f}, median of broken episodes {median:.1f}), without bloc {excess_kurtosis(returns_from(without)):.2f}")
print(f"maker first empty at step {inventory.index(0)}, price there {prices[inventory.index(0) - 1]:.1f} -> {prices[inventory.index(0)]:.1f}, episode high {max(prices):.1f}, fundamental paths differ by at most {fundamental_gap:.2e}")

with open(output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["seed", "step", "fundamental", "price_no_bloc", "inventory_no_bloc", "price_rl_bloc", "inventory_rl_bloc"])
    writer.writeheader()
    for step, (a, b) in enumerate(zip(without, with_bloc)):
        writer.writerow({"seed": seed, "step": step, "fundamental": f"{a['fundamental_value']:.4f}", "price_no_bloc": f"{a['market_price']:.4f}", "inventory_no_bloc": a["market_inventory"], "price_rl_bloc": f"{b['market_price']:.4f}", "inventory_rl_bloc": b["market_inventory"]})
print(f"written to {output}")
