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
from market_quality import bootstrap_ci


num_seeds = 1000
seeds = list(range(1, num_seeds + 1))
COPY_COUNT = 16
depth_agents = 25
output = "background_returns.csv"
# output = "background_returns_thin.csv"
depth_divisor = 1.0
# depth_divisor = 4.0
spread = False
# spread = True
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


# avg % return for a group of agents
def percent_return(agents, price):
    finals = []
    for agent in agents:
        finals.append(agent.portfolio_value(price))
    return (np.mean(finals) - VALUES["initial_cash"]) / VALUES["initial_cash"] * 100


def measure(population, copy_count, fixed_depth, writer):
    if copy_count:
        extra = [dict(populations)[population](i) for i in range(copy_count)]
    else:
        extra = []
    background = make_agents() # the normal 24
    market = Market(background + extra, OrnsteinUhlenbeckFundamental())
    market.quiet = True

    # fixed depth: maker sized like there are 25 agents
    if fixed_depth:
        market.market_starting_cash = VALUES["market_base_cash"] + VALUES["market_cash_per_agent"] * depth_agents
        market.market_starting_inventory = VALUES["market_base_inventory"] + VALUES["market_inventory_per_agent"] * depth_agents

    market.market_starting_cash = market.market_starting_cash / depth_divisor
    market.market_starting_inventory = int(market.market_starting_inventory / depth_divisor)

    background_returns = []
    for seed in seeds:
        env_rng = random.Random(seed)
        market.rng = env_rng
        market.fundamental_process.rng = env_rng
        random.seed(seed)
        market.reset()
        for i in range(VALUES["num_steps"]):
            market.step()

        background_return = percent_return(background, market.price)
        background_returns.append(background_return)
        if extra:
            bloc_return = percent_return(extra, market.price)
        else:
            bloc_return = 0.0
        # market maker marked at the final price
        maker_start = market.market_starting_cash + market.market_starting_inventory * VALUES["initial_price"]
        maker_end = market.market_cash + market.market_inventory * market.price
        maker_pnl = (maker_end - maker_start) / maker_start * 100

        writer.writerow({
            "population": population,
            "copies": copy_count,
            "depth": "fixed" if fixed_depth else "scaling",
            "depth_divisor": depth_divisor,
            "spread": int(spread),
            "seed": seed,
            "background_return": f"{background_return:.4f}",
            "bloc_return": f"{bloc_return:.4f}",
            "maker_pnl": f"{maker_pnl:.4f}",
        })

    return background_returns


print(f"background returns over {len(seeds)} seeds, n=0 against n={COPY_COUNT}")
print(f"depth divisor {depth_divisor}, market maker spread {spread}")

fields = ["population", "copies", "depth", "depth_divisor", "spread", "seed", "background_return", "bloc_return", "maker_pnl"]
output_file = open(output, "w", newline="")
writer = csv.DictWriter(output_file, fieldnames=fields)
writer.writeheader()

for fixed_depth in [False, True]:
    print(f"\nfixed_depth={fixed_depth}")
    baseline = measure("none", 0, fixed_depth, writer)
    mean, low, high = bootstrap_ci(baseline)
    print(f"  none N= 0 background return={mean:+.3f}% 95% ci=[{low:+.3f}, {high:+.3f}]")

    for population, _ in populations:
        returns = measure(population, COPY_COUNT, fixed_depth, writer)
        shift, shift_low, shift_high = bootstrap_ci(np.array(returns) - np.array(baseline))
        print(f"  {population} N={COPY_COUNT} background return={np.mean(returns):+.3f}% paired shift vs N=0: {shift:+.3f} [{shift_low:+.3f}, {shift_high:+.3f}] pts")

output_file.close()
print(f"\nper seed rows written to {output}")
