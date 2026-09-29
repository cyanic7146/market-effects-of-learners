import os
import random

import numpy as np

from values import VALUES
from PRICE_CALCULATIONS.cycles import OrnsteinUhlenbeckFundamental
from market import Market
from run_experiment import make_agents
from AGENTS.rl_agent import RLAgent


seeds = list(range(1, 1001))
copy_counts = [1, 2, 4, 8]
fixed_depth = False
depth_agents = 25


if not (os.path.exists("MODELS/best_model_31obs_checkpoint/best_model.zip") and os.path.exists("MODELS/vec_normalize.pkl")):
    print("no trained rl model found, train first then run this")
else:
    starting_cash = VALUES["initial_cash"]
    print(f"crowding experiment over {len(seeds)} seeds")
    if fixed_depth:
        print(f"depth pinned at {depth_agents} agents")
    print()

    per_seed_by_count = {}

    for copy_count in copy_counts:
        agents = make_agents() + [RLAgent(f"RLCopy{i + 1}") for i in range(copy_count)]
        fundamental_process = OrnsteinUhlenbeckFundamental()
        market = Market(agents, fundamental_process)
        market.quiet = True

        if fixed_depth:
            market.market_starting_cash = VALUES["market_base_cash"] + VALUES["market_cash_per_agent"] * depth_agents
            market.market_starting_inventory = VALUES["market_base_inventory"] + VALUES["market_inventory_per_agent"] * depth_agents

        copy_names = [f"RLCopy{i + 1}" for i in range(copy_count)]
        per_copy_finals = []
        bah_finals = []

        for seed in seeds:
            random.seed(seed)
            market.reset()
            for i in range(VALUES["num_steps"]):
                market.step()

            copy_values = []
            bah_value = 0.0
            for agent in agents:
                if agent.name in copy_names:
                    copy_values.append(agent.portfolio_value(market.price))
                if agent.name == "BuyAndHold1":
                    bah_value = agent.portfolio_value(market.price)

            per_copy_finals.append(sum(copy_values) / len(copy_values))
            bah_finals.append(bah_value)

        average_copy = sum(per_copy_finals) / len(seeds)
        average_return = (average_copy - starting_cash) / starting_cash * 100
        beat_bah = sum(1 for v, b in zip(per_copy_finals, bah_finals) if v > b)
        per_seed_by_count[copy_count] = np.array(per_copy_finals)
        print(f"  {copy_count} copies: per-copy avg={average_copy:.2f} return={average_return:.2f}% beat_bah={beat_bah}/{len(seeds)}")

    one = per_seed_by_count[copy_counts[0]]
    many = per_seed_by_count[copy_counts[-1]]
    differences = (one - many) / starting_cash * 100
    standard_error = differences.std(ddof=1) / np.sqrt(len(differences))
    print(f"\npaired {copy_counts[0]} copy minus {copy_counts[-1]} copies, same seeds")
    print(f"  mean gap={differences.mean():.2f} points se={standard_error:.3f} t={differences.mean() / standard_error:.2f}")
    print(f"  95% ci=[{differences.mean() - 1.96 * standard_error:.2f}, {differences.mean() + 1.96 * standard_error:.2f}]")
