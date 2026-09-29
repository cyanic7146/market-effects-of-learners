import random
import numpy as np

from values import VALUES
from PRICE_CALCULATIONS.cycles import OrnsteinUhlenbeckFundamental
from market import Market
from run_experiment import make_agents
from AGENTS.momentum_agent import MomentumAgent


seeds = list(range(1, 101))
depth_divisors = [1.0, 2.0, 4.0, 5.0, 6.0, 7.0]
copy_counts = [0, 4, 8, 16]


def run(copy_count, depth_divisor):
    names = {f"MomCopy{i + 1}" for i in range(copy_count)}
    extra = [MomentumAgent(f"MomCopy{i + 1}", lookback=8, trade_fraction=0.20, hard_cap=100) for i in range(copy_count)]
    market = Market(make_agents() + extra, OrnsteinUhlenbeckFundamental())
    market.quiet = True

    market.market_starting_cash = market.market_starting_cash / depth_divisor
    market.market_starting_inventory = int(market.market_starting_inventory / depth_divisor)

    flow_impacts = []
    bloc_shares = []
    gaps = []
    kurtoses = []

    for seed in seeds:
        env_rng = random.Random(seed)
        market.rng = env_rng
        market.fundamental_process.rng = env_rng
        random.seed(seed)
        market.reset()
        for i in range(VALUES["num_steps"]):
            market.step()

        prices = np.array([entry["market_price"] for entry in market.history])
        fundamentals = np.array([entry["fundamental_value"] for entry in market.history])
        gaps.append(np.mean(np.abs(prices - fundamentals) / fundamentals) * 100)

        returns = np.diff(prices) / prices[:-1]
        if returns.std() > 0:
            kurtoses.append(((returns - returns.mean()) ** 4).mean() / returns.std() ** 4 - 3)

        for entry in market.history:
            flow_impacts.append(abs(entry["liquidity_adjusted_order_flow_impact"]))
            bloc_shares.append(sum(a["executed_quantity"] for a in entry["actions"] if a["agent"] in names))

    flow_ratio = np.mean(flow_impacts) / VALUES["market_noise"]
    return flow_ratio, np.mean(bloc_shares), np.mean(gaps), np.median(gaps), np.mean(kurtoses)


def sweep(label):
    print(f"\n{label}")
    print("  depth N bloc/step flow/noise mean gap median gap kurtosis")
    for divisor in depth_divisors:
        for copy_count in copy_counts:
            flow_ratio, bloc, mean_gap, median_gap, kurtosis = run(copy_count, divisor)
            print(f"  {'/' + str(int(divisor))} {copy_count} {bloc:.1f} {flow_ratio:.3f} {mean_gap:.2f}% {median_gap:.2f}% {kurtosis:.2f}")
        print()


print(f"treatment power check over {len(seeds)} seeds")

sweep("spread off, the reference market")

saved_spread = VALUES["market_spread_enabled"]
VALUES["market_spread_enabled"] = True
sweep("spread on, market maker skews quotes against its own inventory")
VALUES["market_spread_enabled"] = saved_spread
