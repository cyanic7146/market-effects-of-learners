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
from market_quality import excess_kurtosis, returns_from, bloc_flow_size, bootstrap_ci


num_seeds = 300
seeds = list(range(1, num_seeds + 1))
output = "recorded_replay.csv"
bloc_size = 16
depth_agents = 25
depth_divisor = 4.0
VALUES["market_spread_enabled"] = True
script_cash = 200000.0


def momentum_copy(i):
    return MomentumAgent(f"Copy{i + 1}", lookback=8, trade_fraction=0.25, hard_cap=100)


def rl_copy(i):
    return RLAgent(f"Copy{i + 1}")


arms = [("momentum", momentum_copy), ("rl_homogeneous", rl_copy)]


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


# what each copy actually traded each step
def record(market, names, seed):
    run(market, seed)
    schedules = {name: [] for name in names}
    for entry in market.history:
        signed = {name: 0 for name in names}
        for action in entry["actions"]:
            if action["agent"] in signed:
                quantity = int(action["executed_quantity"])
                signed[action["agent"]] = quantity if action["action_type"] == "buy" else -quantity
        for name in names:
            schedules[name].append(signed[name])
    return schedules


print(f"recorded flow replay over {len(seeds)} seeds, {VALUES['num_steps']} steps")

fields = ["arm", "stage", "seed", "excess_kurtosis", "bloc_volume"]
output_file = open(output, "w", newline="")
writer = csv.DictWriter(output_file, fieldnames=fields)
writer.writeheader()

baseline_market = build([], True)
baseline = []
for seed in seeds:
    run(baseline_market, seed)
    kurtosis = excess_kurtosis(returns_from(baseline_market.history))
    baseline.append(kurtosis)
    writer.writerow({"arm": "none", "stage": "fixed", "seed": seed, "excess_kurtosis": f"{kurtosis:.6f}", "bloc_volume": 0})

mean, low, high = bootstrap_ci(baseline)
print(f"\n no bloc fixed depth kurtosis={mean:.3f} 95% ci=[{low:.3f}, {high:.3f}]")

# record at scaling depth where the market holds, replay at fixed depth
for arm, factory in arms:
    live = [factory(i) for i in range(bloc_size)]
    names = [agent.name for agent in live]
    record_market = build(live, False)

    scripts = [ScriptedAgent(f"Script{i + 1}", [0] * VALUES["num_steps"], cash=script_cash) for i in range(bloc_size)]
    script_names = [agent.name for agent in scripts]
    replay_market = build(scripts, True)

    live_kurtoses = []
    replay_kurtoses = []
    volumes = []
    for seed in seeds:
        schedules = record(record_market, names, seed)
        live_kurtosis = excess_kurtosis(returns_from(record_market.history))
        live_kurtoses.append(live_kurtosis)
        writer.writerow({"arm": arm, "stage": "recorded_at_scaling", "seed": seed, "excess_kurtosis": f"{live_kurtosis:.6f}", "bloc_volume": sum(abs(q) for name in names for q in schedules[name])})

        # each script agent replays one copy's orders
        for i, agent in enumerate(scripts):
            agent.schedule = schedules[names[i]]

        run(replay_market, seed, scripts)
        kurtosis = excess_kurtosis(returns_from(replay_market.history))
        replay_kurtoses.append(kurtosis)
        mean_flow, _ = bloc_flow_size(replay_market.history, script_names)
        volumes.append(mean_flow * VALUES["num_steps"])
        writer.writerow({"arm": arm, "stage": "replayed_at_fixed", "seed": seed, "excess_kurtosis": f"{kurtosis:.6f}", "bloc_volume": f"{mean_flow * VALUES['num_steps']:.1f}"})

    mean, low, high = bootstrap_ci(live_kurtoses)
    print(f"\n {arm} live, scaling kurtosis={mean:.3f} 95% ci=[{low:.3f}, {high:.3f}]")
    mean, low, high = bootstrap_ci(replay_kurtoses)
    shift, shift_low, shift_high = bootstrap_ci(np.array(replay_kurtoses) - np.array(baseline))
    print(f"  {arm} replay, fixed kurtosis={mean:.3f} 95% ci=[{low:.3f}, {high:.3f}] shift={shift:+.3f} [{shift_low:+.3f}, {shift_high:+.3f}] volume={np.mean(volumes):.0f}")

output_file.close()
print(f"\nper seed rows written to {output}")
