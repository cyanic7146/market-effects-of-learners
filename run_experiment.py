import os
import random
import numpy as np

from values import VALUES
from PRICE_CALCULATIONS.cycles import OrnsteinUhlenbeckFundamental
from AGENTS.random_agent import RandomAgent
from AGENTS.momentum_agent import MomentumAgent
from AGENTS.mean_reversion_agent import MeanReversionAgent
from AGENTS.buy_and_hold_agent import BuyAndHoldAgent
from AGENTS.contrarian_agent import ContrarianAgent
from AGENTS.volume_pressure_agent import VolumePressureAgent
from AGENTS.value_estimate_agent import ValueEstimateAgent
from AGENTS.risk_managed_agent import RiskManagedAgent
from market import Market


seeds = list(range(1, 1001))


def make_agents():
    return [
        RandomAgent("Random1", trade_fraction=0.20, hard_cap=80),
        RandomAgent("Random2", trade_fraction=0.30, hard_cap=100),
        RandomAgent("Random3", trade_fraction=0.40, hard_cap=120),
        MomentumAgent("Momentum1", lookback=3, trade_fraction=0.15, hard_cap=80),
        MomentumAgent("Momentum2", lookback=8, trade_fraction=0.20, hard_cap=100),
        MomentumAgent("Momentum3", lookback=15, trade_fraction=0.25, hard_cap=120),
        MeanReversionAgent("MeanReversion1", lookback=10, threshold=0.015, trade_fraction=0.15, hard_cap=80),
        MeanReversionAgent("MeanReversion2", lookback=20, threshold=0.02, trade_fraction=0.20, hard_cap=100),
        MeanReversionAgent("MeanReversion3", lookback=40, threshold=0.03, trade_fraction=0.25, hard_cap=120),
        BuyAndHoldAgent("BuyAndHold1", buy_fraction=0.60),
        BuyAndHoldAgent("BuyAndHold2", buy_fraction=0.75),
        BuyAndHoldAgent("BuyAndHold3", buy_fraction=0.90),
        ContrarianAgent("Contrarian1", threshold=0.006, trade_fraction=0.15, hard_cap=80),
        ContrarianAgent("Contrarian2", threshold=0.008, trade_fraction=0.20, hard_cap=100),
        ContrarianAgent("Contrarian3", threshold=0.012, trade_fraction=0.25, hard_cap=120),
        VolumePressureAgent("VolumePressure1", sensitivity=0.20, trade_fraction=0.15, hard_cap=80),
        VolumePressureAgent("VolumePressure2", sensitivity=0.30, trade_fraction=0.20, hard_cap=100),
        VolumePressureAgent("VolumePressure3", sensitivity=0.40, trade_fraction=0.25, hard_cap=120),
        ValueEstimateAgent("ValueEstimate1", lookback=40, threshold=0.02, trade_fraction=0.15, hard_cap=80),
        ValueEstimateAgent("ValueEstimate2", lookback=60, threshold=0.03, trade_fraction=0.20, hard_cap=100),
        ValueEstimateAgent("ValueEstimate3", lookback=80, threshold=0.04, trade_fraction=0.25, hard_cap=120),
        RiskManagedAgent("RiskManaged1", lookback=15, threshold=0.02, trade_fraction=0.15, hard_cap=80),
        RiskManagedAgent("RiskManaged2", lookback=20, threshold=0.025, trade_fraction=0.20, hard_cap=100),
        RiskManagedAgent("RiskManaged3", lookback=30, threshold=0.03, trade_fraction=0.25, hard_cap=120),
    ]


def max_drawdown(path):
    peak = path[0]
    worst = 0.0
    for value in path:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > worst:
            worst = drawdown
    return worst


def sharpe_of_path(path):
    returns = []
    for i in range(1, len(path)):
        returns.append((path[i] - path[i-1]) / path[i-1])
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    if variance == 0:
        return None
    return mean_return / variance ** 0.5


def path_never_moved(path):
    return max(path) == min(path)


def run_experiment():
    agents = make_agents()
    agent_names = [a.name for a in agents]

    results = {name: [] for name in agent_names}
    drawdowns = {name: [] for name in agent_names}
    sharpes = {name: [] for name in agent_names}
    idle_counts = {name: 0 for name in agent_names}

    fundamental_process = OrnsteinUhlenbeckFundamental()
    market = Market(agents, fundamental_process)
    market.quiet = True

    for seed in seeds:
        random.seed(seed)
        market.reset()

        paths = {name: [] for name in agent_names}
        for i in range(VALUES["num_steps"]):
            market.step()
            for agent in agents:
                paths[agent.name].append(agent.portfolio_value(market.price))

        for agent in agents:
            path = paths[agent.name]
            results[agent.name].append(path[-1])
            if path_never_moved(path):
                idle_counts[agent.name] += 1
            else:
                drawdowns[agent.name].append(max_drawdown(path))
                sharpes[agent.name].append(sharpe_of_path(path))

    bah_results = results["BuyAndHold1"]
    starting_cash = VALUES["initial_cash"]
    num_seeds = len(seeds)
    worst_count = max(1, int(0.05 * num_seeds))

    print(f"\nresults over {num_seeds} seeds")
    for name in agent_names:
        portfolio_results = results[name]
        average_portfolio = sum(portfolio_results) / num_seeds
        average_return = (average_portfolio - starting_cash) / starting_cash * 100
        times_beat_bah = sum(1 for v, b in zip(portfolio_results, bah_results) if v > b)
        traded_seeds = max(1, len(sharpes[name]))
        average_sharpe = sum(sharpes[name]) / traded_seeds
        average_drawdown = sum(drawdowns[name]) / traded_seeds * 100
        worst_finals = sorted(portfolio_results)[:worst_count]
        cvar = (sum(worst_finals) / worst_count - starting_cash) / starting_cash * 100
        if idle_counts[name]:
            idle_note = f"  idle={idle_counts[name]}/{num_seeds}"
        else:
            idle_note = ""
        print(f"  {name}: avg={average_portfolio:.2f} return={average_return:.2f}% beat_bah={times_beat_bah}/{num_seeds} sharpe={average_sharpe:.3f} maxdd={average_drawdown:.1f}% cvar5={cvar:.2f}%{idle_note}")

    rl_models = [
        ("MODELS/best_model_31obs_checkpoint/best_model", "MODELS/vec_normalize.pkl"),
        ("MODELS/best_model_calm_31obs_checkpoint/best_model", "MODELS/vec_normalize_calm.pkl"),
    ]

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    from market_env import MarketEnv

    print(f"\nrl model results over {num_seeds} seeds")
    for model_path, normalizer_path in rl_models:
        if not os.path.exists(model_path + ".zip") or not os.path.exists(normalizer_path):
            print(f"  {model_path}: not found, skipping")
            continue

        norm_loader = DummyVecEnv([MarketEnv])
        norm_loader = VecNormalize.load(normalizer_path, norm_loader)
        obs_mean = norm_loader.obs_rms.mean
        obs_var = norm_loader.obs_rms.var
        obs_clip = norm_loader.clip_obs

        model = PPO.load(model_path)
        env = MarketEnv()
        portfolio_results = []
        env_bah_results = []
        rl_drawdowns = []
        rl_sharpes = []
        rl_idle = 0

        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            obs = np.clip((obs - obs_mean) / np.sqrt(obs_var + 1e-8), -obs_clip, obs_clip).astype(np.float32)
            rl_path = []
            terminated = False
            while not terminated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                obs = np.clip((obs - obs_mean) / np.sqrt(obs_var + 1e-8), -obs_clip, obs_clip).astype(np.float32)
                rl_path.append(info["portfolio_value"])
            portfolio_results.append(rl_path[-1])
            env_bah_results.append(env.bah_agent.portfolio_value(env.market.price))
            if path_never_moved(rl_path):
                rl_idle += 1
            else:
                rl_drawdowns.append(max_drawdown(rl_path))
                rl_sharpes.append(sharpe_of_path(rl_path))

        average_portfolio = sum(portfolio_results) / num_seeds
        average_return = (average_portfolio - starting_cash) / starting_cash * 100
        times_beat_bah = sum(1 for v, b in zip(portfolio_results, env_bah_results) if v > b)
        traded_seeds = max(1, len(rl_sharpes))
        average_sharpe = sum(rl_sharpes) / traded_seeds
        average_drawdown = sum(rl_drawdowns) / traded_seeds * 100
        worst_finals = sorted(portfolio_results)[:worst_count]
        cvar = (sum(worst_finals) / worst_count - starting_cash) / starting_cash * 100
        print(f"  {model_path}: avg={average_portfolio:.2f} return={average_return:.2f}% beat_bah={times_beat_bah}/{num_seeds} sharpe={average_sharpe:.3f} maxdd={average_drawdown:.1f}% cvar5={cvar:.2f}% idle={rl_idle}/{num_seeds}")

        if rl_idle:
            traded_finals = []
            for v in portfolio_results:
                if v != starting_cash:
                    traded_finals.append(v)
            traded_return = (sum(traded_finals) / len(traded_finals) - starting_cash) / starting_cash * 100
            print(f"    over the {len(traded_finals)} traded episodes only: return={traded_return:.2f}%")

        differences = np.array(portfolio_results) - np.array(env_bah_results)
        rng = np.random.default_rng(0)
        boot_means = [rng.choice(differences, size=len(differences), replace=True).mean() for _ in range(10000)]
        ci_low = np.percentile(boot_means, 2.5)
        ci_high = np.percentile(boot_means, 97.5)
        wins = int((differences > 0).sum())
        print(f"    paired vs in-episode bah: mean diff={differences.mean():.2f} 95% ci=[{ci_low:.2f}, {ci_high:.2f}] wins={wins}/{num_seeds}")

        standard_error = differences.std(ddof=1) / np.sqrt(len(differences))
        direct_low = differences.mean() - 1.96 * standard_error
        direct_high = differences.mean() + 1.96 * standard_error
        print(f"    direct standard error interval: se={standard_error:.2f} 95% ci=[{direct_low:.2f}, {direct_high:.2f}]")


if __name__ == "__main__":
    run_experiment()
