import random
import numpy as np

from values import VALUES
from PRICE_CALCULATIONS.cycles import OrnsteinUhlenbeckFundamental
from market import Market
from run_experiment import make_agents
from AGENTS.informed_agent import InformedAgent


seeds = list(range(1, 1001))
informed_counts = [0, 1, 3, 6]
signal_noises = [1.0, 2.0, 3.0, 1.5, 2.5, 0.8]


starting_cash = VALUES["initial_cash"]
print(f"information experiment over {len(seeds)} seeds")

for informed_count in informed_counts:
    informed_names = [f"Informed{i + 1}" for i in range(informed_count)]
    agents = make_agents() + [InformedAgent(f"Informed{i + 1}", signal_noise=signal_noises[i]) for i in range(informed_count)]
    fundamental_process = OrnsteinUhlenbeckFundamental()
    market = Market(agents, fundamental_process)
    market.quiet = True

    price_fund_corrs = []
    average_gaps = []
    informed_finals = []
    background_finals = []

    for seed in seeds:
        random.seed(seed)
        market.reset()
        for i in range(VALUES["num_steps"]):
            market.step()

        prices = np.array([entry["market_price"] for entry in market.history])
        fundamentals = np.array([entry["fundamental_value"] for entry in market.history])

        if prices.std() > 0 and fundamentals.std() > 0:
            price_fund_corrs.append(np.corrcoef(prices, fundamentals)[0, 1])
        average_gaps.append(np.mean(np.abs(prices - fundamentals) / fundamentals) * 100)

        for agent in agents:
            final = agent.portfolio_value(market.price)
            if agent.name in informed_names:
                informed_finals.append(final)
            else:
                background_finals.append(final)

    if price_fund_corrs:
        corr = np.mean(price_fund_corrs)
    else:
        corr = 0.0
    gap = np.mean(average_gaps)
    background_return = (np.mean(background_finals) - starting_cash) / starting_cash * 100

    line = f"  {informed_count} informed: price-fundamental corr={corr:.3f}  avg gap={gap:.2f}%  background return={background_return:.2f}%"
    if informed_count > 0:
        informed_return = (np.mean(informed_finals) - starting_cash) / starting_cash * 100
        line += f"  informed return={informed_return:.2f}%"
    print(line)
