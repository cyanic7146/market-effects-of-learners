import os
import random
import numpy as np

from values import VALUES
from PRICE_CALCULATIONS.cycles import OrnsteinUhlenbeckFundamental
from market import Market
from run_experiment import make_agents
from market_quality import autocorrelation


num_sims = 100
steps_per_sim = 1000 # longer runs, steadier acf


def collect_runs(order_flow_impact=None):
    all_returns = []
    all_volumes = []
    for sim in range(num_sims):
        random.seed(1000 + sim)
        agents = make_agents()
        fundamental_process = OrnsteinUhlenbeckFundamental()
        if order_flow_impact is None:
            market = Market(agents, fundamental_process)
        else:
            market = Market(agents, fundamental_process, order_flow_impact=order_flow_impact)
        market.quiet = True

        for i in range(steps_per_sim):
            market.step()

        prices = np.array([entry["market_price"] for entry in market.history])
        returns = np.diff(prices) / prices[:-1]
        volumes = []
        for entry in market.history:
            volumes.append(sum(a["executed_quantity"] for a in entry["actions"]))

        all_returns.append(returns)
        all_volumes.append(np.array(volumes[1:], dtype=float))
    return all_returns, all_volumes


def report(label, all_returns, all_volumes):
    pooled = np.concatenate(all_returns)
    mean_r = pooled.mean()
    std_r = pooled.std()
    skew = ((pooled - mean_r) ** 3).mean() / std_r ** 3
    excess_kurtosis = ((pooled - mean_r) ** 4).mean() / std_r ** 4 - 3

    # acf per sim then averaged so sims dont get mixed together
    lags = [1, 2, 5, 10, 20]
    return_acf = {}
    for lag in lags:
        return_acf[lag] = np.mean([autocorrelation(r, lag) for r in all_returns])
    abs_acf = {}
    for lag in lags:
        abs_acf[lag] = np.mean([autocorrelation(np.abs(r), lag) for r in all_returns])

    vol_corrs = []
    for returns, volumes in zip(all_returns, all_volumes):
        if volumes.std() > 0 and returns.std() > 0:
            vol_corrs.append(np.corrcoef(volumes, np.abs(returns))[0, 1])
    if vol_corrs:
        volume_vol_corr = np.mean(vol_corrs)
    else:
        volume_vol_corr = 0.0

    print(f"\n{label}")
    print(f"  excess kurtosis: {excess_kurtosis:.2f} skew: {skew:.2f}")
    print("  return acf: " + "  ".join(f"lag{lag}={return_acf[lag]:.3f}" for lag in lags))
    print("  |return| acf: " + "  ".join(f"lag{lag}={abs_acf[lag]:.3f}" for lag in lags))
    print(f"  volume vs |return| corr: {volume_vol_corr:.3f}")


print("stylized facts check, real daily equity returns for reference:")
print("  excess kurtosis well above 0 (often 5-30), skew slightly negative")
print("  return acf near 0 at all lags")
print("  |return| acf positive and slowly decaying (volatility clustering)")
print("  volume vs |return| corr positive, roughly 0.3-0.5")

returns, volumes = collect_runs()
report("baseline", returns, volumes)

saved_switch = VALUES["regime_switch_chance"]
saved_length = VALUES["regime_length"]
# regimes off, should kill most of the volatility clustering
VALUES["regime_switch_chance"] = 0.0
VALUES["regime_length"] = 10 ** 9
returns, volumes = collect_runs()
report("no regime switching", returns, volumes)
VALUES["regime_switch_chance"] = saved_switch
VALUES["regime_length"] = saved_length

saved_event = VALUES["event_chance"]
# no events, tails should get thinner
VALUES["event_chance"] = 0.0
returns, volumes = collect_runs()
report("no event shocks", returns, volumes)
VALUES["event_chance"] = saved_event

# no order flow impact, volume/vol link should get weaker
returns, volumes = collect_runs(order_flow_impact=0.0)
report("no order flow impact", returns, volumes)
