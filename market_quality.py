import numpy as np


def prices_from(history):
    return np.array([entry["market_price"] for entry in history])


def fundamentals_from(history):
    return np.array([entry["fundamental_value"] for entry in history])


def returns_from(history):
    prices = prices_from(history)
    return np.diff(prices) / prices[:-1]


def autocorrelation(series, lag):
    series = np.asarray(series)
    if len(series) <= lag:
        return 0.0
    a = series[:-lag] - series[:-lag].mean()
    b = series[lag:] - series[lag:].mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


# per seed, pooling seeds together inflates it
def excess_kurtosis(returns):
    returns = np.asarray(returns)
    std = returns.std()
    if std == 0:
        return 0.0
    return float(((returns - returns.mean()) ** 4).mean() / std ** 4 - 3)


def mispricing(history):
    prices = prices_from(history)
    fundamentals = fundamentals_from(history)
    return (prices - fundamentals) / fundamentals


def mean_abs_gap(history):
    return float(np.mean(np.abs(mispricing(history))) * 100)


# resamples seeds, in chunks of 500 so memory doesnt blow up
def bootstrap_ci(values, confidence=0.95, resamples=10000, seed=0):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 500):
        size = min(500, resamples - start)
        means[start:start + size] = rng.choice(values, size=(size, len(values)), replace=True).mean(axis=1)
    low = float(np.percentile(means, (1 - confidence) / 2 * 100))
    high = float(np.percentile(means, (1 + confidence) / 2 * 100))
    return float(values.mean()), low, high


def bloc_flows(history, names):
    names = list(names)
    flows = {name: [] for name in names}
    for entry in history:
        signed = {name: 0.0 for name in names}
        for action in entry["actions"]:
            if action["agent"] in signed:
                quantity = action["executed_quantity"]
                signed[action["agent"]] = quantity if action["action_type"] == "buy" else -quantity
        for name in names:
            flows[name].append(signed[name])
    return flows


# avg pairwise corr of the copies signed flow
def bloc_flow_correlation(history, names):
    names = list(names)
    if len(names) < 2:
        return 0.0

    flows = bloc_flows(history, names)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = np.array(flows[names[i]])
            b = np.array(flows[names[j]])
            if a.std() > 0 and b.std() > 0:
                pairs.append(np.corrcoef(a, b)[0, 1])
    if pairs:
        return float(np.mean(pairs))
    return 0.0


def bloc_flow_size(history, names):
    names = list(names)
    if not names:
        return 0.0, 0.0

    flows = bloc_flows(history, names)
    total = np.abs(np.array([flows[name] for name in names]).sum(axis=0))
    return float(total.mean()), float(total.max())


# biggest one-direction run vs what the maker holds, holds dont end a run
def bloc_run_demand(history, names, starting_inventory):
    names = list(names)
    if not names:
        return 0.0, 0

    flows = bloc_flows(history, names)
    total = np.array([flows[name] for name in names]).sum(axis=0)

    peak_demand = 0.0
    longest_run = 0
    running = 0.0
    steps = 0
    for value in total:
        if value == 0:
            continue
        if running != 0 and np.sign(value) != np.sign(running):
            running = 0.0
            steps = 0
        running += value
        steps += 1
        peak_demand = max(peak_demand, abs(running))
        longest_run = max(longest_run, steps)

    if starting_inventory:
        percent = 100 * peak_demand / starting_inventory
    else:
        percent = 0.0
    return percent, longest_run


#same rule, only a sign change ends a run
def flow_runs(total):
    runs = []
    running = 0.0
    for value in total:
        if value == 0:
            continue
        if running != 0 and np.sign(value) != np.sign(running):
            runs.append(abs(running))
            running = 0.0
        running += value
    if running != 0:
        runs.append(abs(running))
    return runs
