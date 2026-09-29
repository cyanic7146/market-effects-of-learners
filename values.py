import os

risky = True # set to False for calmer market, True for riskier market
VALUES = {
    "initial_cash": 10000.0, # starting cash for each agent
    "initial_price": 100.0, # starting market price
    "num_steps": 300, # how many steps of the market simulate
    #all OU parameters
    "ou_mu": 100.0, # what the mean price should return to in OU
    "ou_theta": 0.05, # how quickly the price reverts to the mean in OU
    "ou_sigma": 1.0, # volatitil of fundamental value
    #market parameters
    "fundamental_pull": 0.08, # how much price is towards fundamental value each step
    "order_flow_impact": 0.03, # base impact of buy-sell orders before liquidity adjustment
    "market_noise": 0.3, # how much noise is added to market price
    "transaction_cost_rate": 0.001, # percent cost per trade, 0.001 = 0.1%
    "slippage_rate": 0.00005, # base slippage before liquidity adjustment
    "slippage_liquidity_multiplier": 10.0, # how much slippage increases when liquidity is low
    "order_flow_liquidity_multiplier": 1.0, # how much order flow impact depends on liquidity
    # cash-conserving market parameters
    "market_base_cash": 700000.0, # base cash held by the market/liquidity provider
    "market_cash_per_agent": 70000.0, # extra market cash added per agent
    "market_base_inventory": 2000, # base stock inventory held by the market/liquidity provider
    "market_inventory_per_agent": 350, # extra stock inventory added per agent
    #random agent parameters
    "random_agent_trade_fraction": 0.30, # max fraction of cash or position that random agent trades each step
    "random_agent_hard_cap": 100, # max amount each agent can trade
    #momentum agent parameters
    "momentum_agent_trade_fraction": 0.25, # max fraction of cash or position that momentum agent trades each step
    "momentum_agent_hard_cap": 100, # max amount each momentum agent can trade
    "momentum_agent_lookback": 5, # how many past prices the momentum agent checks
    # mean reversion agent parameters
    "mean_reversion_agent_trade_fraction": 0.25, # max fraction of cash or position that mean reversion agent trades each step
    "mean_reversion_agent_hard_cap": 100, # max amount each mean reversion agent can trade
    "mean_reversion_agent_lookback": 20, # how many past prices the mean reversion agent checks
    "mean_reversion_agent_threshold": 0.04, # how far current price must be from moving average for mean reversion agent to act
    # buy and hold agent parameters
    "buy_and_hold_agent_buy_fraction": 0.80, # what fraction of cash the buy and hold agent uses to buy at the start
    # contrarian agent parameters
    "contrarian_agent_trade_fraction": 0.20,
    "contrarian_agent_hard_cap": 100,
    "contrarian_agent_threshold": 0.008, # last-step return must exceed this to trigger trade
    # volume pressure agent parameters
    "volume_pressure_agent_trade_fraction": 0.20,
    "volume_pressure_agent_hard_cap": 100,
    "volume_pressure_agent_sensitivity": 0.3, # net_order_flow/volume ratio needed to act
    # value estimate agent parameters
    "value_estimate_agent_trade_fraction": 0.20,
    "value_estimate_agent_hard_cap": 100,
    "value_estimate_agent_lookback": 60, # long window for fair value estimate
    "value_estimate_agent_threshold": 0.03,
    # risk managed agent parameters
    "risk_managed_agent_trade_fraction": 0.25,
    "risk_managed_agent_hard_cap": 100,
    "risk_managed_agent_lookback": 20,
    "risk_managed_agent_threshold": 0.025,
    "risk_managed_agent_max_position": 300, # hard position ceiling
    "risk_managed_agent_vol_lookback": 10,
    "risk_managed_agent_vol_floor": 0.25, # minimum vol scaling factor
    "risk_managed_agent_vol_sensitivity": 0.05, # vol level that triggers max reduction
    "risk_managed_agent_drawdown_limit": 0.20, # portfolio drawdown that stops buying
    # rl agent parameters
    "rl_agent_trade_fraction": 0.25,
    "rl_agent_hard_cap": 100,
    "rl_num_steps": 300, # episode length, matches regular sim
    "rl_history_len": 20, # how many past prices the rl agent sees
    # market regime parameters
    "regime_length": 50, # steps between deterministic regime switches
    "trend_strength": 0.15, # how much last price change carries forward in trending regime
    "regime_switch_chance": 0.05, # per-step probability of random regime switch
    "times_to_print": 5, # how many times to print market status during the run
    "event_chance": 0.03, # probability of an event each step
    "event_impact_std": 8.0, # how big impact events

    "shuffle_agent_order": True,
    "intra_step_impact": False,
    "market_spread_enabled": False,
    "market_base_half_spread": 0.0015,
    "spread_vol_sensitivity": 0.5,
    "spread_vol_lookback": 10,
    "inventory_skew_strength": 0.02,
    "dividend_yield_per_step": 0.0,
    "risk_free_rate_per_step": 0.0,
    "allow_short": False,
    "short_cap": 100,
    "borrow_rate_per_step": 0.0002,
    "margin_maintenance": 0.5,
    "informed_agent_trade_fraction": 0.20,
    "informed_agent_hard_cap": 100,
    "informed_agent_threshold": 0.02,
    "informed_agent_signal_noise": 2.0,




    "seed": 676767
}

if risky:
    VALUES.update({
        # riskier OU parameters
        "ou_mu": 100.0,
        "ou_theta": 0.03,
        "ou_sigma": 1.5,

        # riskier market parameters
        "fundamental_pull": 0.04,
        "order_flow_impact": 0.04,
        "market_noise": 0.7,
        "transaction_cost_rate": 0.001,
        "slippage_rate": 0.00008,
        "slippage_liquidity_multiplier": 12.0,
        "order_flow_liquidity_multiplier": 1.2,
        "event_chance": 0.04,
        "event_impact_std": 10.0,
    })
else:
    VALUES.update({
        # calmer OU parameters
        "ou_mu": 100.0,
        "ou_theta": 0.05,
        "ou_sigma": 0.8,

        # calmer market parameters
        "fundamental_pull": 0.06,
        "order_flow_impact": 0.02,
        "market_noise": 0.25,
        "transaction_cost_rate": 0.0005,
        "slippage_rate": 0.00003,
        "slippage_liquidity_multiplier": 8.0,
        "order_flow_liquidity_multiplier": 0.8,
        "event_chance": 0.015,
        "event_impact_std": 5.0,
    })
