import random
from values import VALUES


class Market:
    def __init__(
        self,
        agents,
        fundamental_process,
        initial_price=VALUES["initial_price"],
        fundamental_pull=VALUES["fundamental_pull"],
        order_flow_impact=VALUES["order_flow_impact"],
        market_noise=VALUES["market_noise"],
        transaction_cost_rate=VALUES["transaction_cost_rate"],
        slippage_rate=VALUES["slippage_rate"],
        rng=None
    ):

        self.agents = agents #agents
        self.fundamental_process = fundamental_process #process to simulate fundamental value
        self.initial_price = float(initial_price) #price starts at
        self.price = float(initial_price) #price right now
        self.fundamental_pull = float(fundamental_pull) #how much the price is pulled towards the fundamental value
        self.order_flow_impact = float(order_flow_impact) #base impact of buy-sell before liquidity adjustment
        self.market_noise = float(market_noise) #more noise
        self.transaction_cost_rate = float(transaction_cost_rate) #transaction cost paid by agents to the market
        self.slippage_rate = float(slippage_rate) #base slippage before liquidity adjustment
        self.step_count = 0 #how many steps
        self.history = [] #history of market maybe for rl agent

        self.market_starting_cash = VALUES["market_base_cash"] + VALUES["market_cash_per_agent"] * len(self.agents)
        self.market_starting_inventory = VALUES["market_base_inventory"] + VALUES["market_inventory_per_agent"] * len(self.agents)
        self.market_cash = float(self.market_starting_cash)
        self.market_inventory = int(self.market_starting_inventory)
        self.regime = "mean_reverting"
        self.quiet = False
        self.rng = rng if rng is not None else random



    def reset(self, initial_cash=VALUES["initial_cash"]):
        self.price = float(self.initial_price)
        self.step_count = 0
        self.history = []
        self.fundamental_process.reset()
        self.market_cash = float(self.market_starting_cash)
        self.market_inventory = int(self.market_starting_inventory)
        self.regime = "mean_reverting"
        for agent in self.agents:
            agent.reset(initial_cash)




    def build_observation(self):
        price_history = [entry["market_price"] for entry in self.history]

        if len(self.history) > 0:
            last_net_order_flow = self.history[-1]["net_order_flow"]

            last_volume = 0
            for action in self.history[-1]["actions"]:
                last_volume += action["executed_quantity"]
        else:
            last_net_order_flow = 0
            last_volume = 0

        return {
            "market_price": self.price,
            "price_history": price_history,
            "step": self.step_count,
            "last_net_order_flow": last_net_order_flow,
            "last_volume": last_volume,
        }


    def step(self):
        if self.step_count > 0 and self.step_count % VALUES["regime_length"] == 0:
            old_regime = self.regime
            self.regime = self.rng.choice(["mean_reverting", "trending", "volatile"])
            if self.regime != old_regime and not self.quiet:
                print(f"Step {self.step_count}: regime changed to {self.regime}")
        elif self.rng.random() < VALUES["regime_switch_chance"]:
            old_regime = self.regime
            self.regime = self.rng.choice(["mean_reverting", "trending", "volatile"])
            if self.regime != old_regime and not self.quiet:
                print(f"Step {self.step_count}: regime changed to {self.regime}")

        fundamental_value = self.fundamental_process.step()

        if self.regime == "volatile" and self.rng.random() < VALUES["event_chance"] * 0.5:
            fundamental_value += self.rng.gauss(0, VALUES["event_impact_std"])

        observation = self.build_observation()
        net_order_flow = 0
        actions_log = []

        half_spread = 0.0
        inventory_skew = 0.0
        if VALUES["market_spread_enabled"]:
            recent_volatility = 0.0
            vol_lookback = VALUES["spread_vol_lookback"]
            if len(self.history) >= vol_lookback + 1:
                vol_prices = []
                for entry in self.history[-(vol_lookback + 1):]:
                    vol_prices.append(entry["market_price"])
                vol_returns = [(vol_prices[i] - vol_prices[i-1]) / vol_prices[i-1] for i in range(1, len(vol_prices))]
                mean_return = sum(vol_returns) / len(vol_returns)
                recent_volatility = (sum((r - mean_return) ** 2 for r in vol_returns) / len(vol_returns)) ** 0.5
            half_spread = VALUES["market_base_half_spread"] + VALUES["spread_vol_sensitivity"] * recent_volatility
            inventory_skew = VALUES["inventory_skew_strength"] * (self.market_inventory - self.market_starting_inventory) / self.market_starting_inventory

        acting_agents = self.agents
        if VALUES["shuffle_agent_order"]:
            acting_agents = self.agents[:]
            random.shuffle(acting_agents)

        working_price = self.price

        for agent in acting_agents:
            agent_observation = observation
            # informed agents get a noisy peek at the fundamental
            if hasattr(agent, "signal_noise"):
                agent_observation = dict(observation)
                agent_observation["fundamental_signal"] = fundamental_value + random.gauss(0, agent.signal_noise)

            if agent.position < 0:
                borrow_fee = VALUES["borrow_rate_per_step"] * (-agent.position) * self.price
                borrow_fee = min(borrow_fee, max(0.0, agent.cash))
                agent.cash -= borrow_fee
                self.market_cash += borrow_fee

            forced_cover = 0
            if agent.position < 0:
                equity = agent.cash + agent.position * self.price
                if equity < VALUES["margin_maintenance"] * (-agent.position) * self.price:
                    forced_cover = -agent.position

            if forced_cover > 0:
                action = {"type": "buy", "quantity": forced_cover}
            else:
                action = agent.act(agent_observation)

            action_type = action.get("type", "hold")
            requested_quantity = int(action.get("quantity", 0))
            if action_type not in ("buy", "sell", "hold"):
                action_type = "hold"
            quantity = requested_quantity
            executed_quantity = 0
            transaction_cost = 0.0
            if VALUES["intra_step_impact"]:
                base_price = working_price
            else:
                base_price = self.price
            fill_price = base_price
            dynamic_slippage_rate = self.slippage_rate


            if action_type == "buy":
                quantity = min(quantity, self.market_inventory)

                if quantity > 0:
                    inventory_liquidity = max(1, self.market_inventory)
                    trade_size_ratio = quantity / inventory_liquidity
                    dynamic_slippage_rate = self.slippage_rate * (1 + trade_size_ratio * VALUES["slippage_liquidity_multiplier"])

                    fill_price = base_price * (1 + half_spread - inventory_skew + dynamic_slippage_rate * quantity)
                    max_agent_can_buy = int(agent.cash // (fill_price * (1 + self.transaction_cost_rate)))
                    quantity = min(quantity, max_agent_can_buy)

                if quantity > 0:
                    inventory_liquidity = max(1, self.market_inventory)
                    trade_size_ratio = quantity / inventory_liquidity
                    dynamic_slippage_rate = self.slippage_rate * (1 + trade_size_ratio * VALUES["slippage_liquidity_multiplier"])

                    fill_price = base_price * (1 + half_spread - inventory_skew + dynamic_slippage_rate * quantity)
                    executed_quantity = agent.execute_buy(quantity, fill_price)
                    transaction_cost = executed_quantity * fill_price * self.transaction_cost_rate

                    agent.cash -= transaction_cost
                    self.market_cash += executed_quantity * fill_price + transaction_cost
                    self.market_inventory -= executed_quantity

                    net_order_flow += executed_quantity
                    if VALUES["intra_step_impact"]:
                        working_price += self.price * self.order_flow_impact * (executed_quantity / max(1, self.market_inventory)) * VALUES["order_flow_liquidity_multiplier"]
            elif action_type == "sell":
                if quantity > 0:
                    cash_liquidity_shares = max(1, int(self.market_cash // self.price))
                    trade_size_ratio = quantity / cash_liquidity_shares
                    dynamic_slippage_rate = self.slippage_rate * (1 + trade_size_ratio * VALUES["slippage_liquidity_multiplier"])

                    fill_price = base_price * (1 - half_spread - inventory_skew - dynamic_slippage_rate * quantity)

                    if fill_price < 0.01:
                        fill_price = 0.01

                    max_market_can_buy = int(self.market_cash // fill_price)
                    quantity = min(quantity, max_market_can_buy)

                if quantity > 0:
                    cash_liquidity_shares = max(1, int(self.market_cash // self.price))
                    trade_size_ratio = quantity / cash_liquidity_shares
                    dynamic_slippage_rate = self.slippage_rate * (1 + trade_size_ratio * VALUES["slippage_liquidity_multiplier"])

                    fill_price = base_price * (1 - half_spread - inventory_skew - dynamic_slippage_rate * quantity)

                    if fill_price < 0.01:
                        fill_price = 0.01

                    executed_quantity = agent.execute_sell(quantity, fill_price)
                    transaction_cost = executed_quantity * fill_price * self.transaction_cost_rate

                    agent.cash -= transaction_cost
                    self.market_cash -= executed_quantity * fill_price
                    self.market_cash += transaction_cost
                    self.market_inventory += executed_quantity

                    net_order_flow -= executed_quantity
                    if VALUES["intra_step_impact"]:
                        working_price -= self.price * self.order_flow_impact * (executed_quantity / max(1, self.market_inventory)) * VALUES["order_flow_liquidity_multiplier"]


            actions_log.append({
                "agent": agent.name,
                "action_type": action_type,
                "requested_quantity": requested_quantity,
                "executed_quantity": executed_quantity,
                "fill_price": fill_price,
                "transaction_cost": transaction_cost,
                "dynamic_slippage_rate": dynamic_slippage_rate,
            })

        noise_std = self.market_noise * 2 if self.regime == "volatile" else self.market_noise
        market_noise = self.rng.gauss(0, noise_std)

        if VALUES["intra_step_impact"]:
            liquidity_adjusted_order_flow_impact = working_price - self.price
        else:
            liquidity = max(1, self.market_inventory)
            order_flow_ratio = net_order_flow / liquidity
            liquidity_adjusted_order_flow_impact = self.price * self.order_flow_impact * order_flow_ratio * VALUES["order_flow_liquidity_multiplier"]

        effective_pull = self.fundamental_pull * 0.5 if self.regime == "trending" else self.fundamental_pull

        trend_effect = 0.0
        if self.regime == "trending" and len(self.history) >= 1:
            prev2 = self.history[-2]["market_price"] if len(self.history) >= 2 else self.initial_price
            trend_effect = (self.price - prev2) * VALUES["trend_strength"]

        self.price = (
            self.price
            + effective_pull * (fundamental_value - self.price)
            + liquidity_adjusted_order_flow_impact
            + trend_effect
            + market_noise
        )

        if self.price < 0.01:
            self.price = 0.01

        if VALUES["dividend_yield_per_step"] > 0:
            for agent in self.agents:
                dividend = VALUES["dividend_yield_per_step"] * fundamental_value * agent.position
                agent.cash += dividend
                self.market_cash -= dividend

        if VALUES["risk_free_rate_per_step"] > 0:
            for agent in self.agents:
                interest = VALUES["risk_free_rate_per_step"] * max(0.0, agent.cash)
                agent.cash += interest
                self.market_cash -= interest

        self.history.append({
            "step": self.step_count,
            "market_price": self.price,
            "fundamental_value": fundamental_value,
            "net_order_flow": net_order_flow,
            "actions": actions_log,
            "market_cash": self.market_cash,
            "market_inventory": self.market_inventory,
            "liquidity_adjusted_order_flow_impact": liquidity_adjusted_order_flow_impact,
            "regime": self.regime,
            "half_spread": half_spread,
        })
        self.step_count += 1
        return self.step_count, self.price, fundamental_value, net_order_flow, actions_log



    def run(self, num_steps=VALUES["num_steps"], times_to_print=VALUES["times_to_print"]):
        
        times_to_print = int(times_to_print)
        print_interval = (num_steps // times_to_print) if times_to_print > 0 else num_steps + 1
        for i in range(num_steps):
            self.step()
            if (i + 1) % print_interval == 0:
                print(f"Step {i + 1}: Market price = {self.price:.2f}, Fundamental value = {self.fundamental_process.value:.2f}")
                print(f"Market cash = {self.market_cash:.2f}, Market inventory = {self.market_inventory}")
                for agent in self.agents:
                    agentstate = agent.get_state(self.price)
                    print(
                        f"  {agentstate['name']}: "
                        f"position={agentstate['position']}, "
                        f"portfolio_value={agentstate['portfolio_value']:.2f}"
                    )

    def print_results(self):
        print(f"Final market price: {self.price:.2f}")
        print(f"Final fundamental value: {self.fundamental_process.value:.2f}")
        print(f"Final market cash: {self.market_cash:.2f}")
        print(f"Final market inventory: {self.market_inventory}")
        print()
        print("Agents:")
        for agent in self.agents:
            agentstate = agent.get_state(self.price)
            print(
                f"{agentstate['name']}: "
                f"cash={agentstate['cash']:.2f}, "
                f"position={agentstate['position']}, "
                f"portfolio_value={agentstate['portfolio_value']:.2f}"
            )