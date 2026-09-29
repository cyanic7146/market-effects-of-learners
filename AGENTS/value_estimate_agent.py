from AGENTS.base_agent import BaseAgent
from values import VALUES


class ValueEstimateAgent(BaseAgent):
    def __init__(
        self,
        name,
        cash=VALUES["initial_cash"],
        trade_fraction=VALUES["value_estimate_agent_trade_fraction"],
        hard_cap=VALUES["value_estimate_agent_hard_cap"],
        lookback=VALUES["value_estimate_agent_lookback"],
        threshold=VALUES["value_estimate_agent_threshold"]
    ):
        super().__init__(name, cash)
        self.trade_fraction = float(trade_fraction)
        self.hard_cap = int(hard_cap)
        self.lookback = int(lookback)
        self.threshold = float(threshold)




    def act(self, observation):
        price = float(observation["market_price"])
        price_history = observation["price_history"]
        if len(price_history) < self.lookback:
            return {
                "type": "hold",
                "quantity": 0
            }



        recent_prices = price_history[-self.lookback:]
        # long lookback makes fair_value slow to change, which is the whole point
        fair_value = sum(recent_prices) / len(recent_prices)
        lower_bound = fair_value * (1 - self.threshold)
        upper_bound = fair_value * (1 + self.threshold)



        if price < lower_bound:
            affordable_quantity = int(self.cash // price)

            if affordable_quantity <= 0:
                return {
                    "type": "hold",
                    "quantity": 0
                }
            max_quantity = max(1, int(self.trade_fraction * affordable_quantity))
            max_quantity = min(max_quantity, self.hard_cap, affordable_quantity)
            return {
                "type": "buy",
                "quantity": max_quantity
            }



        if price > upper_bound:
            sellable_quantity = self.sell_capacity()
            if sellable_quantity <= 0:
                return {
                    "type": "hold",
                    "quantity": 0
                }
            max_quantity = max(1, int(self.trade_fraction * sellable_quantity))
            max_quantity = min(max_quantity, self.hard_cap, sellable_quantity)
            return {
                "type": "sell",
                "quantity": max_quantity
            }

        return {
            "type": "hold",
            "quantity": 0
        }
