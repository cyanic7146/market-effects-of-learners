from AGENTS.base_agent import BaseAgent
from values import VALUES


class MeanReversionAgent(BaseAgent):
    def __init__(
        self,
        name,
        cash=VALUES["initial_cash"],
        trade_fraction=VALUES["mean_reversion_agent_trade_fraction"],
        hard_cap=VALUES["mean_reversion_agent_hard_cap"],
        lookback=VALUES["mean_reversion_agent_lookback"],
        threshold=VALUES["mean_reversion_agent_threshold"]
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
        moving_average = sum(recent_prices) / len(recent_prices)
        lower_bound = moving_average * (1 - self.threshold)
        upper_bound = moving_average * (1 + self.threshold)



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