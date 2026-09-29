from AGENTS.base_agent import BaseAgent
from values import VALUES


class MomentumAgent(BaseAgent):
    def __init__(
        self,
        name,
        cash=VALUES["initial_cash"],
        trade_fraction=VALUES["momentum_agent_trade_fraction"],
        hard_cap=VALUES["momentum_agent_hard_cap"],
        lookback=VALUES["momentum_agent_lookback"]
    ):
        super().__init__(name, cash)
        self.trade_fraction = float(trade_fraction)
        self.hard_cap = int(hard_cap)
        self.lookback = int(lookback)






    def act(self, observation):
        price = float(observation["market_price"])
        price_history = observation["price_history"]
        if len(price_history) < self.lookback:
            return {
                "type": "hold",
                "quantity": 0
            }



        old_price = price_history[-self.lookback]
        if price > old_price:
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





        if price < old_price:
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