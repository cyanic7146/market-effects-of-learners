from AGENTS.base_agent import BaseAgent
from values import VALUES


class InformedAgent(BaseAgent):
    def __init__(
        self,
        name,
        cash=VALUES["initial_cash"],
        trade_fraction=VALUES["informed_agent_trade_fraction"],
        hard_cap=VALUES["informed_agent_hard_cap"],
        threshold=VALUES["informed_agent_threshold"],
        signal_noise=VALUES["informed_agent_signal_noise"]
    ):
        super().__init__(name, cash)
        self.trade_fraction = float(trade_fraction)
        self.hard_cap = int(hard_cap)
        self.threshold = float(threshold)
        self.signal_noise = float(signal_noise)




    def act(self, observation):
        price = float(observation["market_price"])
        signal = observation.get("fundamental_signal")
        if signal is None:
            return {
                "type": "hold",
                "quantity": 0
            }



        if signal > price * (1 + self.threshold):
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



        if signal < price * (1 - self.threshold):
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
