from AGENTS.base_agent import BaseAgent
from values import VALUES


class VolumePressureAgent(BaseAgent):
    def __init__(
        self,
        name,
        cash=VALUES["initial_cash"],
        trade_fraction=VALUES["volume_pressure_agent_trade_fraction"],
        hard_cap=VALUES["volume_pressure_agent_hard_cap"],
        sensitivity=VALUES["volume_pressure_agent_sensitivity"]
    ):
        super().__init__(name, cash)
        self.trade_fraction = float(trade_fraction)
        self.hard_cap = int(hard_cap)
        self.sensitivity = float(sensitivity)  # sensitivity is pretty market dependent, might need retuning




    def act(self, observation):
        price = float(observation["market_price"])
        last_net_order_flow = observation["last_net_order_flow"]
        last_volume = observation["last_volume"]

        if last_volume <= 0:
            return {
                "type": "hold",
                "quantity": 0
            }



        buy_pressure = last_net_order_flow / last_volume



        if buy_pressure > self.sensitivity:
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



        if buy_pressure < -self.sensitivity:
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
