from AGENTS.base_agent import BaseAgent
from values import VALUES


class BuyAndHoldAgent(BaseAgent):
    def __init__(
        self,
        name,
        cash=VALUES["initial_cash"],
        buy_fraction=VALUES["buy_and_hold_agent_buy_fraction"]
    ):
        super().__init__(name, cash)
        self.buy_fraction = float(buy_fraction)
        self.has_bought = False



    def act(self, observation):
        price = float(observation["market_price"])
        if self.has_bought:
            return {
                "type": "hold",
                "quantity": 0
            }



        affordable_quantity = int(self.cash // price)
        quantity = int(affordable_quantity * self.buy_fraction)
        if quantity <= 0:
            return {
                "type": "hold",
                "quantity": 0
            }
        self.has_bought = True



        return {
            "type": "buy",
            "quantity": quantity
        }

    def reset(self, cash=VALUES["initial_cash"]):
        super().reset(cash)
        self.has_bought = False