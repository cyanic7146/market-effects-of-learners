import random
from AGENTS.base_agent import BaseAgent
from values import VALUES


class RandomAgent(BaseAgent):
    def __init__(self, name, cash=VALUES["initial_cash"], trade_fraction=VALUES["random_agent_trade_fraction"], hard_cap=VALUES["random_agent_hard_cap"]):
        super().__init__(name, cash)
        self.trade_fraction = float(trade_fraction)
        self.hard_cap = int(hard_cap)




    def act(self, observation):
        price = float(observation["market_price"])
        action_type = random.choice(["buy", "sell", "hold"])

        if action_type == "hold":
            return {
                "type": "hold",
                "quantity": 0
            }
        if action_type == "buy":
            affordable_quantity = int(self.cash // price)
            if affordable_quantity <= 0:
                return {
                    "type": "hold",
                    "quantity": 0
                }
            max_quantity = max(1, int(self.trade_fraction * affordable_quantity))
            max_quantity = min(max_quantity, self.hard_cap, affordable_quantity)
            quantity = random.randint(1, max_quantity)




            return {
                "type": "buy",
                "quantity": quantity
            }




        if action_type == "sell":
            if self.position <= 0:
                return {
                    "type": "hold",
                    "quantity": 0
                }
            max_quantity = max(1, int(self.trade_fraction * self.position))
            max_quantity = min(max_quantity, self.hard_cap, self.position)

            quantity = random.randint(1, max_quantity)

            return {
                "type": "sell",
                "quantity": quantity
            }

        return {
            "type": "hold",
            "quantity": 0
        }