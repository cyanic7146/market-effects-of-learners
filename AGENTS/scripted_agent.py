from AGENTS.base_agent import BaseAgent
from values import VALUES


class ScriptedAgent(BaseAgent):
    def __init__(self, name, schedule, cash=VALUES["initial_cash"]):
        super().__init__(name, cash)
        self.schedule = list(schedule)
        self.step_index = 0




    def act(self, observation):
        if self.step_index >= len(self.schedule):
            return {"type": "hold", "quantity": 0}

        quantity = int(self.schedule[self.step_index])
        self.step_index += 1

        if quantity > 0:
            return {"type": "buy", "quantity": quantity}

        if quantity < 0:
            return {"type": "sell", "quantity": -quantity}

        return {"type": "hold", "quantity": 0}




    def reset(self, cash=VALUES["initial_cash"]):
        super().reset(cash)
        self.step_index = 0
