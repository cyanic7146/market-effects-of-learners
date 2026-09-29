from values import VALUES


class BaseAgent:
    def __init__(self, name, cash=VALUES["initial_cash"]):
        self.name = name
        self.cash = float(cash)
        self.position = 0


    def act(self, observation):
        pass




    def execute_buy(self, quantity, price):
        quantity = int(quantity)
        price = float(price)
        if quantity <= 0 or price <= 0:
            return 0
        affordable_quantity = int(self.cash // price)
        quantity = min(quantity, affordable_quantity)


        if quantity <= 0:
            return 0

        self.cash -= quantity * price
        self.position += quantity
        return quantity




    def sell_capacity(self):
        if VALUES["allow_short"]:
            return self.position + VALUES["short_cap"]
        return self.position




    def execute_sell(self, quantity, price):
        quantity = int(quantity)
        price = float(price)
        if quantity <= 0 or price <= 0:
            return 0
        quantity = min(quantity, self.sell_capacity())


        if quantity <= 0:
            return 0

        self.cash += quantity * price
        self.position -= quantity
        return quantity

    def portfolio_value(self, current_price):
        return self.cash + self.position * float(current_price)



    def get_state(self, current_price):
        return {
            "name": self.name,
            "cash": self.cash,
            "position": self.position,
            "portfolio_value": self.portfolio_value(current_price),
        }



    def reset(self, cash=VALUES["initial_cash"]):
        self.cash = float(cash)
        self.position = 0