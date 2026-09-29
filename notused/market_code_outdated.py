from values import VALUES

class Market:
    def __init__(self, initial_price=VALUES["mean_price"]):
        self.price = float(initial_price)

    def ou_equation_simulate(self, price, quantity): #Ornstein-Uhlenbeck process
    #maybe choose GBM instead but OU is more mean reverting which might be better at calculating
    #the impact of the agent's actions through obervations
        mean_price = VALUES["mean_price"]
        theta = VALUES["mean_reversion_speed"]
        sigma = VALUES["price_volatility"]

        price_change = theta * (mean_price - price) + sigma * np.random.normal()
        new_price = price + price_change

        return max(new_price, 0.001)
#don't know if I will use this, don't know if I wil obtain decision in market.py or cycles.py
    def obtain_decision(self, agent):
        observation = self.get_observation()
        action = agent.act(observation)
        return action



    def market_step(self, action):
        self.price = ou_equation_simulate(self.price, self.position)

        if action > 0:
            executed_quantity = self.execute_buy(action, self.price)
        elif action < 0:
            executed_quantity = self.execute_sell(-action, self.price)
        else:
            executed_quantity = 0
        return executed_quantity


    def get_observation(self):
        return {
            "state": self.get_state(),
            "price": self.price,
            ""
        }