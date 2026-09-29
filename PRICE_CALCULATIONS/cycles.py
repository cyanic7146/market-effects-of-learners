import random
from values import VALUES


class OrnsteinUhlenbeckFundamental:
    def __init__(
        self,
        mu=VALUES["ou_mu"],
        theta=VALUES["ou_theta"],
        sigma=VALUES["ou_sigma"],
        initial_value=VALUES["initial_price"],
        rng=None
    ):
        self.mu = float(mu) # long-term mean price
        self.theta = float(theta)# speed of reversion
        self.sigma = float(sigma) # volatility of price
        self.value = float(initial_value) # current price
        self.rng = rng if rng is not None else random

    def step(self):
        shock = self.rng.gauss(0, 1) #randon.random would be uniform, we want normal distribution to make high shock unlikely
        self.value =self.value + self.theta *(self.mu-self.value) + self.sigma *shock

        if self.rng.random() < VALUES["event_chance"]:
            event_impact = self.rng.gauss(0, VALUES["event_impact_std"])
            self.value += event_impact

        return self.value



    def reset(self, value=VALUES["initial_price"]):
        self.value = float(value)
