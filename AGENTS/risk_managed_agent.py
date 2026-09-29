from AGENTS.base_agent import BaseAgent
from values import VALUES


class RiskManagedAgent(BaseAgent):
    def __init__(
        self,
        name,
        cash=VALUES["initial_cash"],
        trade_fraction=VALUES["risk_managed_agent_trade_fraction"],
        hard_cap=VALUES["risk_managed_agent_hard_cap"],
        lookback=VALUES["risk_managed_agent_lookback"],
        threshold=VALUES["risk_managed_agent_threshold"],
        max_position=VALUES["risk_managed_agent_max_position"],
        vol_lookback=VALUES["risk_managed_agent_vol_lookback"],
        vol_floor=VALUES["risk_managed_agent_vol_floor"],
        vol_sensitivity=VALUES["risk_managed_agent_vol_sensitivity"],
        drawdown_limit=VALUES["risk_managed_agent_drawdown_limit"]
    ):
        super().__init__(name, cash)
        self.trade_fraction = float(trade_fraction)
        self.hard_cap = int(hard_cap)
        self.lookback = int(lookback)
        self.threshold = float(threshold)
        self.max_position = int(max_position)
        self.vol_lookback = int(vol_lookback)
        self.vol_floor = float(vol_floor)
        self.vol_sensitivity = float(vol_sensitivity)
        self.drawdown_limit = float(drawdown_limit)
        self.initial_cash = float(cash)




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

        vol_prices = price_history[-self.vol_lookback:] if len(price_history) >= self.vol_lookback else price_history[:]
        if len(vol_prices) > 1:
            returns = [(vol_prices[i] - vol_prices[i-1]) / vol_prices[i-1] for i in range(1, len(vol_prices))]
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = variance ** 0.5
        else:
            volatility = 0.0

        # vol_dampener shrinks trade size in choppy markets, can make this very passive
        vol_dampener = max(self.vol_floor, 1.0 - volatility / self.vol_sensitivity)

        portfolio_value = self.cash + self.position * price
        drawdown = (portfolio_value - self.initial_cash) / self.initial_cash



        if price < lower_bound and drawdown > -self.drawdown_limit:
            if self.position >= self.max_position:
                return {
                    "type": "hold",
                    "quantity": 0
                }
            affordable_quantity = int(self.cash // price)

            if affordable_quantity <= 0:
                return {
                    "type": "hold",
                    "quantity": 0
                }
            max_quantity = max(1, int(self.trade_fraction * vol_dampener * affordable_quantity))
            max_quantity = min(max_quantity, self.hard_cap, self.max_position - self.position, affordable_quantity)
            if max_quantity <= 0:
                return {
                    "type": "hold",
                    "quantity": 0
                }
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
            max_quantity = max(1, int(self.trade_fraction * vol_dampener * sellable_quantity))
            max_quantity = min(max_quantity, self.hard_cap, sellable_quantity)
            return {
                "type": "sell",
                "quantity": max_quantity
            }

        return {
            "type": "hold",
            "quantity": 0
        }


    def reset(self, cash=VALUES["initial_cash"]):
        super().reset(cash)
        self.initial_cash = float(cash)
