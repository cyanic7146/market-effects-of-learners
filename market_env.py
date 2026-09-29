import random
import numpy as np
import gymnasium

from values import VALUES
from market import Market
from PRICE_CALCULATIONS.cycles import OrnsteinUhlenbeckFundamental
from AGENTS.base_agent import BaseAgent
from AGENTS.random_agent import RandomAgent
from AGENTS.momentum_agent import MomentumAgent
from AGENTS.mean_reversion_agent import MeanReversionAgent
from AGENTS.buy_and_hold_agent import BuyAndHoldAgent
from AGENTS.contrarian_agent import ContrarianAgent
from AGENTS.volume_pressure_agent import VolumePressureAgent
from AGENTS.value_estimate_agent import ValueEstimateAgent
from AGENTS.risk_managed_agent import RiskManagedAgent


# agent the env drives, act() just returns next_action
class RLEnvAgent(BaseAgent):
    def __init__(self, name, cash=VALUES["initial_cash"], trade_fraction=VALUES["rl_agent_trade_fraction"], hard_cap=VALUES["rl_agent_hard_cap"]):
        super().__init__(name, cash)
        self.trade_fraction = float(trade_fraction)
        self.hard_cap = int(hard_cap)
        self.next_action = "hold"
        self.active_fraction = self.trade_fraction  # reset so it doesn't carry over




    def act(self, observation):
        active_fraction = self.active_fraction
        self.active_fraction = self.trade_fraction
        price = float(observation["market_price"])
        action_type = self.next_action

        if action_type == "buy":
            affordable_quantity = int(self.cash // price)
            if affordable_quantity <= 0:
                return {"type": "hold", "quantity": 0}
            max_quantity = max(1, int(active_fraction * affordable_quantity))
            max_quantity = min(max_quantity, self.hard_cap, affordable_quantity)
            return {"type": "buy", "quantity": max_quantity}

        if action_type == "sell":
            sellable_quantity = self.sell_capacity()
            if sellable_quantity <= 0:
                return {"type": "hold", "quantity": 0}
            max_quantity = max(1, int(active_fraction * sellable_quantity))
            max_quantity = min(max_quantity, self.hard_cap, sellable_quantity)
            return {"type": "sell", "quantity": max_quantity}

        return {"type": "hold", "quantity": 0}


class MarketEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()

        self.action_space = gymnasium.spaces.Discrete(7)  # 0=hold 1=buy_s 2=buy_m 3=buy_l 4=sell_s 5=sell_m 6=sell_l

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(31,),  # price, 20 history, position, cash, portfolio, flow, 6 signals
            dtype=np.float32
        )

        self.initial_cash = VALUES["initial_cash"]
        self.num_steps = VALUES["rl_num_steps"]

        self.rl_agent = RLEnvAgent("RLAgent")

        background_agents = [
            RandomAgent("Random1", trade_fraction=0.20, hard_cap=80),
            RandomAgent("Random2", trade_fraction=0.30, hard_cap=100),
            RandomAgent("Random3", trade_fraction=0.40, hard_cap=120),
            MomentumAgent("Momentum1", lookback=3, trade_fraction=0.15, hard_cap=80),
            MomentumAgent("Momentum2", lookback=8, trade_fraction=0.20, hard_cap=100),
            MomentumAgent("Momentum3", lookback=15, trade_fraction=0.25, hard_cap=120),
            MeanReversionAgent("MeanReversion1", lookback=10, threshold=0.015, trade_fraction=0.15, hard_cap=80),
            MeanReversionAgent("MeanReversion2", lookback=20, threshold=0.02, trade_fraction=0.20, hard_cap=100),
            MeanReversionAgent("MeanReversion3", lookback=40, threshold=0.03, trade_fraction=0.25, hard_cap=120),
            BuyAndHoldAgent("BuyAndHold1", buy_fraction=0.60),
            BuyAndHoldAgent("BuyAndHold2", buy_fraction=0.75),
            BuyAndHoldAgent("BuyAndHold3", buy_fraction=0.90),
            ContrarianAgent("Contrarian1", threshold=0.006, trade_fraction=0.15, hard_cap=80),
            ContrarianAgent("Contrarian2", threshold=0.008, trade_fraction=0.20, hard_cap=100),
            ContrarianAgent("Contrarian3", threshold=0.012, trade_fraction=0.25, hard_cap=120),
            VolumePressureAgent("VolumePressure1", sensitivity=0.20, trade_fraction=0.15, hard_cap=80),
            VolumePressureAgent("VolumePressure2", sensitivity=0.30, trade_fraction=0.20, hard_cap=100),
            VolumePressureAgent("VolumePressure3", sensitivity=0.40, trade_fraction=0.25, hard_cap=120),
            ValueEstimateAgent("ValueEstimate1", lookback=40, threshold=0.02, trade_fraction=0.15, hard_cap=80),
            ValueEstimateAgent("ValueEstimate2", lookback=60, threshold=0.03, trade_fraction=0.20, hard_cap=100),
            ValueEstimateAgent("ValueEstimate3", lookback=80, threshold=0.04, trade_fraction=0.25, hard_cap=120),
            RiskManagedAgent("RiskManaged1", lookback=15, threshold=0.02, trade_fraction=0.15, hard_cap=80),
            RiskManagedAgent("RiskManaged2", lookback=20, threshold=0.025, trade_fraction=0.20, hard_cap=100),
            RiskManagedAgent("RiskManaged3", lookback=30, threshold=0.03, trade_fraction=0.25, hard_cap=120),
        ]

        self.bah_agent = None
        for background_agent in background_agents:
            if background_agent.name == "BuyAndHold1":
                self.bah_agent = background_agent

        all_agents = background_agents + [self.rl_agent]
        fundamental_process = OrnsteinUhlenbeckFundamental()
        self.market = Market(all_agents, fundamental_process)
        self.market.quiet = True

        self.last_price = VALUES["initial_price"]




    def _build_obs(self):
        price = self.market.price
        price_history = [entry["market_price"] for entry in self.market.history]

        # pad with current price if not enough history yet
        history_len = VALUES["rl_history_len"]
        if len(price_history) >= history_len:
            recent_prices = price_history[-history_len:]
        else:
            padding = [price] * (history_len - len(price_history))
            recent_prices = padding + price_history

        history_relatives = [p / price for p in recent_prices]

        if len(self.market.history) > 0:
            last_net_order_flow = self.market.history[-1]["net_order_flow"]
            last_volume = sum(a["executed_quantity"] for a in self.market.history[-1]["actions"])
        else:
            last_net_order_flow = 0
            last_volume = 0
        order_imbalance = last_net_order_flow / max(1, last_volume)

        position_fill = self.rl_agent.position / self.rl_agent.hard_cap
        cash_fill = self.rl_agent.cash / self.initial_cash
        portfolio_value = self.rl_agent.cash + self.rl_agent.position * price
        portfolio_level = portfolio_value / self.initial_cash

        short_momentum = (price - price_history[-5]) / price_history[-5] if len(price_history) >= 5 else 0.0
        long_momentum = (price - price_history[-20]) / price_history[-20] if len(price_history) >= 20 else 0.0
        short_ma_gap = price / (sum(price_history[-10:]) / len(price_history[-10:])) - 1 if len(price_history) >= 10 else 0.0
        long_ma_gap = price / (sum(price_history[-40:]) / len(price_history[-40:])) - 1 if len(price_history) >= 40 else 0.0

        if len(price_history) >= 11:
            vol_prices = price_history[-11:]
            vol_returns = [(vol_prices[i] - vol_prices[i-1]) / vol_prices[i-1] for i in range(1, len(vol_prices))]
            mean_r = sum(vol_returns) / len(vol_returns)
            recent_volatility = (sum((r - mean_r) ** 2 for r in vol_returns) / len(vol_returns)) ** 0.5
        else:
            recent_volatility = 0.0

        above_start = (portfolio_value - self.initial_cash) / self.initial_cash

        # might want to add regime to obs later, would bump shape to 35
        obs = [price / VALUES["initial_price"]] + history_relatives + [position_fill, cash_fill, portfolio_level, order_imbalance, short_momentum, long_momentum, short_ma_gap, long_ma_gap, recent_volatility, above_start]
        return np.array(obs, dtype=np.float32)




    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        use_seed = seed if seed is not None else random.randint(0, 999999)
        random.seed(use_seed)

        self.market.reset()
        self.rl_agent.next_action = "hold"
        self.last_price = VALUES["initial_price"]

        obs = self._build_obs()
        return obs, {}




    def step(self, action):
        if action == 1:
            self.rl_agent.next_action = "buy"
            self.rl_agent.active_fraction = self.rl_agent.trade_fraction * 0.25
        elif action == 2:
            self.rl_agent.next_action = "buy"
            self.rl_agent.active_fraction = self.rl_agent.trade_fraction * 0.75
        elif action == 3:
            self.rl_agent.next_action = "buy"
            self.rl_agent.active_fraction = self.rl_agent.trade_fraction * 1.5
        elif action == 4:
            self.rl_agent.next_action = "sell"
            self.rl_agent.active_fraction = self.rl_agent.trade_fraction * 0.25
        elif action == 5:
            self.rl_agent.next_action = "sell"
            self.rl_agent.active_fraction = self.rl_agent.trade_fraction * 0.75
        elif action == 6:
            self.rl_agent.next_action = "sell"
            self.rl_agent.active_fraction = self.rl_agent.trade_fraction * 1.5
        else:
            self.rl_agent.next_action = "hold"
            self.rl_agent.active_fraction = self.rl_agent.trade_fraction

        portfolio_before = self.rl_agent.cash + self.rl_agent.position * self.market.price

        self.market.step()

        price = self.market.price
        portfolio_value = self.rl_agent.cash + self.rl_agent.position * price

        market_value_now = self.initial_cash * (price / VALUES["initial_price"])
        market_value_last = self.initial_cash * (self.last_price / VALUES["initial_price"])
        bah_step_change = market_value_now - market_value_last
        agent_step_change = portfolio_value - portfolio_before
        step_reward = (agent_step_change - bah_step_change) / self.initial_cash

        self.last_price = price
        terminated = self.market.step_count >= self.num_steps

        cash_fill = self.rl_agent.cash / self.initial_cash
        time_elapsed = self.market.step_count / self.num_steps
        cash_penalty = 0.0
        if time_elapsed > 0.5 and cash_fill > 0.8:
            cash_penalty = -0.001

        reward = step_reward + cash_penalty
        if terminated:
            bah_final = self.initial_cash * (price / VALUES["initial_price"])
            episode_bonus = ((portfolio_value - self.initial_cash) - (bah_final - self.initial_cash)) / self.initial_cash * 10
            reward = step_reward + cash_penalty + episode_bonus

        obs = self._build_obs()
        info = {
            "portfolio_value": portfolio_value,
            "position": self.rl_agent.position,
            "cash": self.rl_agent.cash,
        }

        return obs, reward, terminated, False, info
