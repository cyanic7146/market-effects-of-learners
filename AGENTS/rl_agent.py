import os
import numpy as np
from AGENTS.base_agent import BaseAgent
from values import VALUES


loaded_models = {}


def load_shared_model(model_path, normalizer_path):
    key = (model_path, normalizer_path)
    if key not in loaded_models:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        from market_env import MarketEnv
        model = PPO.load(model_path)
        norm_loader = DummyVecEnv([MarketEnv])
        norm_loader = VecNormalize.load(normalizer_path, norm_loader)
        loaded_models[key] = (model, norm_loader.obs_rms.mean, norm_loader.obs_rms.var, norm_loader.clip_obs)
    return loaded_models[key]


class RLAgent(BaseAgent):
    def __init__(self, name, cash=VALUES["initial_cash"], trade_fraction=VALUES["rl_agent_trade_fraction"], hard_cap=VALUES["rl_agent_hard_cap"], model_path="MODELS/best_model_31obs_checkpoint/best_model", normalizer_path="MODELS/vec_normalize.pkl"):
        super().__init__(name, cash)
        self.trade_fraction = float(trade_fraction)
        self.hard_cap = int(hard_cap)
        self.initial_cash = float(cash)

        self.model = None
        self.obs_mean = None
        self.obs_var = None
        self.obs_clip = 10.0

        if os.path.exists(model_path + ".zip") and os.path.exists(normalizer_path):
            self.model, self.obs_mean, self.obs_var, self.obs_clip = load_shared_model(model_path, normalizer_path)




    def build_model_obs(self, observation):
        price = float(observation["market_price"])
        price_history = observation["price_history"]

        history_len = VALUES["rl_history_len"]
        if len(price_history) >= history_len:
            recent_prices = price_history[-history_len:]
        else:
            padding = [price] * (history_len - len(price_history))
            recent_prices = padding + price_history

        history_relatives = [p / price for p in recent_prices]

        last_net_order_flow = observation["last_net_order_flow"]
        last_volume = observation["last_volume"]
        order_imbalance = last_net_order_flow / max(1, last_volume)

        position_fill = self.position / self.hard_cap
        cash_fill = self.cash / self.initial_cash
        portfolio_value = self.cash + self.position * price
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

        obs = [price / VALUES["initial_price"]] + history_relatives + [position_fill, cash_fill, portfolio_level, order_imbalance, short_momentum, long_momentum, short_ma_gap, long_ma_gap, recent_volatility, above_start]
        return np.array(obs, dtype=np.float32)




    def act(self, observation):
        if self.model is None:
            return {
                "type": "hold",
                "quantity": 0
            }

        obs = self.build_model_obs(observation)
        obs = np.clip((obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8), -self.obs_clip, self.obs_clip).astype(np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)

        price = float(observation["market_price"])

        if action in (1, 2, 3):
            if action == 1:
                fraction = self.trade_fraction * 0.25
            elif action == 2:
                fraction = self.trade_fraction * 0.75
            else:
                fraction = self.trade_fraction * 1.5

            affordable_quantity = int(self.cash // price)
            if affordable_quantity <= 0:
                return {
                    "type": "hold",
                    "quantity": 0
                }
            max_quantity = max(1, int(fraction * affordable_quantity))
            max_quantity = min(max_quantity, self.hard_cap, affordable_quantity)
            return {
                "type": "buy",
                "quantity": max_quantity
            }

        if action in (4, 5, 6):
            sellable_quantity = self.sell_capacity()
            if sellable_quantity <= 0:
                return {
                    "type": "hold",
                    "quantity": 0
                }
            if action == 4:
                fraction = self.trade_fraction * 0.25
            elif action == 5:
                fraction = self.trade_fraction * 0.75
            else:
                fraction = self.trade_fraction * 1.5

            max_quantity = max(1, int(fraction * sellable_quantity))
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
