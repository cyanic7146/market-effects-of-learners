import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from market_env import MarketEnv
from values import VALUES

os.makedirs("MODELS", exist_ok=True)

env = make_vec_env(MarketEnv, n_envs=8)  # increase n_envs if you have more cores
env = VecNormalize(env, norm_obs=True, norm_reward=True)

eval_env = DummyVecEnv([MarketEnv])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
eval_env_plain = MarketEnv()


# saves normalizer at the same moment a new best model is found
class EvalWithNormSave(EvalCallback):
    def __init__(self, training_env, norm_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._training_env = training_env
        self._norm_path = norm_path
        self._prev_best = -np.inf

    def _on_step(self):
        result = super()._on_step()
        if self.best_mean_reward > self._prev_best:
            self._prev_best = self.best_mean_reward
            self._training_env.save(self._norm_path)
        return result


stop_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=500,
    min_evals=100,
    verbose=1
)

eval_callback = EvalWithNormSave(
    env,
    "MODELS/vec_normalize.pkl",
    eval_env,
    best_model_save_path="MODELS/best_model_31obs_checkpoint/",
    log_path="MODELS/",
    eval_freq=10000,
    n_eval_episodes=50,
    callback_after_eval=stop_callback,
    verbose=1
)

policy_kwargs = dict(net_arch=[256, 256, 128])
model = PPO("MlpPolicy", env, verbose=1,
            ent_coef=0.01, n_steps=4096, batch_size=256,
            policy_kwargs=policy_kwargs,
            learning_rate=0.00003,
            target_kl=0.01,
            n_epochs=5)
model.learn(total_timesteps=1000000000, callback=eval_callback)

best = PPO.load("MODELS/best_model_31obs_checkpoint/best_model")
best.save("MODELS/rl_model_best_improved_v1000")

print("\ntraining done, running eval episode...")

# load the normalizer saved at the best checkpoint, not end-of-training stats
norm_loader = DummyVecEnv([MarketEnv])
norm_loader = VecNormalize.load("MODELS/vec_normalize.pkl", norm_loader)
obs_mean = norm_loader.obs_rms.mean
obs_var = norm_loader.obs_rms.var
obs_clip = norm_loader.clip_obs

obs, _ = eval_env_plain.reset(seed=67676767)
obs = np.clip((obs - obs_mean) / np.sqrt(obs_var + 1e-8), -obs_clip, obs_clip).astype(np.float32)
done = False
while not done:
    action, _ = best.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env_plain.step(action)
    obs = np.clip((obs - obs_mean) / np.sqrt(obs_var + 1e-8), -obs_clip, obs_clip).astype(np.float32)
    done = terminated or truncated

final_portfolio = info["portfolio_value"]
final_return = (final_portfolio - VALUES["initial_cash"]) / VALUES["initial_cash"] * 100
print(f"eval portfolio value: {final_portfolio:.2f}")
print(f"eval return: {final_return:.2f}%")
