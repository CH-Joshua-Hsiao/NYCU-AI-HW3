"""
Train DQN and PPO on LunarLander-v3 using Stable-Baselines3 for algorithm comparison.

This script runs both algorithms with identical budgets and seeds, then saves
reward logs for plotting a learning-curve comparison.

Usage:
    python train_lunarlander.py [--total-timesteps 500000] [--seed 42]

References:
    Stable-Baselines3: https://stable-baselines3.readthedocs.io/
    LunarLander-v3:    https://gymnasium.farama.org/environments/box2d/lunar_lander/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

# --- Argument Parsing ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--total-timesteps", type=int, default=500_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--env-id", type=str, default="LunarLander-v3")
args = parser.parse_args()

ENV_ID = args.env_id
TOTAL  = args.total_timesteps
SEED   = args.seed

# --- Helper: make monitored env ------------------------------------------------
def make_env(log_dir: Path, seed: int):
    log_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(ENV_ID)
    env = Monitor(env, str(log_dir))
    return env

# --- PPO ------------------------------------------------------------------------
print("=" * 60)
print(f"[LunarLander] Training PPO for {TOTAL:,} steps ...")
print("=" * 60)

ppo_log = Path("logs/ppo_lunar")
ppo_log.mkdir(parents=True, exist_ok=True)
ppo_ckpt = Path("checkpoints/ppo_lunar")
ppo_ckpt.mkdir(parents=True, exist_ok=True)

ppo_env = make_env(ppo_log, SEED)
ppo_eval_env = Monitor(gym.make(ENV_ID), str(ppo_log / "eval"))

ppo_eval_cb = EvalCallback(
    ppo_eval_env,
    best_model_save_path=str(ppo_ckpt / "best"),
    log_path=str(ppo_log),
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
)

ppo_model = PPO(
    "MlpPolicy",
    ppo_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.0,
    clip_range=0.2,
    verbose=1,
    tensorboard_log=str(ppo_log / "tensorboard"),
    device="cuda",
    seed=SEED,
)
ppo_model.learn(total_timesteps=TOTAL, callback=ppo_eval_cb, progress_bar=True)
ppo_model.save(str(ppo_ckpt / "final"))
ppo_env.close()
ppo_eval_env.close()

# --- DQN ------------------------------------------------------------------------
print("=" * 60)
print(f"[LunarLander] Training DQN for {TOTAL:,} steps ...")
print("=" * 60)

dqn_log = Path("logs/dqn_lunar_sb3")
dqn_log.mkdir(parents=True, exist_ok=True)
dqn_ckpt = Path("checkpoints/dqn_lunar_sb3")
dqn_ckpt.mkdir(parents=True, exist_ok=True)

dqn_env = make_env(dqn_log, SEED)
dqn_eval_env = Monitor(gym.make(ENV_ID), str(dqn_log / "eval"))

dqn_eval_cb = EvalCallback(
    dqn_eval_env,
    best_model_save_path=str(dqn_ckpt / "best"),
    log_path=str(dqn_log),
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
)

dqn_model = DQN(
    "MlpPolicy",
    dqn_env,
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1_000,
    exploration_fraction=0.2,
    exploration_final_eps=0.01,
    verbose=1,
    tensorboard_log=str(dqn_log / "tensorboard"),
    device="cuda",
    seed=SEED,
)
dqn_model.learn(total_timesteps=TOTAL, callback=dqn_eval_cb, progress_bar=True)
dqn_model.save(str(dqn_ckpt / "final"))
dqn_env.close()
dqn_eval_env.close()

print("\n[Done] LunarLander experiments complete.")
print(f"       PPO model: checkpoints/ppo_lunar/final.zip")
print(f"       DQN model: checkpoints/dqn_lunar_sb3/final.zip")

