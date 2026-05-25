"""
Train PPO on Atari Pong using Stable-Baselines3.

This script uses SB3's PPO with a CnnPolicy (same Nature-DQN CNN backbone)
and the gymnasium Atari wrappers for preprocessing (frame stacking, grayscale, etc.)

Usage:
    python train_ppo_pong.py [--total-timesteps 2000000] [--lr 2.5e-4]

References:
    Schulman et al. (2017), "Proximal Policy Optimization Algorithms",
    arXiv:1707.06347. https://arxiv.org/abs/1707.06347

    Antonin Raffin et al. (2021), "Stable-Baselines3: Reliable Reinforcement
    Learning Implementations", JMLR. https://jmlr.org/papers/v22/20-1364.html
"""

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

# --- Argument Parsing ----------------------------------------------------------
parser = argparse.ArgumentParser(description="Train PPO (SB3) on Pong")
parser.add_argument("--total-timesteps", type=int, default=2_000_000)
parser.add_argument("--lr", type=float, default=2.5e-4)
parser.add_argument("--n-steps", type=int, default=128,
                    help="Steps per environment before each update")
parser.add_argument("--n-envs", type=int, default=16,
                    help="Number of parallel environments")
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--n-epochs", type=int, default=4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--clip-range", type=float, default=0.1)
parser.add_argument("--ent-coef", type=float, default=0.01)
parser.add_argument("--env-id", type=str, default="ALE/Pong-v5")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# --- Directories ---------------------------------------------------------------
run_name = f"ppo_pong_lr{args.lr}_seed{args.seed}"
log_dir  = Path("logs") / run_name
ckpt_dir = Path("checkpoints") / run_name
log_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

# --- Environments --------------------------------------------------------------
# SB3 provides make_atari_env which handles: NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv,
# FireResetEnv, WarpFrame (84x84 grayscale), ClipRewardEnv.
env = make_atari_env(args.env_id, n_envs=args.n_envs, seed=args.seed)
env = VecFrameStack(env, n_stack=4)

eval_env = make_atari_env(args.env_id, n_envs=1, seed=args.seed + 1000)
eval_env = VecFrameStack(eval_env, n_stack=4)

# --- Callbacks -----------------------------------------------------------------
checkpoint_cb = CheckpointCallback(
    save_freq=max(100_000 // args.n_envs, 1),
    save_path=str(ckpt_dir),
    name_prefix="ppo_pong",
)
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=str(ckpt_dir / "best"),
    log_path=str(log_dir),
    eval_freq=max(250_000 // args.n_envs, 1),
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)

# --- Model ---------------------------------------------------------------------
model = PPO(
    policy="CnnPolicy",
    env=env,
    learning_rate=args.lr,
    n_steps=args.n_steps,
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    gae_lambda=args.gae_lambda,
    clip_range=args.clip_range,
    ent_coef=args.ent_coef,
    verbose=1,
    tensorboard_log=str(log_dir / "tensorboard"),
    device="cuda",
    seed=args.seed,
)

print(f"[PPO-Pong] Training on {args.env_id} | n_envs={args.n_envs} | run: {run_name}")
print(f"           total_timesteps={args.total_timesteps:,} | lr={args.lr}")

# --- Train ---------------------------------------------------------------------
model.learn(
    total_timesteps=args.total_timesteps,
    callback=[checkpoint_cb, eval_cb],
    progress_bar=True,
)

# --- Save Final Model ----------------------------------------------------------
model.save(str(ckpt_dir / "final"))
env.close()
eval_env.close()

print(f"\n[Done] PPO-Pong training complete. Final model: {ckpt_dir / 'final.zip'}")

