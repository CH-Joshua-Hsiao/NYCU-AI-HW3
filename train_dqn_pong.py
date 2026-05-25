"""
Train DQN (from scratch) on Atari Pong.

Usage:
    python train_dqn_pong.py [--total-steps 2000000] [--lr 1e-4]

Logs reward curves to logs/dqn_pong/ and saves checkpoints to
checkpoints/dqn_pong/.
"""

import argparse
import time
import json
import os
import sys
import numpy as np
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from dqn_scratch import DQNAgent, FrameStack

# --- Argument Parsing ----------------------------------------------------------
parser = argparse.ArgumentParser(description="Train DQN (scratch) on Pong")
parser.add_argument("--total-steps", type=int, default=2_000_000)
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate for Adam optimizer")
parser.add_argument("--buffer-size", type=int, default=100_000)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--target-update-freq", type=int, default=1000)
parser.add_argument("--eps-decay", type=int, default=250_000)
parser.add_argument("--warmup-steps", type=int, default=10_000,
                    help="Steps to fill buffer before training starts")
parser.add_argument("--env-id", type=str, default="ALE/Pong-v5")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log-interval", type=int, default=10,
                    help="Log every N episodes")
parser.add_argument("--save-interval", type=int, default=100_000,
                    help="Save checkpoint every N steps")
args = parser.parse_args()

# --- Directories ---------------------------------------------------------------
run_name = f"dqn_pong_lr{args.lr}_seed{args.seed}"
log_dir  = Path("logs") / run_name
ckpt_dir = Path("checkpoints") / run_name
log_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Save hyperparameters
with open(log_dir / "hparams.json", "w") as f:
    json.dump(vars(args), f, indent=2)

# --- Environment ---------------------------------------------------------------
env = gym.make(args.env_id, render_mode=None)
env.reset(seed=args.seed)
np.random.seed(args.seed)

n_actions = env.action_space.n
obs_shape = (4, 84, 84)  # 4 stacked grayscale frames

frame_stack = FrameStack(n_frames=4)

# --- Agent ---------------------------------------------------------------------
agent = DQNAgent(
    obs_shape=obs_shape,
    n_actions=n_actions,
    device="cuda",
    lr=args.lr,
    gamma=args.gamma,
    buffer_size=args.buffer_size,
    batch_size=args.batch_size,
    target_update_freq=args.target_update_freq,
    eps_decay=args.eps_decay,
)

print(f"[DQN-Pong] Training on {args.env_id} | device: {agent.device} | run: {run_name}")
print(f"           total_steps={args.total_steps:,} | lr={args.lr} | buffer={args.buffer_size:,}")

# --- Training Loop -------------------------------------------------------------
episode_rewards = []
reward_log = []  # (step, mean_reward_last_10)

global_step = 0
episode = 0
start_time = time.time()

while global_step < args.total_steps:
    raw_obs, _ = env.reset()
    obs = frame_stack.reset(raw_obs)
    episode_reward = 0.0
    done = False
    truncated = False

    while not (done or truncated):
        action = agent.select_action(obs)
        raw_next, reward, done, truncated, _ = env.step(action)
        next_obs = frame_stack.step(raw_next)

        # Clip reward to [-1, 1] per Mnih et al. (2015)
        clipped_reward = np.clip(reward, -1.0, 1.0)

        agent.store_transition(obs, action, clipped_reward, next_obs, float(done))
        obs = next_obs
        episode_reward += reward
        global_step += 1

        # Only start training after warm-up
        if global_step >= args.warmup_steps:
            agent.update()

        # Save checkpoint
        if global_step % args.save_interval == 0:
            ckpt_path = ckpt_dir / f"step_{global_step}.pt"
            agent.save(str(ckpt_path))

    episode += 1
    episode_rewards.append(episode_reward)

    if episode % args.log_interval == 0:
        mean_r = np.mean(episode_rewards[-10:])
        elapsed = time.time() - start_time
        fps = global_step / elapsed
        eps = agent.epsilon()
        print(
            f"[Ep {episode:5d} | Step {global_step:>8,}]  "
            f"MeanR(10)={mean_r:7.2f}  Eps={eps:.3f}  FPS={fps:.0f}"
        )
        reward_log.append({"step": global_step, "episode": episode, "mean_reward_10": mean_r})
        # Save reward log incrementally
        with open(log_dir / "rewards.json", "w") as f:
            json.dump(reward_log, f, indent=2)

env.close()

# --- Final Save ----------------------------------------------------------------
agent.save(str(ckpt_dir / "final.pt"))
with open(log_dir / "rewards.json", "w") as f:
    json.dump(reward_log, f, indent=2)

print(f"\n[Done] Training complete. Final model saved to {ckpt_dir / 'final.pt'}")
print(f"       Reward log saved to {log_dir / 'rewards.json'}")

