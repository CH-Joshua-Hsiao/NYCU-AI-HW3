"""
Plot learning curves from experiment logs.

Produces:
    1. figures/lunar_ppo_vs_dqn.pdf  - LunarLander PPO vs DQN comparison
    2. figures/pong_dqn_scratch.pdf  - Pong DQN (from scratch) reward curve
    3. figures/pong_ppo_sb3.pdf      - Pong PPO (SB3) reward curve from eval log

Usage:
    python plot_results.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
})

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)


def smooth(values, window=10):
    """Moving average smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_sb3_eval_log(path: str):
    """
    Load SB3 EvalCallback npz log.
    Returns (timesteps, mean_rewards, std_rewards).
    """
    data = np.load(path)
    return data["timesteps"], data["results"].mean(axis=1), data["results"].std(axis=1)


def load_dqn_scratch_log(path: str):
    """
    Load our custom JSON reward log.
    Returns (steps, mean_rewards).
    """
    with open(path) as f:
        records = json.load(f)
    steps   = [r["step"]             for r in records]
    rewards = [r["mean_reward_10"]   for r in records]
    return np.array(steps), np.array(rewards)


# --- Figure 1: LunarLander PPO vs DQN -----------------------------------------
ppo_lunar_log = "logs/ppo_lunar/evaluations.npz"
dqn_lunar_log = "logs/dqn_lunar_sb3/evaluations.npz"

if Path(ppo_lunar_log).exists() and Path(dqn_lunar_log).exists():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for label, log_path, color in [
        ("PPO (SB3)", ppo_lunar_log, "#2563EB"),
        ("DQN (SB3)", dqn_lunar_log, "#DC2626"),
    ]:
        ts, means, stds = load_sb3_eval_log(log_path)
        ax.plot(ts, means, label=label, color=color, linewidth=1.8)
        ax.fill_between(ts, means - stds, means + stds, alpha=0.15, color=color)

    ax.axhline(200, color="gray", linestyle="--", linewidth=1.0, label="Solved threshold")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Mean Episodic Reward (10 episodes)")
    ax.set_title("LunarLander-v3: PPO vs DQN Learning Curves")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "lunar_ppo_vs_dqn.pdf")
    fig.savefig(FIGURES / "lunar_ppo_vs_dqn.png")
    plt.close(fig)
    print("Saved: figures/lunar_ppo_vs_dqn.pdf")
else:
    print(f"[SKIP] LunarLander logs not found - run train_lunarlander.py first.")

# --- Figure 2: Pong DQN (from scratch) ----------------------------------------
dqn_pong_log = "logs/dqn_pong_lr0.0001_seed42/rewards.json"

if Path(dqn_pong_log).exists():
    steps, rewards = load_dqn_scratch_log(dqn_pong_log)
    smoothed = smooth(rewards, window=5)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, rewards, color="#16A34A", alpha=0.3, linewidth=1.0, label="Raw mean(10)")
    ax.plot(steps[len(steps) - len(smoothed):], smoothed, color="#16A34A",
            linewidth=2.0, label="Smoothed (window=5)")

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Mean Episodic Reward (last 10 eps)")
    ax.set_title("Pong: DQN (from scratch) Learning Curve")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "pong_dqn_scratch.pdf")
    fig.savefig(FIGURES / "pong_dqn_scratch.png")
    plt.close(fig)
    print("Saved: figures/pong_dqn_scratch.pdf")
else:
    print(f"[SKIP] DQN-Pong log not found - run train_dqn_pong.py first.")

# --- Figure 3: Pong PPO (SB3) -------------------------------------------------
ppo_pong_log = "logs/ppo_pong_lr0.00025_seed42/evaluations.npz"

if Path(ppo_pong_log).exists():
    ts, means, stds = load_sb3_eval_log(ppo_pong_log)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ts, means, color="#7C3AED", linewidth=2.0, label="PPO (SB3)")
    ax.fill_between(ts, means - stds, means + stds, alpha=0.15, color="#7C3AED")

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Mean Episodic Reward (5 eval episodes)")
    ax.set_title("Pong: PPO (SB3) Learning Curve")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "pong_ppo_sb3.pdf")
    fig.savefig(FIGURES / "pong_ppo_sb3.png")
    plt.close(fig)
    print("Saved: figures/pong_ppo_sb3.pdf")
else:
    print(f"[SKIP] PPO-Pong eval log not found - run train_ppo_pong.py first.")

print("\nDone. All available figures saved in figures/")
