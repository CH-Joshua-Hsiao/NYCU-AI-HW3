# AI HW3 — Reinforcement Learning

**NYCU Spring 2026 | Project #3 | Due 2026-05-29**

## Project Structure

```
NYCU-AI-HW3/
├── dqn_scratch/              # DQN from-scratch implementation
│   ├── __init__.py
│   ├── agent.py              # DQN agent (Double DQN, target network, eps-greedy)
│   ├── networks.py           # MLP (LunarLander) + CNN (Atari) Q-networks
│   ├── replay_buffer.py      # Experience replay buffer
│   └── atari_wrappers.py     # Frame grayscale, resize, stack
│
├── train_dqn_pong.py         # Train DQN (scratch) on ALE/Pong-v5
├── train_ppo_pong.py         # Train PPO (SB3) on ALE/Pong-v5
├── train_lunarlander.py      # Train PPO + DQN (SB3) on LunarLander-v3
├── evaluate.py               # Evaluate any model (supports sb3 and scratch)
├── plot_results.py           # Generate all figures for the report
│
├── report.tex                # LaTeX report (fill in results after training)
├── references.bib            # BibTeX bibliography
├── run_all.ps1               # Run everything in one shot (PowerShell)
│
├── logs/                     # Training logs (auto-created)
├── checkpoints/              # Model checkpoints (auto-created)
└── figures/                  # Saved plots (auto-created)
```

## Quick Start

### 1. Install dependencies
```powershell
pip install stable-baselines3[extra] "gymnasium[atari,box2d,accept-rom-license]" `
    ale-py autorom tensorboard matplotlib opencv-python "imageio[ffmpeg]"
```

### 2. Accept Atari ROMs
```powershell
python -m autorom --accept-license
```

### 3. Run all training (or use run_all.ps1)
```powershell
# Atari Pong — DQN (from scratch, ~hours on RTX 3090)
python train_dqn_pong.py --total-steps 2000000

# Atari Pong — PPO (SB3, 8 parallel envs)
python train_ppo_pong.py --total-timesteps 2000000

# LunarLander — PPO vs DQN comparison
python train_lunarlander.py --total-timesteps 500000
```

### 4. Plot results
```powershell
python plot_results.py
```

### 5. Compile report
```powershell
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

> **⚠️ Remember:** Add your Student ID to `report.tex` and fill in the results
> tables after training completes.

## Algorithms

| Algorithm | Environment | Implementation |
|-----------|------------|----------------|
| DQN (Nature 2015) + Double DQN | Pong | **From scratch** |
| PPO (Schulman 2017) | Pong | Stable-Baselines3 |
| PPO | LunarLander | Stable-Baselines3 |
| DQN | LunarLander | Stable-Baselines3 |

## References
- Mnih et al. (2015) — Human-level control through deep reinforcement learning
- Schulman et al. (2017) — Proximal Policy Optimization Algorithms
- Raffin et al. (2021) — Stable-Baselines3
- Towers et al. (2024) — Gymnasium