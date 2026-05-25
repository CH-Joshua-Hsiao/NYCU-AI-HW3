# run_all.ps1
# One-shot script to train everything and generate all figures.
# Adjust --total-steps for quick tests (e.g., 100000 for a smoke test).

$ErrorActionPreference = "Stop"

Write-Host "=== Step 1: Accept Atari ROMs ===" -ForegroundColor Cyan
python -m autorom --accept-license

Write-Host "=== Step 2: Train DQN (scratch) on Pong ===" -ForegroundColor Cyan
python train_dqn_pong.py --total-steps 2000000 --lr 1e-4 --seed 42

Write-Host "=== Step 3: Train PPO (SB3) on Pong ===" -ForegroundColor Cyan
python train_ppo_pong.py --total-timesteps 2000000 --lr 2.5e-4 --seed 42

Write-Host "=== Step 4: Train PPO + DQN on LunarLander ===" -ForegroundColor Cyan
python train_lunarlander.py --total-timesteps 500000 --seed 42

Write-Host "=== Step 5: Plot results ===" -ForegroundColor Cyan
python plot_results.py

Write-Host "=== Step 6: Evaluate DQN (scratch) on Pong ===" -ForegroundColor Cyan
python evaluate.py `
    --model-type scratch `
    --model-path checkpoints/dqn_pong_lr0.0001_seed42/final.pt `
    --env-id ALE/Pong-v5 `
    --atari `
    --n-eval-episodes 20

Write-Host "=== Step 7: Evaluate PPO (SB3) on Pong ===" -ForegroundColor Cyan
python evaluate.py `
    --model-type sb3 --algo ppo `
    --model-path checkpoints/ppo_pong_lr0.00025_seed42/final.zip `
    --env-id ALE/Pong-v5 `
    --atari `
    --n-eval-episodes 20

Write-Host "=== Step 8: Evaluate PPO on LunarLander ===" -ForegroundColor Cyan
python evaluate.py `
    --model-type sb3 --algo ppo `
    --model-path checkpoints/ppo_lunar/final.zip `
    --env-id LunarLander-v3 `
    --n-eval-episodes 20

Write-Host "=== Step 9: Evaluate DQN on LunarLander ===" -ForegroundColor Cyan
python evaluate.py `
    --model-type sb3 --algo dqn `
    --model-path checkpoints/dqn_lunar_sb3/final.zip `
    --env-id LunarLander-v3 `
    --n-eval-episodes 20

Write-Host ""
Write-Host "All done! Update the table values in report.tex and compile:" -ForegroundColor Green
Write-Host "  pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex"
