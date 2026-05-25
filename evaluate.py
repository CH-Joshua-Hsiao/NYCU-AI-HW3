import argparse
import numpy as np
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import torch
from pathlib import Path

from dqn_scratch import DQNAgent, FrameStack

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained agent")
    parser.add_argument("--model-type", type=str, required=True, choices=["sb3", "scratch"])
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], help="RL algorithm (needed for SB3)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model file (.zip or .pt)")
    parser.add_argument("--env-id", type=str, required=True, help="Gymnasium env ID")
    parser.add_argument("--atari", action="store_true", help="Is it an Atari environment?")
    parser.add_argument("--n-eval-episodes", type=int, default=20, help="Number of evaluation episodes")
    args = parser.parse_args()

    if args.atari:
        if args.model_type == "sb3":
            from stable_baselines3.common.env_util import make_atari_env
            from stable_baselines3.common.vec_env import VecFrameStack
            eval_env = make_atari_env(args.env_id, n_envs=1, seed=123)
            eval_env = VecFrameStack(eval_env, n_stack=4)
        else:
            eval_env = gym.make(args.env_id, render_mode=None)
            frame_stack = FrameStack(n_frames=4)
    else:
        eval_env = gym.make(args.env_id, render_mode=None)

    if args.model_type == "sb3":
        from stable_baselines3 import PPO, DQN
        if args.algo == "ppo":
            model = PPO.load(args.model_path)
        elif args.algo == "dqn":
            model = DQN.load(args.model_path)
        else:
            raise ValueError("Algorithm (--algo) must be 'ppo' or 'dqn' for SB3 models")
    else:
        if args.atari:
            n_actions = eval_env.action_space.n
            obs_shape = (4, 84, 84)
        else:
            n_actions = eval_env.action_space.n
            obs_shape = eval_env.observation_space.shape
        
        agent = DQNAgent(
            obs_shape=obs_shape,
            n_actions=n_actions,
            device="cuda" if torch.cuda.is_available() else "cpu",
            lr=1e-4,
            gamma=0.99,
            buffer_size=1000,
            batch_size=32,
            target_update_freq=1000,
            eps_decay=1000,
        )
        agent.load(args.model_path)
        agent.epsilon = lambda: 0.0

    episode_rewards = []
    
    for ep in range(args.n_eval_episodes):
        if args.model_type == "sb3" and args.atari:
            obs = eval_env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, info = eval_env.step(action)
                ep_reward += reward[0]
                done = dones[0]
            episode_rewards.append(ep_reward)
        else:
            obs, _ = eval_env.reset(seed=123 + ep)
            if args.atari and args.model_type == "scratch":
                obs = frame_stack.reset(obs)
            
            done = False
            truncated = False
            ep_reward = 0.0
            while not (done or truncated):
                if args.model_type == "sb3":
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = agent.select_action(obs)
                
                obs, reward, done, truncated, _ = eval_env.step(action)
                if args.atari and args.model_type == "scratch":
                    obs = frame_stack.step(obs)
                ep_reward += reward
            episode_rewards.append(ep_reward)
            
    eval_env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print("-" * 50)
    print(f"Evaluation of {args.model_path} on {args.env_id}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Episodes: {args.n_eval_episodes}")
    print("-" * 50)

if __name__ == "__main__":
    main()
