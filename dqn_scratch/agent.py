import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .replay_buffer import ReplayBuffer
from .networks import CNN_QNetwork

class DQNAgent:
    def __init__(self, obs_shape, n_actions, device, lr, gamma, buffer_size, batch_size, target_update_freq, eps_decay):
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps_decay = eps_decay
        self.global_step = 0

        self.policy_net = CNN_QNetwork(obs_shape, n_actions).to(self.device)
        self.target_net = CNN_QNetwork(obs_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.loss_fn = nn.HuberLoss()

    def select_action(self, obs):
        eps = self.epsilon()
        if np.random.rand() < eps:
            return np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, device=self.device).unsqueeze(0)
                q_values = self.policy_net(obs_t)
                return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def epsilon(self):
        eps_start = 1.0
        eps_end = 0.01
        fraction = min(float(self.global_step) / self.eps_decay, 1.0)
        return eps_start + fraction * (eps_end - eps_start)

    def update(self):
        self.global_step += 1
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device).unsqueeze(1)

        curr_q = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            max_next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q

        loss = self.loss_fn(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.global_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.global_step = checkpoint['global_step']
