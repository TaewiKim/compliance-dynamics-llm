# agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from buffer import RolloutBuffer
import numpy as np

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor_mean = nn.Linear(hidden_dim, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))  # 학습 가능한 std
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        x = self.shared(state)
        mean = self.actor_mean(x)
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(x).squeeze(-1)
        return mean, std, value

    def act(self, state):
        mean, std, value = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.squeeze().item(), log_prob.squeeze(), value

class PPOAgent:
    def __init__(self, state_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, entropy_coef=0.01, value_coef=0.5, device='cpu'):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = device

        self.policy = ActorCriticNetwork(state_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state, buffer):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.act(state_tensor)

        # Clamp to environment action bounds
        action = float(np.clip(action, 2.0, 5.0))

        buffer.states.append(state)
        buffer.actions.append(action)
        buffer.log_probs.append(log_prob.item())
        buffer.state_values.append(value.item())

        return action

    def compute_returns_and_advantages(self, rewards, dones, values):
        returns, G = [], 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = returns - values
        return returns, advantages

    def update(self, buffer, batch_epochs=4, batch_size=64):
        data = buffer.to_tensors(self.device)
        states, actions = data['states'], data['actions']
        old_log_probs, rewards, dones, old_values = data['log_probs'], data['rewards'], data['dones'], data['state_values']
        with torch.no_grad():
            old_values_tensor = torch.tensor(old_values, dtype=torch.float32).to(self.device)

        returns, advantages = self.compute_returns_and_advantages(rewards, dones, old_values_tensor)

        for _ in range(batch_epochs):
            for i in range(0, len(states), batch_size):
                idx = slice(i, i + batch_size)
                mean, std, values = self.policy(states[idx])
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(actions[idx]).squeeze()
                entropy = dist.entropy().mean()

                # PPO loss
                ratios = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratios * advantages[idx]
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns[idx])
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
