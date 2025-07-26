import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor_mean = nn.Linear(hidden_dim, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))  # 학습 가능한 log_std
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='tanh')
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        x = self.shared(state)
        mean = self.actor_mean(x)
        std = self.actor_log_std.exp().clamp(min=1e-2, max=2.0).expand_as(mean)

        if torch.isnan(mean).any() or torch.isnan(std).any():
            print("[NaN DETECTED in forward]")
            print("mean:", mean)
            print("std:", std)
            print("state:", state)
            raise ValueError("NaN in policy output")
        
        value = self.critic(x).squeeze(-1)
        return mean, std, value

    def act(self, state):
        mean, std, value = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.squeeze().item(), log_prob.squeeze(), value

class PPOAgent:
    def __init__(self, state_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, entropy_coef=0.01, value_coef=0.5, device='cpu',
                 action_low=2.0, action_high=5.0):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = device

        self.action_low = action_low
        self.action_high = action_high
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

        self.policy = ActorCriticNetwork(state_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def act(self, state):
        mean, std, value = self.policy(state)
        base_dist = torch.distributions.Normal(mean, std)
        tanh = torch.distributions.transforms.TanhTransform()
        dist = torch.distributions.TransformedDistribution(base_dist, tanh)
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action) - torch.log(torch.tensor(self.action_scale))
        action = raw_action * self.action_scale + self.action_bias
        return action.squeeze().item(), log_prob.squeeze(), value.squeeze()

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

        returns, advantages = self.compute_returns_and_advantages(rewards, dones, old_values)

        if advantages.std().item() < 1e-5 or len(advantages) < 2:
            print("⚠️ Skip normalization due to insufficient std or batch size")
        else:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(batch_epochs):
            for i in range(0, len(states), batch_size):
                idx = slice(i, i + batch_size)
                mean, std, values = self.policy(states[idx])
                base_dist = torch.distributions.Normal(mean, std)
                tanh = torch.distributions.transforms.TanhTransform()
                dist = torch.distributions.TransformedDistribution(base_dist, tanh)

                scaled_actions = actions[idx]
                norm_actions = (scaled_actions - self.action_bias) / self.action_scale
                norm_actions = norm_actions.clamp(-0.999, 0.999)
                new_log_probs = dist.log_prob(norm_actions) - torch.log(torch.tensor(self.action_scale))
                new_log_probs = new_log_probs.squeeze()
                entropy = base_dist.entropy().mean()

                ratios = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratios * advantages[idx]
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns[idx])
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                if torch.isnan(total_loss):
                    print("❌ [NaN] total_loss 발생 - optimizer step 건너뜀")
                    print("  mean:", mean.detach().cpu().numpy().flatten())
                    print("  std:", std.detach().cpu().numpy().flatten())
                    print("  ratios:", ratios.detach().cpu().numpy().flatten())
                    print("  advantages:", advantages[idx].detach().cpu().numpy().flatten())
                    print(f"  value_loss: {value_loss.item():.4f}, policy_loss: {policy_loss.item():.4f}")
                    continue

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()