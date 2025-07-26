import numpy as np
import torch
from buffer import RolloutBuffer
from reward_utils import compute_reward

class PPOTrainerSimulator:
    def __init__(self, user, ppo_agent, total_steps=1000, update_interval=5, action_low=2.0, action_high=5.0):
        self.user = user
        self.ppo_agent = ppo_agent
        self.total_steps = total_steps
        self.update_interval = update_interval
        self.action_low = action_low
        self.action_high = action_high
        self.buffer = RolloutBuffer()
        self.logs = {
            'suggestions': [], 'actions': [], 'rewards': [], 'compliance': [],
            'true_mu': [], 'avg_rewards': [], 'update_steps': []
        }

    def state_vector(self):
        return np.array([
            self.user.behavior_mean,
            self.user.compliance_prob(3.5),
        ])

    def run(self):
        for step in range(self.total_steps):
            state = self.state_vector()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.ppo_agent.device)

            action, log_prob, value = self.ppo_agent.act(state_tensor)

            user_action = self.user.respond(action)
            compliance = self.user.compliance_prob(action)
            suggestion_idx = self.user.get_nearest_action_index(action)

            reward, _ = compute_reward(
                suggestion_idx=suggestion_idx,
                action=user_action,
                actual_compliance=compliance,
                behavior_mean=self.user.behavior_mean,
                action_space=self.user.action_space,
                goal=self.user.goal
            )

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.rewards.append(reward)
            self.buffer.dones.append(False)
            self.buffer.log_probs.append(log_prob.detach().item()) 
            self.buffer.state_values.append(value.detach())

            self.logs['suggestions'].append(action)
            self.logs['actions'].append(user_action)
            self.logs['compliance'].append(compliance)
            self.logs['rewards'].append(reward)
            self.logs['true_mu'].append(self.user.behavior_mean)

            if (step + 1) % self.update_interval == 0:
                avg_reward = np.mean(self.logs['rewards'][-self.update_interval:])
                print(f"\\n--- PPO Update at Step {step + 1} ---")
                print(f"Average Reward: {avg_reward:.4f} | True Mu: {self.user.behavior_mean:.4f}")
                print(f"Sample State: {state}")
                print(f"Sample Action: {action:.4f} | Compliance: {compliance:.4f}")
                self.logs['avg_rewards'].append(avg_reward)
                self.logs['update_steps'].append(step + 1)

                self.ppo_agent.update(self.buffer)
                self.buffer.clear()