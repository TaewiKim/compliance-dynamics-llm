import numpy as np
from buffer import RolloutBuffer
from reward_utils import compute_reward

class PPOTrainerSimulator:
    def __init__(self, user, ppo_agent, total_steps=1000, update_interval=64, action_low=2.0, action_high=5.0):
        self.user = user
        self.ppo_agent = ppo_agent
        self.total_steps = total_steps
        self.update_interval = update_interval
        self.action_low = action_low
        self.action_high = action_high
        self.buffer = RolloutBuffer()
        self.logs = {
            'suggestions': [],
            'actions': [],
            'rewards': [],
            'compliance': [],
            'true_mu': [],
            'avg_rewards': [],
            'update_steps': []
        }

    def state_vector(self):
        return np.array([
            self.user.behavior_mean,
            self.user.compliance_prob(3.5),   # 기준 제안과의 거리 기반 예시
            np.random.rand(),                 # 노이즈
            0.0,                              # placeholder
            1.0                               # fixed temp
        ])

    def run(self):
        for step in range(self.total_steps):
            state = self.state_vector()
            action = self.ppo_agent.select_action(state, self.buffer)
            action = float(np.clip(action, self.action_low, self.action_high))

            user_action = self.user.respond(action)
            compliance = self.user.compliance_prob(action)

            suggestion_idx = self.user.get_nearest_action_index(action)  # Continuous이므로 가장 가까운 index 추정 필요

            reward, _ = compute_reward(
                suggestion_idx=suggestion_idx,
                action=user_action,
                actual_compliance=compliance,
                estimated_behavior_mean=self.user.estimated_behavior_mean,
                action_space=self.user.action_space,
                goal=self.user.goal,
                reward_weight_compliance=self.user.reward_weight_compliance,
                reward_weight_goal=self.user.reward_weight_goal,
                penalty_weight_behavior_mean=self.user.penalty_weight_behavior_mean,
                penalty_weight_suggestion=self.user.penalty_weight_suggestion,
            )

            self.user._update_q(suggestion_idx, reward)
            self.user._update_compliance(compliance)
            self.user._update_behavior_mean(user_action)

            self.buffer.rewards.append(reward)
            self.buffer.dones.append(False)

            self.logs['suggestions'].append(action)
            self.logs['actions'].append(user_action)
            self.logs['compliance'].append(compliance)
            self.logs['rewards'].append(reward)
            self.logs['true_mu'].append(self.user.behavior_mean)

            if (step + 1) % self.update_interval == 0:
                print(f"\n--- PPO Update at Step {step + 1} ---")
                avg_reward = np.mean(self.logs['rewards'][-self.update_interval:])
                print(f"Average Reward: {avg_reward:.4f} | True Mu: {self.user.behavior_mean:.4f}")
                self.logs['avg_rewards'].append(avg_reward)
                self.logs['update_steps'].append(step + 1)

                self.ppo_agent.update(self.buffer)
                self.buffer.clear()
