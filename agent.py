import numpy as np

class Agent:
    """
    Reinforcement learning-based agent that learns to make behavior suggestions
    based on estimated user preference (μ̂), compliance probability, and goal alignment.
    """

    def __init__(
        self,
        action_space,
        goal_behavior=4.0,
        reward_weight_compliance=1.0,
        reward_weight_goal=1.5,
        penalty_weight_suggestion=0.1,
        softness_mu=0.1,
        softness_goal=0.1,
        lr_q=0.6,
        lr_compliance=0.6,
        lr_mu=0.6,
        policy_temp_decay=0.04
    ):

        self.action_space = np.array(action_space)
        self.num_actions = len(action_space)
        self.q_values = np.zeros(self.num_actions)
        self.goal = goal_behavior

        self.estimated_compliance = 0.5
        self.estimated_behavior_mean = goal_behavior  # μ̂

        # Reward weights
        self.reward_weight_compliance = reward_weight_compliance
        self.reward_weight_goal = reward_weight_goal
        self.penalty_weight_suggestion = penalty_weight_suggestion
        self.penalty_weight_behavior_mean = 0.1

        # Learning rates
        self.learning_rate_q = lr_q
        self.learning_rate_compliance = lr_compliance
        self.learning_rate_behavior_mean = lr_mu

        # Softmax policy temperature control
        self.policy_temperature = 1.0
        self.min_policy_temperature = 0.1
        self.policy_temp_decay = policy_temp_decay

        # Exploration soft focus
        self.softness_mu = softness_mu
        self.softness_goal = softness_goal

        self.action_history = []
        self.prev_suggestion_idx = None

    def _update_behavior_mean(self, action):
        """
        Update estimated user preference (μ̂) based on recent actions.
        """
        self.action_history.append(action)
        window = 10
        if len(self.action_history) >= window:
            self.estimated_behavior_mean = np.mean(self.action_history[-window:])
        else:
            self.estimated_behavior_mean += self.learning_rate_behavior_mean * (action - self.estimated_behavior_mean)
        self.estimated_behavior_mean = np.clip(self.estimated_behavior_mean, 0.0, 5.0)

    def _update_compliance(self, actual_compliance):
        """
        Update agent's estimate of user compliance.
        """
        error = actual_compliance - self.estimated_compliance
        self.estimated_compliance += self.learning_rate_compliance * error
        self.estimated_compliance = np.clip(self.estimated_compliance, 0.0, 1.0)

    def _update_q(self, suggestion_idx, reward):
        """
        Q-learning update.
        """
        old_q = self.q_values[suggestion_idx]
        self.q_values[suggestion_idx] += self.learning_rate_q * (reward - old_q)

    def _decay_temperature(self):
        """
        Adjust policy temperature based on distance from goal.
        """
        mu_distance = abs(self.estimated_behavior_mean - self.goal)
        max_distance = np.ptp(self.action_space)
        ratio = np.clip(mu_distance / max_distance, 0.0, 1.0)
        target_temp = self.min_policy_temperature + ratio * (1.0 - self.min_policy_temperature)
        self.policy_temperature = (1 - self.policy_temp_decay) * self.policy_temperature + self.policy_temp_decay * target_temp

    def reward(self, suggestion_idx, action, actual_compliance):
        """
        Calculate composite reward based on:
        - compliance with suggestion
        - proximity of action and behavior mean to goal
        """
        self._update_compliance(actual_compliance)
        self._update_behavior_mean(action)

        goal_score = np.exp(-((action - self.goal) ** 2))
        mu_score = np.exp(-((self.estimated_behavior_mean - self.goal) ** 2))
        suggestion_score = np.exp(-((self.action_space[suggestion_idx] - self.goal) ** 2))

        total_reward = (
            self.reward_weight_compliance * actual_compliance +
            self.reward_weight_goal * goal_score +
            self.penalty_weight_behavior_mean * mu_score +
            self.penalty_weight_suggestion * suggestion_score
        )

        self._update_q(suggestion_idx, total_reward)
        return total_reward, actual_compliance

    def policy(self):
        """
        Softmax action selection influenced by both estimated user mean (μ̂) and goal (G).
        """
        scaled_q = self.q_values / self.policy_temperature

        idx_mu = np.argmin(np.abs(self.action_space - self.estimated_behavior_mean))
        idx_goal = np.argmin(np.abs(self.action_space - self.goal))

        weight_mu = np.exp(-self.softness_mu * np.abs(np.arange(len(self.action_space)) - idx_mu))
        weight_goal = np.exp(-self.softness_goal * np.abs(np.arange(len(self.action_space)) - idx_goal))

        smooth_weight = weight_mu * weight_goal
        adjusted_scaled = scaled_q + np.log(smooth_weight + 1e-8)

        probs = np.exp(adjusted_scaled - np.max(adjusted_scaled))
        probs /= np.sum(probs)

        suggestion_idx = np.random.choice(len(self.action_space), p=probs)
        self.prev_suggestion_idx = suggestion_idx
        suggestion = self.action_space[suggestion_idx]

        self._decay_temperature()
        return suggestion, suggestion_idx, probs
