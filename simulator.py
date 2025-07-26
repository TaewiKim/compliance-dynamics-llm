import numpy as np
import matplotlib.pyplot as plt
import os
import time

class Simulator:
    """
    Coordinates the interaction between a psychologically grounded user model
    and a reinforcement learning-based agent. Responsible for warm-up (exploratory)
    initialization, training iterations, and comprehensive logging and visualization.

    Parameters:
    ----------
    user : AdaptiveUser
        The user model (compliance-aware and adaptively learning).
    agent : Agent
        The reinforcement learning agent providing behavioral suggestions.
    action_space : np.ndarray
        Discretized set of possible actions (suggestions).
    total_steps : int
        Number of time steps for simulation (includes warmup).
    warmup_steps : int
        Initial steps where random actions are used to initialize user behavior.
    """

    def __init__(self, user, agent, action_space, total_steps=400, warmup_steps=10):
        self.user = user
        self.agent = agent
        self.action_space = action_space
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

        self._init_logs()

    def _init_logs(self):
        self.suggestion_trace = []
        self.action_trace = []
        self.reward_trace = []
        self.compliance_trace = []
        self.estimated_compliance_trace = []
        self.true_mu_trace = []
        self.estimated_mu_trace = []
        self.temperature_trace = []
        self.q_value_trace = []

    def warmup(self):
        """
        Initialize user behavior with random suggestions.
        Estimate initial agent goal based on observed behavior.
        """
        warmup_data = [
            (s := np.random.choice(self.action_space), self.user.respond(s))
            for _ in range(self.warmup_steps)
        ]
        _, actions = zip(*warmup_data)
        self.goal = np.mean(actions)  # Optional: agent sets goal behavior from warm-up
        for suggestion, action in warmup_data:
            self._log(suggestion, action, None, None)

    def train(self):
        """
        Main simulation loop where the agent interacts with the user,
        suggests actions, observes compliance, and updates internal estimates.
        """
        for t in range(self.warmup_steps, self.total_steps):
            print(f"\nStep {t - self.warmup_steps + 1} -------------------------------")

            suggestion, suggestion_idx, _ = self.agent.policy()
            action = self.user.respond(suggestion)
            actual_compliance = self.user.compliance_prob(suggestion)

            reward, _ = self.agent.reward(suggestion_idx, action, actual_compliance)

            print(f"Suggestion: {suggestion:.2f}, Action: {action:.2f}, Compliance: {actual_compliance:.4f}, Reward: {reward:.4f}")

            self._log(suggestion, action, reward, actual_compliance)

        self.save_log("simulation_log.txt")

    def _log(self, suggestion, action, reward, actual_compliance):
        self.suggestion_trace.append(suggestion)
        self.action_trace.append(action)
        self.reward_trace.append(reward)
        self.compliance_trace.append(actual_compliance)
        self.estimated_compliance_trace.append(self.agent.estimated_compliance)
        self.true_mu_trace.append(self.user.behavior_mean)
        self.estimated_mu_trace.append(self.agent.estimated_behavior_mean)
        self.temperature_trace.append(self.agent.policy_temperature)
        self.q_value_trace.append(self.agent.q_values.copy())

    def save_log(self, filename="simulation_log.txt"):
        with open(filename, "w") as f:
            # 헤더에 Q-value 최대값 추가
            f.write("Step\tSuggestion\tAction\tCompliance\tReward\tTrue_mu\tEstimated_mu\tTemperature\tQ_max\n")

            for i in range(len(self.suggestion_trace)):
                def fmt(val):
                    return f"{val:.4f}" if val is not None else "NA"

                q_val_max = np.max(self.q_value_trace[i]) if i < len(self.q_value_trace) else None

                f.write(
                    f"{i+1}\t"
                    f"{fmt(self.suggestion_trace[i])}\t"
                    f"{fmt(self.action_trace[i])}\t"
                    f"{fmt(self.compliance_trace[i])}\t"
                    f"{fmt(self.reward_trace[i])}\t"
                    f"{fmt(self.true_mu_trace[i])}\t"
                    f"{fmt(self.estimated_mu_trace[i])}\t"
                    f"{fmt(self.temperature_trace[i])}\t"
                    f"{fmt(q_val_max)}\n"
                )

    def plot(self, save=False, filename=None):
        """
        Generate 5-panel visualization:
        - Agent suggestion vs user behavior
        - Reward over time
        - Compliance tracking
        - Policy temperature
        - Estimated vs true behavior mean
        """

        fig, axs = plt.subplots(6, 1, figsize=(12, 16))
        alpha = self.user.adaptation_rate
        beta = self.user.compliance_sensitivity
        gamma = self.user.noise_sensitivity
        user_info = r"$(\alpha=" + f"{alpha},\\beta={beta},\\gamma={gamma})$"

        axs[0].plot(self.suggestion_trace, label='Agent Suggestion', color='orange')
        axs[0].scatter(range(len(self.action_trace)), self.action_trace, label='User Action', color='blue', s=10)
        axs[0].axhline(self.user.behavior_mean, linestyle='--', color='gray', label=r'Final True $\mu$')
        axs[0].axhline(self.user.initial_behavior_mean, linestyle=':', color='purple', label=r'Initial $\mu_0$')
        axs[0].axhline(self.agent.goal, linestyle='--', color='red', label=r'Agent Goal $G$')
        axs[0].axvline(self.warmup_steps, linestyle=':', color='black', label='Warm-up End')
        axs[0].set_ylabel("Behavior Value")
        axs[0].set_title(rf"Agent Suggestion vs Adaptive User Behavior {user_info}")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(self.reward_trace, label='Reward', color='blue')
        axs[1].axvline(self.warmup_steps, linestyle=':', color='black')
        axs[1].set_ylabel("Reward")
        axs[1].set_title("Reward Dynamics Over Time")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(self.compliance_trace, label='Actual Compliance', color='darkgreen')
        axs[2].plot(self.estimated_compliance_trace, label='Estimated Compliance', color='magenta', linestyle='--')
        axs[2].axvline(self.warmup_steps, linestyle=':', color='black')
        axs[2].set_ylabel("Compliance")
        axs[2].set_title("Compliance Estimation Over Time")
        axs[2].legend()
        axs[2].grid(True)

        axs[3].plot(self.temperature_trace, label='Policy Temperature', color='brown')
        axs[3].set_ylabel("Temperature")
        axs[3].set_title("Softmax Temperature Evolution")
        axs[3].legend()
        axs[3].grid(True)

        axs[4].plot(self.estimated_mu_trace, label=r'Estimated $\hat{\mu}$', color='green')
        axs[4].plot(self.true_mu_trace, linestyle='--', color='gray', label=r'True $\mu$')
        axs[4].axvline(self.warmup_steps, linestyle=':', color='black', label='Warm-up End')
        axs[4].set_ylabel("Behavior Mean")
        axs[4].set_title(r"Estimated vs Actual Behavior Mean ($\hat{\mu}$ vs $\mu$)")
        axs[4].legend()
        axs[4].grid(True)

        # --- 6. Max Q-value over time ---
        q_values_over_time = np.array(self.q_value_trace)  # shape: (timesteps, num_actions)

        # 각 timestep에서 Q-value 중 최댓값 추출
        max_q_values = np.max(q_values_over_time, axis=1)

        axs[5].plot(max_q_values, label='Max Q-value', color='navy')
        axs[5].set_ylabel("Max Q-value")
        axs[5].set_title("Maximum Q-value Over Time")
        axs[5].axvline(self.warmup_steps, linestyle=':', color='black', label='Warm-up End')
        axs[5].legend()
        axs[5].grid(True)

        plt.xlabel("Time Step")
        plt.tight_layout()

        if save and filename:
            # Save PNG
            png_dir = "plots"
            os.makedirs(png_dir, exist_ok=True)
            png_path = os.path.join(png_dir, f"{filename}.png")
            plt.savefig(png_path)
            return png_path
        
        plt.show()
