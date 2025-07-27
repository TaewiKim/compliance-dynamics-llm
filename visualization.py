# visualization.py

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_simulation(simulator, save=False, filename=None):
    """
    Generates visualization of simulation traces from a Simulator object.

    Parameters
    ----------
    simulator : Simulator
        An instance of the Simulator class with logged traces.
    save : bool
        Whether to save the plot as an image file.
    filename : str or None
        Name of the file to save if save=True.
    """

    fig, axs = plt.subplots(6, 1, figsize=(12, 16))
    alpha = simulator.user.adaptation_rate
    beta = simulator.user.compliance_sensitivity
    gamma = simulator.user.noise_sensitivity
    user_info = r"$(\alpha=" + f"{alpha},\\beta={beta},\\gamma={gamma})$"

    axs[0].plot(simulator.suggestion_trace, label='Agent Suggestion', color='orange')
    axs[0].scatter(range(len(simulator.action_trace)), simulator.action_trace, label='User Action', color='blue', s=10)
    axs[0].axhline(simulator.user.behavior_mean, linestyle='--', color='gray', label=r'Final True $\mu$')
    axs[0].axhline(simulator.user.initial_behavior_mean, linestyle=':', color='purple', label=r'Initial $\mu_0$')
    axs[0].axhline(simulator.agent.goal, linestyle='--', color='red', label=r'Agent Goal $G$')
    axs[0].axvline(simulator.warmup_steps, linestyle=':', color='black', label='Warm-up End')
    axs[0].set_ylabel("Behavior Value")
    axs[0].set_title(rf"Agent Suggestion vs Adaptive User Behavior {user_info}")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(simulator.reward_trace, label='Reward', color='blue')
    axs[1].axvline(simulator.warmup_steps, linestyle=':', color='black')
    axs[1].set_ylabel("Reward")
    axs[1].set_title("Reward Dynamics Over Time")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(simulator.compliance_trace, label='Actual Compliance', color='darkgreen')
    axs[2].plot(simulator.estimated_compliance_trace, label='Estimated Compliance', color='magenta', linestyle='--')
    axs[2].axvline(simulator.warmup_steps, linestyle=':', color='black')
    axs[2].set_ylabel("Compliance")
    axs[2].set_title("Compliance Estimation Over Time")
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(simulator.temperature_trace, label='Policy Temperature', color='brown')
    axs[3].set_ylabel("Temperature")
    axs[3].set_title("Softmax Temperature Evolution")
    axs[3].legend()
    axs[3].grid(True)

    axs[4].plot(simulator.estimated_mu_trace, label=r'Estimated $\hat{\mu}$', color='green')
    axs[4].plot(simulator.true_mu_trace, linestyle='--', color='gray', label=r'True $\mu$')
    axs[4].axvline(simulator.warmup_steps, linestyle=':', color='black', label='Warm-up End')
    axs[4].set_ylabel("Behavior Mean")
    axs[4].set_title(r"Estimated vs Actual Behavior Mean ($\hat{\mu}$ vs $\mu$)")
    axs[4].legend()
    axs[4].grid(True)

    q_values_over_time = np.array(simulator.q_value_trace)
    max_q_values = np.max(q_values_over_time, axis=1)

    axs[5].plot(max_q_values, label='Max Q-value', color='navy')
    axs[5].set_ylabel("Max Q-value")
    axs[5].set_title("Maximum Q-value Over Time")
    axs[5].axvline(simulator.warmup_steps, linestyle=':', color='black', label='Warm-up End')
    axs[5].legend()
    axs[5].grid(True)

    plt.xlabel("Time Step")
    plt.tight_layout()

    if save and filename:
        png_dir = "plots"
        os.makedirs(png_dir, exist_ok=True)
        png_path = os.path.join(png_dir, f"{filename}.png")
        plt.savefig(png_path)
        return png_path

    plt.show()
