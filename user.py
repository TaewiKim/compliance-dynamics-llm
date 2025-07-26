# user.py
import numpy as np

class AdaptiveUser:
    """
    Simulates a psychologically grounded adaptive user.
    The user has an internal behavior preference (μ) and 
    probabilistically follows suggestions from an agent based on psychological distance.
    """

    def __init__(self, mu=2.0, beta=0.05, alpha=0.0, gamma=0.5, memory=10, delta=0.2, epsilon=1e-3):
        self.initial_behavior_mean = mu     # μ₀: Initial preference
        self.behavior_mean = mu             # μ: Current internal behavior mean
        self.compliance_sensitivity = beta  # β: Sensitivity to suggestion distance
        self.adaptation_rate = alpha        # α: Speed of adaptation toward consistent behavior
        self.noise_sensitivity = gamma      # γ: Noise scaling depending on (1 - compliance)
        self.memory = memory                # Number of past actions for stability check
        self.convergence_threshold = delta  # δ: Variance threshold to trigger adaptation
        self.min_noise = epsilon            # ε: Minimum noise level
        self.history = []                   # Memory of recent user actions

    def compliance_prob(self, suggestion):
        """
        Compute user's compliance probability based on psychological distance.
        """
        return np.exp(-self.compliance_sensitivity * (suggestion - self.behavior_mean) ** 2)

    def respond(self, suggestion):
        """
        Generate a user action in response to an agent suggestion.
        Combines internal preference, suggestion, and stochastic noise.
        """
        compliance = self.compliance_prob(suggestion)
        noise_std = self.noise_sensitivity * (1 - compliance) + self.min_noise
        noise = np.random.normal(0, noise_std)

        action = (1 - compliance) * self.behavior_mean + compliance * suggestion + noise
        action = np.clip(action, 0.0, 5.0)

        self.history.append(action)
        if len(self.history) > self.memory:
            self.history.pop(0)

        if len(self.history) == self.memory:
            std_recent = np.std(self.history)
            if std_recent < self.convergence_threshold:
                # Update μ toward mean of stable actions
                delta_mu = np.sign(np.mean(self.history) - self.behavior_mean)
                self.behavior_mean += self.adaptation_rate * delta_mu

        return action
