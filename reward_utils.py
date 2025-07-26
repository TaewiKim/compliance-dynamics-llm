import numpy as np

def compute_reward(
    suggestion_idx: int,
    action: float,
    actual_compliance: float,
    behavior_mean: float,
    action_space: np.ndarray,
    goal: float = 4.0,
    reward_weight_compliance: float = 0.1,
    reward_weight_goal: float = 1.5,
    reward_weight_behavior_mean: float = 0.1,
    reward_weight_suggestion: float = 0.1,
):
    """
    Composite reward function used in AdaptiveUser.

    Args:
        suggestion_idx (int): index of agent's suggested action
        action (float): actual user action
        actual_compliance (float): probability of compliance
        behavior_mean (float): current behavior mean
        action_space (np.ndarray): full action space
        goal (float): desired target behavior
        reward_weight_compliance (float): weight for compliance term
        reward_weight_goal (float): weight for goal proximity term
        reward_weight_behavior_mean (float): reward for behavior mean offset
        reward_weight_suggestion (float): reward for on-target suggestions

    Returns:
        float: total reward
        dict: intermediate values for debugging/logging
    """
    goal_score = np.exp(-((action - goal) ** 2))
    mu_distance_penalty = (behavior_mean - goal) ** 2
    suggestion_distance_penalty = (action_space[suggestion_idx] - goal) ** 2

    total_reward = (
        reward_weight_compliance * actual_compliance +
        reward_weight_goal * goal_score -
        reward_weight_behavior_mean * mu_distance_penalty -
        reward_weight_suggestion * suggestion_distance_penalty
    )

    details = {
        "compliance": actual_compliance,
        "goal_score": goal_score,
        "mu_distance_penalty": mu_distance_penalty,
        "suggestion_distance_penalty": suggestion_distance_penalty,
        "total_reward": total_reward
    }
    return total_reward, details
