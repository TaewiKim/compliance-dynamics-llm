import numpy as np
import os

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
        self.inferred_user_profile = {}

        # Initialize the agent's model and API settings
        self.model_name = "gpt-4o-mini"
        self.api_url = f"https://api.openai.com/v1/chat/completions"
        self.api_key = os.getenv("OPENAI_API_KEY") 
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

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

    def format_agent_1st_session_prompt(self) -> str:
        return """
            You are a behavior coaching agent for dietary improvement. This is your first session with a user.

            Your goal is to **gently guide a natural conversation** to discover the user's background and eating behavior traits.  
            Ask about one topic at a time in a warm and conversational tone.

            You're trying to fill in the following **user profile attributes**:

            1. General Demographics:
            - `age`: Age group
            - `gender`: Gender identity

            2. Health-Related Information:
            - `condition`: Known health condition related to diet

            3. Dietary Behavior Traits:
            - `mu`: Regularity of current eating behavior
            - `beta`: Sensitivity to external suggestion
            - `alpha`: Flexibility in adapting to new habits
            - `gamma`: Sensitivity to emotional/environmental factors
            - `memory`: Recall capacity for eating patterns
            - `delta`: Stability requirement for change
            - `epsilon`: Likelihood of unexpected or irregular behaviors

            At each step:
            - Ask just **one question** targeting a single attribute.
            - After receiving a user response, store the interpreted value in the `inferred_attributes` field (if reasonably identifiable).
            - If the user’s reply is unclear, you may leave that attribute empty for now.
            - Continue this dialogue until either all attributes are filled or the session ends.

            ### Output format (JSON):
            ```json
            {
                "utterance": "What the agent says to the user",
                "monologue": "Agent’s internal reasoning or reflection",
                "endkey": false,
                "inferred_attributes": {
                    "age": null or "30s",
                    "gender": null or "Female",
                    "condition": null or "Binge Eating Disorder (BED)",
                    "mu": null or "Moderately irregular",
                    "beta": null or "Highly suggestible",
                    "alpha": null,
                    "gamma": null,
                    "memory": null,
                    "delta": null,
                    "epsilon": null
                }
            }
            ```
        """.strip()
    
    def format_agent_prompt(self, goal_behavior: float) -> str:
        return f"""
        You are a dietary behavior coaching agent in an ongoing session with a user.  
        The user's behavioral profile has already been inferred during a previous session.

        Your job now is to analyze the user's current behavioral tendencies and gently suggest  
        a healthy dietary action toward the target behavior level of **{goal_behavior:.1f}** (on a scale from 1.0 to 5.0).  
        Your suggestion should consider the user's unique tendencies as described below:

        ## 🧠 User Behavioral Profile

        - **Age Group**: {self.user_profile.get('age', 'Unknown')}
        - **Gender**: {self.user_profile.get('gender', 'Unknown')}
        - **Diet-related Condition**: {self.ser_profile.get('condition', 'Unknown')}

        ### 🧬 Dietary Behavior Traits:
        - **Eating Behavior Regularity (μ)**: {self.user_profile.get('mu', 'Unknown')}
        - **Suggestion Sensitivity (β)**: {self.user_profile.get('beta', 'Unknown')}
        - **Habit Adaptability (α)**: {self.user_profile.get('alpha', 'Unknown')}
        - **Environmental/Emotional Sensitivity (γ)**: {self.user_profile.get('gamma', 'Unknown')}
        - **Behavior Recall Span (Memory)**: {self.user_profile.get('memory', 'Unknown')}
        - **Stability Requirement for Change (Δ)**: {self.user_profile.get('delta', 'Unknown')}
        - **Irregular Behavior Tendency (ε)**: {self.user_profile.get('epsilon', 'Unknown')}

        ---

        ## 🎯 Instructions:

        Generate a natural, warm response (utterance) that suggests a small but meaningful dietary action aligned with the target goal.  
        Also include your **internal reasoning** (monologue) on how the user's traits influenced your choice.

        ### Output Format (JSON):

        ```json
        {{
        "utterance": "A supportive, conversational suggestion (1~2 sentences)",
        "monologue": "Your internal reflection explaining the reasoning",
        "endkey": false
        }}
        """.strip()

    def format_agent_session_analysis_prompt(self, session_log: list) -> str:
        return f"""
    You are a behavior coaching analyst reviewing a completed session between an AI agent and a user.

    Your task is to analyze the conversation and produce a structured psychological profile that will guide future coaching sessions.

    The session is provided below in JSON format. It includes a turn-by-turn log of both the agent's and the user's utterances, monologues, and flags indicating whether the session should end.

    ---

    ### 🔍 Analyze the session for:

    1. **Cognitive Dissonance**  
    - Does the user express any internal conflict (e.g., "I want to change but I can’t")?

    2. **Negative Thought Patterns**  
    - Identify self-defeating beliefs, hopelessness, guilt, avoidance, or all-or-nothing thinking.

    3. **Emotional Triggers**  
    - What emotional states or contextual factors (e.g., stress, loneliness) appear to precede unhealthy behaviors?

    4. **Effective Reinforcement**  
    - Which types of agent responses (empathy, praise, gentle challenge, etc.) led to positive shifts in the user’s tone or openness?

    5. **Coaching Notes**  
    - What should the next agent session keep in mind in terms of tone, pacing, and strategy?

    ---

    ### 🧾 Output Format (JSON)

    ```json
    {{
    "cognitive_dissonance": "A clear, concise summary of the user’s internal conflicts",
    "negative_thought_patterns": "Summary of recurring negative beliefs or emotional responses",
    "emotional_triggers": "Factors that seem to drive unhealthy eating or resistance to change",
    "effective_reinforcement": ["Empathy", "Praise", "Normalizing setbacks"], 
    "coaching_notes": "Recommendations for how the agent should communicate in the next session"
    }}
    """.strip()
