# agent.py

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
        policy_temp_decay=0.04,
        user_age=None,
        user_gender=None,
        model_name="gpt-5-nano",
    ):
        self.action_space = np.array(action_space)
        self.num_actions = len(action_space)
        self.q_values = np.zeros(self.num_actions)
        self.goal_behavior = goal_behavior

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
        self.user_profile = {
            "age": user_age,
            "gender": user_gender
        }

        self.inferred_user_profile = {}

        # Initialize the agent's model and API settings
        self.model_name = model_name
        self.api_url = f"https://api.openai.com/v1/chat/completions"
        self.api_key = os.getenv("OPENAI_API_KEY") 
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        self.run_context = {
            "session_id": None,        # 현재 세션 인덱스 (int)
            "current_turn": None,      # 현재 턴 (1-base)
            "max_turns": None,         # 세션당 최대 턴
            "total_sessions": None,    # 전체 세션 수(계획)
            "compliance_summary": {}   # 순응도 집계(아래 simulator에서 채움)
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
        mu_distance = abs(self.estimated_behavior_mean - self.goal_behavior)
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

        goal_score = np.exp(-((action - self.goal_behavior) ** 2))
        mu_score = np.exp(-((self.estimated_behavior_mean - self.goal_behavior) ** 2))
        suggestion_score = np.exp(-((self.action_space[suggestion_idx] - self.goal_behavior) ** 2))

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
        idx_goal = np.argmin(np.abs(self.action_space - self.goal_behavior))

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
        ctx = self.run_context or {}

        # 안전한 표시값
        def _d(v, fallback="Unknown"):
            return fallback if v is None else v

        return f"""
        You are a behavior coaching agent focused on dietary improvement. This is your **first** session with the user.

        ---
        ## ⏱ Session Context
        - Total planned sessions: {_d(ctx.get('total_sessions'))}
        - Current session index: {_d(ctx.get('session_id'))}
        - maximum turns per session: {_d(ctx.get('max_turns'))}
        - Current turn: {_d(ctx.get('current_turn'))} / {_d(ctx.get('max_turns'))}

        Your objective is to **gently guide a natural and empathetic conversation** to understand the user's background and eating behavior traits,  
        **and collaboratively agree on a desired goal behavior level** for future coaching.

        Maintain a warm, human-like tone and explore **one topic at a time**, in a back-and-forth conversational style.

        ---

        ## 🧭 Session Objectives:

        1. **Understand the User**  
        Discover their background and dietary behavior traits across the following dimensions:

        - `condition`: Known health conditions that influence diet  
        - `mu`: Regularity of current eating behaviors  
        - `beta`: Sensitivity to external suggestions (e.g., trends, social cues)  
        - `alpha`: Flexibility in adopting new habits  
        - `gamma`: Sensitivity to emotional/environmental factors  
        - `memory`: Ability to recall and reflect on eating patterns  
        - `delta`: Need for structure/stability in dietary change  
        - `epsilon`: Tendency for spontaneous or irregular eating behaviors  

        2. **Goal Alignment**  
        Toward the end of the session, guide the user to **reflect on their current habits** and help them **set a desired dietary behavior goal level**  
        (on a scale from **1.0 = very poor dietary behavior** to **5.0 = highly healthy and consistent**).

        ---

        ## 📝 Guidance:

        - Ask **only one question at a time**, targeting a single attribute.
        - Use **conversational, friendly, non-judgmental language** (avoid clinical terms).
        - For each response:
        - If reasonably clear, update the `inferred_attributes` dictionary.
        - If the response is unclear or ambiguous, leave the value as null for now.
        - Ask the goal-setting question **after most attributes have been explored**, and store the result as `goal_behavior`.

        You may end the session when:
        - All relevant attributes are reasonably inferred, **and**
        - A goal behavior level has been discussed and agreed upon, **or**
        - The session ends naturally due to user disengagement.

        ---

        ### Output Format (JSON):

        ```json
        {{
            "monologue": "Brief internal reflection keywords on what was inferred or next steps",
            "utterance": "What the agent says to the user",
            "endkey": false
        }}
        """.strip()
    
    def format_agent_1st_session_analysis_prompt(self) -> str:
        return """
        You are analyzing the user's responses from a first-time dietary behavior coaching session.

        Based on the conversation, populate the following `inferred_attributes` object with your best interpretation.  
        If any attribute is not clearly inferable from the conversation, leave it as null.

        ### User Profile Attributes:
        1. **Health-Related Information**
        - `condition`: Known health conditions that influence diet

        2. **Dietary Behavior Traits**
        - `mu`: Regularity of current eating behaviors
        - `beta`: Sensitivity to external suggestions (e.g., trends, social cues)
        - `alpha`: Flexibility in adopting new habits
        - `gamma`: Sensitivity to emotional/environmental factors
        - `memory`: Ability to recall and reflect on eating patterns
        - `delta`: Need for structure/stability in dietary change
        - `epsilon`: Tendency for spontaneous or irregular eating behaviors

        ### User Profile Output Format:
        ```json
        {
            "goal_behavior": 4.0,  // Default goal behavior level
            "inferred_attributes": {
                "condition": null or e.g. "Type 2 Diabetes",
                "mu": null or e.g. "Highly irregular",
                "beta": null or e.g. "Moderately suggestible",
                "alpha": null or e.g. "Flexible and open to change",
                "gamma": null or e.g. "Highly emotionally influenced",
                "memory": null or e.g. "Poor short-term recall",
                "delta": null or e.g. "Requires clear structure",
                "epsilon": null or e.g. "Often deviates from plans"
            }
        }
        """.strip()

    def format_agent_prompt(self, suggestion_score, suggestion_history, prior_analysis, planned_suggestion) -> str:
        ctx = self.run_context or {}
        comp = ctx.get("compliance_summary", {}) or {}

        def _d(v, fallback="Unknown"):
            return fallback if v is None else v

        return f"""
        You are a **dietary behavior coaching agent** working with a user in an ongoing session.  
        The user has already completed an initial profiling session, and their **behavioral traits** are available below.

        ---
        ## ⏱ Session Context
        - Total planned sessions: {_d(ctx.get('total_sessions'))}
        - Current session index: {_d(ctx.get('session_id'))}
        - Maximum turns per session: {_d(ctx.get('max_turns'))}
        - Current turn: {_d(ctx.get('current_turn'))} / {_d(ctx.get('max_turns'))}

        ## 📈 Compliance (so far)
        - Agent running estimate: {_d(comp.get('estimated_by_agent'), 'NA')}
        - Mean (all sessions): {_d(comp.get('mean'), 'NA')}
        - Recent mean (last 10): {_d(comp.get('recent_mean'), 'NA')}
        - Last observed: {_d(comp.get('last'), 'NA')}
        - Count: {_d(comp.get('count'), 0)}

        Your role now is to:
        - Interpret the user's current behavioral state (as measured by `suggestion_score`).
        - Reflect on their **recent history and past actions** (`suggestion_history`).
        - Consider your own **prior analysis or strategies tried** (`prior_analysis`).
        - Based on these, offer a **warm, encouraging suggestion** that nudges the user gently toward the **target behavior level of 5.0**.

        ---
        ## 🔢 Current Behavior Snapshot

        - **Planned Numeric Suggestion (internal, do not reveal the number)**: {planned_suggestion:.2f}
        - **Current Behavior Score**: {suggestion_score:.1f}
        - **Recent Suggestion History**:
            ```
            {suggestion_history}
            ```
        - **Prior Agent Analysis**:
            ```
            {prior_analysis}
            ```

        ---
        ## 🧠 User Behavioral Profile

        - **Age Group**: {self.user_profile.get('age', 'Unknown')}
        - **Gender**: {self.user_profile.get('gender', 'Unknown')}
        - **Diet-related Condition**: {self.inferred_user_profile.get('condition', 'Unknown')}

        ### 🧬 Dietary Behavior Traits:
        - **Eating Behavior Regularity (μ)**: {self.inferred_user_profile.get('mu', 'Unknown')}
        - **Suggestion Sensitivity (β)**: {self.inferred_user_profile.get('beta', 'Unknown')}
        - **Habit Adaptability (α)**: {self.inferred_user_profile.get('alpha', 'Unknown')}
        - **Emotional/Environmental Sensitivity (γ)**: {self.inferred_user_profile.get('gamma', 'Unknown')}
        - **Behavior Recall Span (Memory)**: {self.inferred_user_profile.get('memory', 'Unknown')}
        - **Stability Requirement for Change (Δ)**: {self.inferred_user_profile.get('delta', 'Unknown')}
        - **Irregular Behavior Tendency (ε)**: {self.inferred_user_profile.get('epsilon', 'Unknown')}

        ---
        ## 🎯 Instructions:

        - Generate a **brief, supportive, and actionable suggestion** that helps the user make a small, meaningful step toward healthier eating behavior.  
        - **Align** the suggestion qualitatively with the **planned numeric suggestion** shown above, but **do not reveal** the number itself.  
        - Tailor the suggestion using both their behavioral profile and current behavior score. Avoid repeating past suggestions from history.

        Also include an **internal monologue** explaining the reasoning behind your choice (e.g., considering traits like adaptability, impulsivity, etc.).

        ### Output Format (JSON):
        {{
            "monologue": "Brief CBT-style reasoning (e.g., 'Using behavioral activation to counter stress-triggered snacking.')",
            "utterance": "Warm, motivational suggestion for a small dietary improvement.",
            "endkey": false
        }}
        """.strip()

    def format_agent_session_analysis_prompt(self, last_suggestion: float) -> str:
        return f"""
        You are a behavior coaching analyst reviewing a completed session between an AI agent and a user.

        Your task:
        1) Infer the **user's actual eating action** taken in this session on a 1.0~5.0 scale.
        2) Estimate **compliance** with the agent's **last numeric suggestion** S = {last_suggestion:.2f}.
           - Compute compliance_estimate ∈ [0,1] as: 1 - |action_estimate - S| / 4.0 (clip to [0,1]).
        3) Produce a structured psychological profile that will guide future coaching sessions.

        The session is provided below in JSON format. It includes the turn-by-turn log of both the agent's and user's utterances.

        ---
        ### 🔍 Analyze the session for:
        1. **Cognitive Dissonance**  
           - Does the user express internal conflict (e.g., "I want to change but I can’t")?
        2. **Negative Thought Patterns**  
           - Identify self-defeating beliefs, hopelessness, guilt, avoidance, or all-or-nothing thinking.
        3. **Emotional Triggers**  
           - What emotional states or contextual factors (e.g., stress, loneliness) precede unhealthy behaviors?
        4. **Effective Reinforcement**  
           - Which types of agent responses (empathy, praise, gentle challenge, etc.) shifted the user’s tone or openness?
        5. **Coaching Notes**  
           - What should the next agent session keep in mind in terms of tone, pacing, and strategy?

        ---
        ### 🧾 Output Format (JSON)
        {{
            "user_action_estimate": 3.2,              // 1.0 ~ 5.0
            "compliance_estimate": 0.73,             // [0, 1]
            "confidence": 0.7,                        // how confident you are in this inference
            "basis": "Brief reason how you inferred the action from the dialogue",
            "cognitive_dissonance": "Concise summary of user's internal conflicts",
            "negative_thought_patterns": "Recurring negative beliefs or emotional responses",
            "emotional_triggers": "Factors driving unhealthy eating or resistance to change",
            "effective_reinforcement": ["Empathy", "Praise", "Normalizing setbacks"], 
            "coaching_notes": "Recommendations for next session tone/pacing/strategy"
        }}
        """.strip()

