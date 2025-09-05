# user.py (최종 수정 버전)
from typing import Dict, Optional, List
import os, math

class UserLlm:
    """
    Generates LLM prompts for adaptive users receiving dietary coaching.
    """

    def __init__(self, user_profile: Dict[str, str], model_name: str = "gpt-5-nano"):
        self.user_profile = user_profile
        self.model_name = model_name
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _tail_str(self, seq: Optional[List[float]], n: int = 5) -> str:
        if not seq:
            return "None"
        nums = [float(x) for x in seq if isinstance(x, (int, float)) and not math.isnan(x)]
        tail = nums[-n:]
        return ", ".join(f"{v:.2f}" for v in tail) if tail else "None"

    def format_user_prompt(
        self,
        recommendation_history: Optional[List[float]] = None,
        action_history: Optional[List[float]] = None,
        history_window: int = 5
    ) -> str:
        rec_tail = self._tail_str(recommendation_history, history_window)
        act_tail = self._tail_str(action_history, history_window)

        return f"""
        ## 🧑‍⚕️ Dietary Coaching User Profile
        You are a user receiving coaching from an AI to improve your dietary habits.

        - **Your Age**: {self.user_profile['age']}
        - **Your Gender**: {self.user_profile['gender']}
        - **Your Condition**: {self.user_profile['condition']}
        - **Your typical habits (μ)** are described as: {self.user_profile['mu']}
        - **You tend to be (β)**: {self.user_profile['beta']} in response to suggestions.
        - **When a pattern is maintained, you (α)**: {self.user_profile['alpha']}
        - **You are (γ)**: {self.user_profile['gamma']} to external influences.
        - **You can recall patterns for (memory)**: {self.user_profile['memory']}
        - **For patterns to change, you need (δ)**: {self.user_profile['delta']}
        - **In unexpected situations, you may show (ε)**: {self.user_profile['epsilon']}

        ### 📈 Recent Context
        - Recent agent recommendations (last {history_window}): [{rec_tail}]
        - My actual actions (last {history_window}): [{act_tail}]

        ### 🎯 Your Task
        React to the agent's latest suggestion in a natural way that is consistent with your profile.

        ### ✏️ Output Instructions
        Generate your response in **JSON format**. **Do not include the 'action' key here.**

        \```json
        {{
            "utterance": "What you would say out loud to the agent.",
            "endkey": false
        }}
        \```
        """.strip()

    def format_user_action_prompt(
        self,
        conversation_history: List[Dict[str, str]],
        recommendation_history: Optional[List[float]] = None,
        action_history: Optional[List[float]] = None,
        history_window: int = 5
    ) -> str:
        dialogue = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history])
        rec_tail = self._tail_str(recommendation_history, history_window)
        act_tail = self._tail_str(action_history, history_window)

        return f"""
        ## 🧠 User Action Decision
        You are the user described below. You have just finished a conversation with your AI coach.
        Based on your personality, the conversation, and your history, decide on your final action for this session.

        ### 👤 Your User Profile
        - **Condition**: {self.user_profile['condition']}
        - **Suggestion Sensitivity (β)**: {self.user_profile['beta']}
        - **Emotional Sensitivity (γ)**: {self.user_profile['gamma']}

        ### 📈 Recent Context
        - Recent agent recommendations (last {history_window}): [{rec_tail}]
        - Your own actual actions (last {history_window}): [{act_tail}]

        ### 💬 Full Conversation Log of This Session
        {dialogue}

        ### 🎯 Your Task
        Reflect on everything. Now, make a final, single decision on your dietary action.
        
        ### ✏️ Output Instructions:
        Generate your response in **JSON format** with only the following key:

        \```json
        {{
            "action": 3.5 // Your final decided action from 1.0 (very unhealthy) to 5.0 (very healthy).
        }}
        \```
        """.strip()