# user.py

from typing import Dict, Optional, List
import os, math

class UserLlm:
    """
    Generates LLM prompts for adaptive users receiving dietary coaching.
    Can work with various LLM providers via a unified HTTP API interface.
    """

    def __init__(self, user_profile: Dict[str, str], model_name: str = "gpt-5-nano"):
        self.user_profile = user_profile
        self.model_name = model_name
        self.api_url = f"https://api.openai.com/v1/chat/completions"
        self.api_key = os.getenv("OPENAI_API_KEY") 
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def format_user_prompt(
        self,
        recommendation_history: Optional[List[float]] = None,
        action_history: Optional[List[float]] = None,
        history_window: int = 5
    ) -> str:
        recommendation_history = recommendation_history or []
        action_history = action_history or []

        def _tail_str(seq, n=5):
            nums = []
            for x in seq:
                if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
                    nums.append(float(x))
            tail = nums[-n:]
            return ", ".join(f"{v:.2f}" for v in tail) if tail else "없음"

        rec_tail = _tail_str(recommendation_history, history_window)
        act_tail = _tail_str(action_history, history_window)

        return f"""
        ## 🧑‍⚕️ Dietary Coaching User Profile

        You are a user receiving coaching from an AI to improve your dietary habits.

        Your age is **{self.user_profile['age']}**, and your gender is **{self.user_profile['gender']}**.  
        You are currently experiencing issues related to **{self.user_profile['condition']}**.

        Your typical eating habits are described as **{self.user_profile['mu']}**,  
        and you tend to **{self.user_profile['beta']}** in response to dietary suggestions from the AI.  
        When a consistent eating pattern is maintained, you **{self.user_profile['alpha']}** in adapting to new habits.  
        However, you are **{self.user_profile['gamma']}** in response to external influences such as emotional states or environmental changes.

        You can recall your recent eating patterns for approximately **{self.user_profile['memory']}**,  
        and if the patterns are stable, you show **{self.user_profile['delta']}** likelihood of change.  
        Nevertheless, in unexpected situations, you may still exhibit **{self.user_profile['epsilon']}** levels of irregular eating behavior.

        ### 📈 Recent Context (for realism)
        - Recent agent numeric recommendations (last {history_window}): [{rec_tail}]
        - My actual behavior numeric actions (last {history_window}): [{act_tail}]

        Please speak naturally as the user would, reacting to the agent's latest suggestion and your recent context.

        ### ✏️ Output Instructions:
        Generate your response in **JSON format** with the following keys:

        \\```json
        {{
            "utterance": "What the user would say out loud, e.g., a sentence or two",
            "endkey": true or false, // true if the user takes an eating action
            "action": 1.0 to 5.0     // only include if endkey is true. 5.0 = very healthy eating, 1.0 = very unhealthy
        }}
        \\```
        """.strip()
