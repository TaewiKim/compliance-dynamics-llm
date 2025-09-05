# user.py (최종 수정 버전)

from typing import Dict, Optional, List
import os, math

class UserLlm:
    """
    Generates LLM prompts for adaptive users receiving dietary coaching.
    Can work with various LLM providers via a unified HTTP API interface.
    """

    def __init__(self, user_profile: Dict[str, str], model_name: str = "gpt-4-turbo"):
        self.user_profile = user_profile
        self.model_name = model_name
        self.api_url = f"https://api.openai.com/v1/chat/completions"
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _tail_str(self, seq: Optional[List[float]], n: int = 5) -> str:
        """Helper function to format the tail of a list of numbers."""
        if not seq:
            return "없음"
        nums = [float(x) for x in seq if isinstance(x, (int, float)) and not math.isnan(x)]
        tail = nums[-n:]
        return ", ".join(f"{v:.2f}" for v in tail) if tail else "없음"

    def format_user_prompt(
        self,
        recommendation_history: Optional[List[float]] = None,
        action_history: Optional[List[float]] = None,
        history_window: int = 5
    ) -> str:
        """대화 중 사용자의 발화(utterance) 생성을 위한 프롬프트"""
        rec_tail = self._tail_str(recommendation_history, history_window)
        act_tail = self._tail_str(action_history, history_window)

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
        Generate your response in **JSON format** with the following keys. **Do not include the 'action' key here.**

        \```json
        {{
            "utterance": "What the user would say out loud, e.g., a sentence or two",
            "endkey": true or false // true if the user wants to end the conversation for this session
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
        """대화 세션이 끝난 후, 사용자의 최종 행동(action) 결정을 위한 프롬프트 (수정됨)"""
        
        dialogue = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history])
        
        # 행동 결정을 위해 이력 추가
        rec_tail = self._tail_str(recommendation_history, history_window)
        act_tail = self._tail_str(action_history, history_window)

        return f"""
        ## 🧠 User Action Decision

        You are the user described in the profile below. You have just finished a conversation with your AI coach.
        Based on your personality, the conversation, and your recent action history, decide what your final eating action will be for this session.

        ### 👤 Your User Profile
        - **Age**: {self.user_profile['age']}
        - **Gender**: {self.user_profile['gender']}
        - **Condition**: {self.user_profile['condition']}
        - **Typical Habits (μ)**: {self.user_profile['mu']}
        - **Suggestion Sensitivity (β)**: {self.user_profile['beta']}
        - **Adaptability (α)**: {self.user_profile['alpha']}
        - **Emotional Sensitivity (γ)**: {self.user_profile['gamma']}
        - **Irregularity Tendency (ε)**: {self.user_profile['epsilon']}

        ### 📈 Recent Context (for realism)
        - Recent agent numeric recommendations (last {history_window}): [{rec_tail}]
        - Your own actual behavior numeric actions (last {history_window}): [{act_tail}]

        ### 💬 Full Conversation Log of This Session
        {dialogue}

        ### 🎯 Your Task
        Reflect on the conversation, your personality traits, and your past actions. Now, make a final, single decision on your dietary action.
        
        ### ✏️ Output Instructions:
        Generate your response in **JSON format** with only the following key:

        \```json
        {{
            "action": 1.0 to 5.0 // Your final decided action. 1.0 = very unhealthy, 5.0 = very healthy.
        }}
        \```
        """.strip()