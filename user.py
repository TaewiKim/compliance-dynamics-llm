# user.py
from typing import Dict, Optional
import os

class UserLlm:
    """
    Generates LLM prompts for adaptive users receiving dietary coaching.
    Can work with various LLM providers via a unified HTTP API interface.
    """

    def __init__(self, user_profile: Dict[str, str]):

        self.user_profile = user_profile
        self.model_name = "gpt-4o-mini"
        self.api_url = f"https://api.openai.com/v1/chat/completions"
        self.api_key = os.getenv("OPENAI_API_KEY") 
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def format_user_prompt(self) -> str:
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

        The AI uses this behavioral profile to provide personalized dietary coaching tailored to your needs.

        ### ✏️ Output Instructions:
        Generate your response in **JSON format** with the following keys:

        \\```json
        {{
        "utterance": "What the user would say out loud, e.g., a sentence or two",
        "monologue": "What the user is thinking internally or emotionally",
        "endkey": true or false, // true if the user takes an eating action
        "action": 1.0 to 5.0     // only include if endkey is true. 5.0 = very healthy eating, 1.0 = very unhealthy
        }}
        \\```
                """.strip()


