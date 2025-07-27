import numpy as np
import matplotlib.pyplot as plt
import os
import json
import requests

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
        Number of time steps for simulation.
    """

    def __init__(self, user, agent, action_space, total_steps=400):
        self.user = user
        self.agent = agent
        self.action_space = action_space
        self.total_steps = total_steps
        self.session_steps = 1
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

    def generate_response(
        self,
        role: str,
        content: str,
        history: list,
        model_name: str,
        api_url: str,
        api_key: str,
        headers: dict,
    ) -> dict:
        assert role in ["user", "agent"], "Role must be either 'user' or 'agent'"

        messages = history + [{"role": "user" if role == "user" else "assistant", "content": content}]
        payload = {
            "model": model_name,
            "messages": messages,
        }

        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        message_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"[{role.capitalize()} Response]: {message_content}")

        # Strip triple backticks if present
        cleaned = message_content.strip().strip("`").strip("json").strip()
        parsed = json.loads(cleaned)

        return parsed

    def _generate_agent_turn(self, session_id, first_session):
        if first_session:
            prompt = self.agent.format_agent_1st_session_prompt()
            suggestion_score = None
        else:
            # Load previous session data
            prior_analysis_path = f"sessions/analysis_{session_id - 1:03}.json"
            prior_analysis = self._load_json(prior_analysis_path) or {}

            suggestion_history = []
            for sid in range(session_id):
                log = self._load_json(f"sessions/session_{sid:03}.json")
                if log:
                    for t in log:
                        if "suggestion_score" in t and "action" in t:
                            suggestion_history.append({
                                "turn": t["turn"],
                                "suggestion_score": t["suggestion_score"],
                                "user_action": t["action"]
                            })

            suggestion, _, prob = self.agent.policy()
            
            prompt = self.agent.format_agent_prompt(
                suggestion_score=suggestion,
                suggestion_history=suggestion_history,
                prior_analysis=prior_analysis
            )

        response = self.generate_response(
            role="agent",
            content=prompt,
            history=self.conversation_history,
            model_name=self.agent.model_name,
            api_url=self.agent.api_url,
            api_key=self.agent.api_key,
            headers=self.agent.headers,
        )

        return {
            "utterance": response.get("utterance", ""),
            "monologue": response.get("monologue", ""),
            "endkey": response.get("endkey", False),
            "inferred_attributes": response.get("inferred_attributes", {}),
        }

    def _generate_user_turn(self, agent_utterance, first_session):
        if first_session:
            prompt = self.user.format_user_prompt() + f"\n\n{agent_utterance}"
        else:
            prompt = self.user.format_prompt(self.user.profile) + f"\n\n{agent_utterance}"

        response = self.generate_response(
            role="user",
            content=prompt,
            history=self.conversation_history,
            model_name=self.user.model_name,
            api_url=self.user.api_url,
            api_key=self.user.api_key,
            headers=self.user.headers,
        )

        return {
            "utterance": response.get("utterance", ""),
            "monologue": response.get("monologue", ""),
            "endkey": response.get("endkey", False),
            "action": float(response["action"]) if response.get("endkey") and "action" in response else None
        }

    def _analyze_session(self, session_log, session_id):
        prompt = self.agent.format_agent_session_analysis_prompt(session_log)
        analysis = self.generate_response(
            role="agent",
            content=prompt,
            history=[],
            model_name=self.agent.model_name,
            api_url=self.agent.api_url,
            api_key=self.agent.api_key,
            headers=self.agent.headers,
        )
        print("\n[🧠 Session Analysis]:")
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
        with open(f"sessions/analysis_{session_id:03}.json", "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

    def _save_session_log(self, session_log, session_id, first_session):
        os.makedirs("sessions", exist_ok=True)
        session_type = "profile_session" if first_session else "session"
        path = f"sessions/{session_type}_{session_id:03}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session_log, f, ensure_ascii=False, indent=2)

    def _load_json(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def run_session(self, session_id: int, max_turns: int = 10, first_session: bool = False):
        self.conversation_history = []
        session_log = []

        inferred_user_profile = {
            "age": None, "gender": None, "condition": None,
            "mu": None, "beta": None, "alpha": None,
            "gamma": None, "memory": None, "delta": None, "epsilon": None
        } if first_session else self.agent.inferred_user_profile

        for t in range(max_turns):
            print(f"\n[Session {session_id} | Turn {t + 1}] ------------------")

            # --- Update Profile if First Session ---
            if first_session:
                # --- Agent Turn ---
                agent_result = self._generate_agent_turn(session_id, first_session)
                agent_utterance = agent_result["utterance"]
                agent_thought = agent_result["monologue"]
                agent_end = agent_result["endkey"]
                inferred_user_profile = agent_result.get("inferred_attributes", {})
            else:
                 # --- Agent Turn ---
                agent_result = self._generate_agent_turn(session_id, first_session)
                agent_utterance = agent_result["utterance"]
                agent_thought = agent_result["monologue"]
                agent_end = agent_result["endkey"]

            self.conversation_history.append({"role": "assistant", "content": agent_utterance})

            # --- User Turn ---
            user_result = self._generate_user_turn(agent_utterance, first_session)
            user_utterance = user_result["utterance"]
            user_monologue = user_result["monologue"]
            user_end = user_result["endkey"]
            action = user_result.get("action", None)

            self.conversation_history.append({"role": "user", "content": user_utterance})


            # --- Log Entry ---
            log_entry = {
                "turn": t + 1,
                "agent_utterance": agent_utterance,
                "agent_monologue": agent_thought,
                "agent_endkey": agent_end,
                "user_utterance": user_utterance,
                "user_monologue": user_monologue,
                "user_endkey": user_end,
            }

            if not first_session:
                log_entry["action"] = action
            else:
                log_entry["inferred_attributes"] = inferred_user_profile

            session_log.append(log_entry)

            if agent_end or user_end:
                break

        # --- After Session ---
        if first_session:
            self.agent.inferred_user_profile = inferred_user_profile
            print("\n[✅ User Profile Inferred]:")
            print(json.dumps(inferred_user_profile, indent=2, ensure_ascii=False))
        else:
            self._analyze_session(session_log, session_id)

        self._save_session_log(session_log, session_id, first_session)

        return session_log
        
    def train(self):
        """
        Main simulation loop:
        - Step 0: First Session (User Profiling)
        - Steps 1~N: RL-driven adaptation and suggestion sessions
        """
        print("\n[Session 0] -------------------------------")
        session_log = self.run_session(session_id=0, first_session=True)

        print("\n[Extracted User Profile]")
        for k, v in self.agent.inferred_user_profile.items():
            print(f"- {k}: {v}")

        # ---- Main Simulation Loop ----
        for session_id in range(1, self.total_steps):
            print(f"\n[Session {session_id}] -------------------------------")

            # Get agent suggestion and index
            suggestion, suggestion_idx, _ = self.agent.policy()

            # Run session with RL-based suggestion
            session_log = self.run_session(session_id=session_id)

            # Extract user action
            action = session_log[-1]["action"] if session_log and "action" in session_log[-1] else None

            # Estimate compliance and reward
            actual_compliance = self.user.compliance_prob(suggestion)
            reward, _ = self.agent.reward(suggestion_idx, action, actual_compliance)

            print(f"Suggestion: {suggestion:.2f}, Action: {action:.2f}, Compliance: {actual_compliance:.4f}, Reward: {reward:.4f}")

            # Log step results
            self._log(suggestion, action, reward, actual_compliance)

        # Save full simulation log
        self.save_log(f"{self.user.user_profile.name}/simulation_log.txt")

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