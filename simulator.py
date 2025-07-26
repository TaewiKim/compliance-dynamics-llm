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
        Number of time steps for simulation (includes warmup).
    warmup_steps : int
        Initial steps where random actions are used to initialize user behavior.
    """

    def __init__(self, user, agent, action_space, total_steps=400):
        self.user = user
        self.agent = agent
        self.action_space = action_space
        self.total_steps = total_steps

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
            "temperature": 0.7,
        }

        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        message_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        # 모든 역할에서 JSON parsing 시도
        try:
            parsed = json.loads(message_content)
        except json.JSONDecodeError:
            parsed = {
                "utterance": message_content.strip(),
                "monologue": "",
                "endkey": False,
            }

        # user 응답 시 action 필수 검증
        if role == "user" and parsed.get("endkey") and "action" not in parsed:
            parsed["action"] = None

        return parsed

    def run_profile_session(self, session_id: int = 0, max_turns: int = 10):
        self.conversation_history = []
        session_log = []
        user_profile = {
            "age": None, "gender": None, "condition": None,
            "mu": None, "beta": None, "alpha": None,
            "gamma": None, "memory": None, "delta": None, "epsilon": None
        }

        for t in range(max_turns):
            print(f"\n[Profile Session {session_id} | Turn {t + 1}] ------------------")

            prompt = self.agent.format_1st_session_prompt()
            agent_response = self.generate_response(
                role="agent",
                content=prompt,
                history=self.conversation_history,
                model_name=self.user.model_name,
                api_url=self.user.api_url,
                api_key=self.user.api_key,
                headers=self.user.headers,
            )

            agent_text = agent_response.get("utterance", "")
            agent_monologue = agent_response.get("monologue", "")
            agent_end = agent_response.get("endkey", False)
            inferred = agent_response.get("inferred_attributes", {})

            self.conversation_history.append({"role": "assistant", "content": agent_text})

            # --- User 응답 ---
            user_prompt = self.user.format_user_prompt(user_profile) + f"\n\n{agent_text}"
            user_response = self.generate_response(
                role="user",
                content=user_prompt,
                history=self.conversation_history,
                model_name=self.user.model_name,
                api_url=self.user.api_url,
                api_key=self.user.api_key,
                headers=self.user.headers,
            )

            user_text = user_response.get("utterance", "")
            user_monologue = user_response.get("monologue", "")
            user_end = user_response.get("endkey", False)

            self.conversation_history.append({"role": "user", "content": user_text})

            # --- 추론된 프로파일 속성 저장 ---
            for k, v in inferred.items():
                if v and not user_profile.get(k):
                    user_profile[k] = v

            # --- 로그 저장 ---
            session_log.append({
                "turn": t + 1,
                "agent_text": agent_text,
                "agent_monologue": agent_monologue,
                "user_utterance": user_text,
                "user_monologue": user_monologue,
                "inferred": inferred,
            })

            if agent_end or user_end:
                break

        # 저장 및 반환
        self.user.profile = user_profile
        os.makedirs("sessions", exist_ok=True)
        session_path = f"sessions/profile_session_{session_id:03}.json"
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(session_log, f, ensure_ascii=False, indent=2)

        print("\n[✅ User Profile Inferred]:")
        print(json.dumps(user_profile, indent=2, ensure_ascii=False))

        return user_profile

    def run_session(self, session_id: int, max_turns: int = 10, first_session: bool = False):
        self.conversation_history = []
        session_log = []

        for t in range(max_turns):
            print(f"\n[Session {session_id} | Turn {t + 1}] ------------------")

            if first_session:
                # 첫 세션용 프롬프트 (agent는 정보 수집 목적)
                agent_prompt = self.agent.format_1st_session_prompt()

                agent_response = self.generate_response(
                    role="agent",
                    content=agent_prompt,
                    history=self.conversation_history,
                    model_name=self.user.model_name,
                    api_url=self.user.api_url,
                    api_key=self.user.api_key,
                    headers=self.user.headers,
                )

                agent_text = agent_response.get("utterance", "")
                agent_thought = agent_response.get("monologue", "")
                agent_end = agent_response.get("endkey", False)
                inferred_attributes = agent_response.get("inferred_attributes", {})

                self.conversation_history.append({"role": "assistant", "content": agent_text})

                # 사용자 응답 생성
                user_prompt = agent_text
                user_response = self.generate_response(
                    role="user",
                    content=user_prompt,
                    history=self.conversation_history,
                    model_name=self.user.model_name,
                    api_url=self.user.api_url,
                    api_key=self.user.api_key,
                    headers=self.user.headers,
                )

                user_utterance = user_response.get("utterance", "")
                user_monologue = user_response.get("monologue", "")
                user_end = user_response.get("endkey", False)

                self.conversation_history.append({"role": "user", "content": user_utterance})

                print(f"[Agent]: {agent_text}")
                print(f"[Agent Thought]: {agent_thought}")
                print(f"[User]: {user_utterance}")
                print(f"[User Thought]: {user_monologue}")

                session_log.append({
                    "turn": t + 1,
                    "agent_text": agent_text,
                    "agent_monologue": agent_thought,
                    "agent_endkey": agent_end,
                    "user_utterance": user_utterance,
                    "user_monologue": user_monologue,
                    "user_endkey": user_end,
                    "inferred_attributes": inferred_attributes
                })

                if user_end or agent_end:
                    break

            else:
                # ---- Regular session: 행동 제안 및 응답 ----
                suggestion, suggestion_idx, suggestion_score = self.agent.policy()
                policy_input = f"Today's suggestion score is: {suggestion_score:.2f}"

                agent_response = self.generate_response(
                    role="agent",
                    content=policy_input,
                    history=self.conversation_history,
                    model_name=self.user.model_name,
                    api_url=self.user.api_url,
                    api_key=self.user.api_key,
                    headers=self.user.headers,
                )

                agent_text = agent_response.get("utterance", "")
                agent_thought = agent_response.get("monologue", "")
                agent_end = agent_response.get("endkey", False)

                self.conversation_history.append({"role": "assistant", "content": agent_text})

                # 사용자 응답 생성
                user_prompt = self.user.format_prompt(self.user.profile) + f"\n\n{agent_text}"
                user_response = self.generate_response(
                    role="user",
                    content=user_prompt,
                    history=self.conversation_history,
                    model_name=self.user.model_name,
                    api_url=self.user.api_url,
                    api_key=self.user.api_key,
                    headers=self.user.headers,
                )

                user_utterance = user_response.get("utterance", "")
                user_monologue = user_response.get("monologue", "")
                user_end = user_response.get("endkey", False)
                action = float(user_response["action"]) if user_end else None

                self.conversation_history.append({"role": "user", "content": user_utterance})

                print(f"[Agent]: {agent_text}")
                print(f"[Agent Thought]: {agent_thought}")
                print(f"[User]: {user_utterance}")
                print(f"[User Thought]: {user_monologue}")
                print(f"[Action]: {action} (End: {user_end})")

                session_log.append({
                    "turn": t + 1,
                    "agent_text": agent_text,
                    "agent_monologue": agent_thought,
                    "agent_endkey": agent_end,
                    "user_utterance": user_utterance,
                    "user_monologue": user_monologue,
                    "user_endkey": user_end,
                    "action": action,
                    "suggestion_score": suggestion_score
                })

                if user_end or agent_end:
                    break

        # Save session
        os.makedirs("sessions", exist_ok=True)
        session_path = f"sessions/session_{session_id:03}.json"
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(session_log, f, ensure_ascii=False, indent=2)

        return session_log


    def train(self):
        """
        Main simulation loop with initial user profiling session (session 0),
        followed by regular RL-based coaching sessions.
        """
        # ---- Session 0: First Session (Profiling) ----
        print("\n[Session 0] -------------------------------")
        session_log_0 = self.run_session(session_id=0, first_session=True)

        # Extract last agent response with inferred attributes
        last_turn = session_log_0[-1]
        inferred_attributes = last_turn.get("inferred_attributes", {})

        # Save user profile
        self.user.profile = inferred_attributes
        print("\n[Extracted User Profile]")
        for k, v in inferred_attributes.items():
            print(f"- {k}: {v}")

        # ---- Sessions 1 ~ N: Main RL Simulation ----
        for t in range(self.warmup_steps, self.total_steps):
            print(f"\n[Session {t - self.warmup_steps + 1}] -------------------------------")

            suggestion, suggestion_idx, _ = self.agent.policy()
            session_id = t - self.warmup_steps + 1

            # Run session with updated user profile
            session_log = self.run_session(session_id)

            # Extract action from user
            action = session_log[-1]["action"] if session_log and "action" in session_log[-1] else None

            # Compliance probability based on user model
            actual_compliance = self.user.compliance_prob(suggestion)

            # Compute reward and update agent
            reward, _ = self.agent.reward(suggestion_idx, action, actual_compliance)

            print(f"Suggestion: {suggestion:.2f}, Action: {action:.2f}, Compliance: {actual_compliance:.4f}, Reward: {reward:.4f}")

            self._log(suggestion, action, reward, actual_compliance)

        # Save logs after all steps
        self.save_log("simulation_log.txt")

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

    def plot(self, save=False, filename=None):
        """
        Generate 5-panel visualization:
        - Agent suggestion vs user behavior
        - Reward over time
        - Compliance tracking
        - Policy temperature
        - Estimated vs true behavior mean
        """

        fig, axs = plt.subplots(6, 1, figsize=(12, 16))
        alpha = self.user.adaptation_rate
        beta = self.user.compliance_sensitivity
        gamma = self.user.noise_sensitivity
        user_info = r"$(\alpha=" + f"{alpha},\\beta={beta},\\gamma={gamma})$"

        axs[0].plot(self.suggestion_trace, label='Agent Suggestion', color='orange')
        axs[0].scatter(range(len(self.action_trace)), self.action_trace, label='User Action', color='blue', s=10)
        axs[0].axhline(self.user.behavior_mean, linestyle='--', color='gray', label=r'Final True $\mu$')
        axs[0].axhline(self.user.initial_behavior_mean, linestyle=':', color='purple', label=r'Initial $\mu_0$')
        axs[0].axhline(self.agent.goal, linestyle='--', color='red', label=r'Agent Goal $G$')
        axs[0].axvline(self.warmup_steps, linestyle=':', color='black', label='Warm-up End')
        axs[0].set_ylabel("Behavior Value")
        axs[0].set_title(rf"Agent Suggestion vs Adaptive User Behavior {user_info}")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(self.reward_trace, label='Reward', color='blue')
        axs[1].axvline(self.warmup_steps, linestyle=':', color='black')
        axs[1].set_ylabel("Reward")
        axs[1].set_title("Reward Dynamics Over Time")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(self.compliance_trace, label='Actual Compliance', color='darkgreen')
        axs[2].plot(self.estimated_compliance_trace, label='Estimated Compliance', color='magenta', linestyle='--')
        axs[2].axvline(self.warmup_steps, linestyle=':', color='black')
        axs[2].set_ylabel("Compliance")
        axs[2].set_title("Compliance Estimation Over Time")
        axs[2].legend()
        axs[2].grid(True)

        axs[3].plot(self.temperature_trace, label='Policy Temperature', color='brown')
        axs[3].set_ylabel("Temperature")
        axs[3].set_title("Softmax Temperature Evolution")
        axs[3].legend()
        axs[3].grid(True)

        axs[4].plot(self.estimated_mu_trace, label=r'Estimated $\hat{\mu}$', color='green')
        axs[4].plot(self.true_mu_trace, linestyle='--', color='gray', label=r'True $\mu$')
        axs[4].axvline(self.warmup_steps, linestyle=':', color='black', label='Warm-up End')
        axs[4].set_ylabel("Behavior Mean")
        axs[4].set_title(r"Estimated vs Actual Behavior Mean ($\hat{\mu}$ vs $\mu$)")
        axs[4].legend()
        axs[4].grid(True)

        # --- 6. Max Q-value over time ---
        q_values_over_time = np.array(self.q_value_trace)  # shape: (timesteps, num_actions)

        # 각 timestep에서 Q-value 중 최댓값 추출
        max_q_values = np.max(q_values_over_time, axis=1)

        axs[5].plot(max_q_values, label='Max Q-value', color='navy')
        axs[5].set_ylabel("Max Q-value")
        axs[5].set_title("Maximum Q-value Over Time")
        axs[5].axvline(self.warmup_steps, linestyle=':', color='black', label='Warm-up End')
        axs[5].legend()
        axs[5].grid(True)

        plt.xlabel("Time Step")
        plt.tight_layout()

        if save and filename:
            # Save PNG
            png_dir = "plots"
            os.makedirs(png_dir, exist_ok=True)
            png_path = os.path.join(png_dir, f"{filename}.png")
            plt.savefig(png_path)
            return png_path
        
        plt.show()
