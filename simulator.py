# simulator.py
import time
import random
import re
import numpy as np
import os
import json
import requests
from requests.exceptions import HTTPError, Timeout, ConnectionError as ReqConnectionError

class Simulator:
    """
    Coordinates the interaction between a psychologically grounded user model
    and a reinforcement learning-based agent. Responsible for warm-up (exploratory)
    initialization, training iterations, and comprehensive logging and visualization.
    """

    def __init__(self, user, agent, action_space, total_steps=400):
        self.user = user
        self.agent = agent
        self.action_space = action_space
        self.total_steps = total_steps
        self.session_steps = 1
        self.RETRY_STATUS = {429, 500, 502, 503, 504}
        self._init_logs()

    def _init_logs(self):
        self.suggestion_trace = []              # RL suggestion (numeric) per session
        self.inferred_action_trace = []         # action inferred from session analysis
        self.ground_truth_action_trace = []     # user's JSON action (not used by agent)
        self.reward_trace = []
        self.compliance_trace = []
        self.estimated_compliance_trace = []
        self.true_mu_trace = []
        self.estimated_mu_trace = []
        self.temperature_trace = []
        self.q_value_trace = []
        self.io_dir = "io_logs"
        os.makedirs(self.io_dir, exist_ok=True)

    # ---------- Helpers ----------

    def _ensure_dir(self, d):
        os.makedirs(d, exist_ok=True)
        return d

    def _save_json(self, path, data):
        self._ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _log_io(self, session_id, turn, role, prompt_text, parsed_json, raw_text=None):
        fname = os.path.join(self.io_dir, f"session_{session_id:03}_turn_{turn:02}_{role}.json")
        payload = {
            "role": role,
            "prompt_text": prompt_text,
            "parsed_response": parsed_json,
        }
        if raw_text is not None:
            payload["raw_text"] = raw_text
        self._save_json(fname, payload)

    def compute_compliance(self, suggestion, inferred_action):
        """
        Distance-based compliance in [0,1]:
        1 - |a - s| / range
        """
        if suggestion is None or inferred_action is None:
            return 0.0
        rng = float(np.ptp(self.action_space)) or 5.0
        comp = 1.0 - abs(float(inferred_action) - float(suggestion)) / rng
        return float(np.clip(comp, 0.0, 1.0))

    def _compliance_summary(self, window: int = 10) -> dict:
        """
        Build a compact compliance summary for prompts.
        mean: 전체 평균, recent_mean: 최근 window개 평균, last: 마지막 값,
        estimated_by_agent: agent가 내부적으로 추정하는 estimated_compliance, count: 유효 샘플 수
        """
        vals = [v for v in self.compliance_trace if v is not None]
        recent = vals[-window:] if vals else []
        to_float = (lambda x: None if x is None else float(x))
        return {
            "count": len(vals),
            "last": to_float(vals[-1]) if vals else None,
            "mean": (float(np.mean(vals)) if vals else None),
            "recent_mean": (float(np.mean(recent)) if recent else None),
            "estimated_by_agent": to_float(getattr(self.agent, "estimated_compliance", None))
        }
    
    # ---------- LLM plumbing ----------

    def generate_response(
        self, role, content, history, model_name, api_url, api_key, headers, return_raw=False,
        retries: int = 5, backoff_base: float = 1.0, timeout: int = 60
    ):
        assert role in ["user", "agent"]

        messages = history + [{"role": "user" if role == "user" else "assistant", "content": content}]
        payload = {
            "model": model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        last_err = None
        result = None

        for attempt in range(retries + 1):
            try:
                response = requests.post(api_url, json=payload, headers=headers, timeout=timeout)

                # 일시적 서버/레이트 리밋 상태코드면 예외로 돌려서 공통 처리
                if response.status_code in self.RETRY_STATUS and attempt < retries:
                    raise HTTPError(f"Transient HTTP {response.status_code}", response=response)

                response.raise_for_status()
                result = response.json()
                break  # 성공 시 루프 종료

            except (Timeout, ReqConnectionError) as e:
                last_err = e
                if attempt < retries:
                    wait = backoff_base * (2 ** attempt) + random.uniform(0, 0.5)
                    # 필요하면 로깅/프린트
                    print(f"[Retry {attempt+1}/{retries}] Network issue ({e.__class__.__name__}). Sleeping {wait:.2f}s...")
                    time.sleep(wait)
                    continue
                else:
                    raise

            except HTTPError as e:
                last_err = e
                status = getattr(e.response, "status_code", None)
                if status in self.RETRY_STATUS and attempt < retries:
                    wait = backoff_base * (2 ** attempt) + random.uniform(0, 0.5)
                    print(f"[Retry {attempt+1}/{retries}] HTTP {status}. Sleeping {wait:.2f}s...")
                    time.sleep(wait)
                    continue
                else:
                    # 재시도 불가 상태코드 또는 마지막 시도
                    raise

        if result is None:
            # 이 경우는 루프를 다 돌아도 성공 못한 케이스
            raise RuntimeError(f"Failed to get response after {retries} retries") from last_err

        message_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"[{role.capitalize()} Response]: {message_content}")

        # --- 안전한 JSON 파싱 (```json ... ``` 또는 ``` ... ``` 커버) ---
        cleaned = message_content.strip()
        # 코드펜스 제거
        if cleaned.startswith("```"):
            # ```json ... ``` 또는 ``` ... ```
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        # 1차 시도
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # 본문에서 가장 그럴듯한 JSON 오브젝트/배열만 추출해서 재시도
            m = re.search(r"(\{.*\}|\[.*\])", message_content, flags=re.DOTALL)
            if not m:
                raise
            parsed = json.loads(m.group(1))

        if return_raw:
            return {"parsed": parsed, "raw": message_content, "messages": messages}
        return parsed

    # ---------- Turns ----------

    def _generate_agent_turn(self, session_id, first_session, planned_suggestion=None):
        if first_session:
            prompt = self.agent.format_agent_1st_session_prompt()
            suggestion_score = None
            suggestion_history = []
            prior_analysis = {}
        else:
            prior_analysis_path = f"sessions/analysis_{session_id - 1:03}.json"
            prior_analysis = self._load_json(prior_analysis_path) or {}

            # 최근 추천 이력(숫자)만 간단히 전달
            suggestion_history = self.suggestion_trace[-10:]
            # 제안 점수(화면 표기용)는 planned_suggestion로 사용
            suggestion_score = planned_suggestion if planned_suggestion is not None else self.agent.estimated_behavior_mean

            prompt = self.agent.format_agent_prompt(
                suggestion_score=suggestion_score,
                suggestion_history=suggestion_history,
                prior_analysis=prior_analysis,
                planned_suggestion=planned_suggestion if planned_suggestion is not None else self.agent.goal_behavior
            )

        ret = self.generate_response(
            role="agent",
            content=prompt,
            history=self.conversation_history,
            model_name=self.agent.model_name,
            api_url=self.agent.api_url,
            api_key=self.agent.api_key,
            headers=self.agent.headers,
            return_raw=True
        )
        response = ret["parsed"]
        self._log_io(session_id, len(self.conversation_history)//2 + 1, "agent", prompt, response, raw_text=ret["raw"])

        return {
            "utterance": response.get("utterance", ""),
            "monologue": response.get("monologue", ""),
            "endkey": response.get("endkey", False)
        }

    def _generate_user_turn(self, session_id):
        prompt = self.user.format_user_prompt(
            recommendation_history=self.suggestion_trace,
            action_history=self.ground_truth_action_trace
        )

        ret = self.generate_response(
            role="user",
            content=prompt,
            history=self.conversation_history,
            model_name=self.user.model_name,
            api_url=self.user.api_url,
            api_key=self.user.api_key,
            headers=self.user.headers,
            return_raw=True
        )
        response = ret["parsed"]
        self._log_io(session_id, len(self.conversation_history)//2 + 1, "user", prompt, response, raw_text=ret["raw"])

        return {
            "utterance": response.get("utterance", ""),
            "monologue": response.get("monologue", ""),
            "endkey": response.get("endkey", False),
            "action": float(response["action"]) if response.get("endkey") and "action" in response else None
        }

    # ---------- Session analysis ----------

    def _analyze_session(self, conversation_history, session_id, last_suggestion):
        prompt = self.agent.format_agent_session_analysis_prompt(last_suggestion=last_suggestion)
        ret = self.generate_response(
            role="agent",
            content=prompt,
            history=conversation_history,  # 중요: 챗 메시지 형식의 history 사용
            model_name=self.agent.model_name,
            api_url=self.agent.api_url,
            api_key=self.agent.api_key,
            headers=self.agent.headers,
            return_raw=True
        )
        analysis = ret["parsed"]
        print("\n[🧠 Session Analysis]:")
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
        self._save_json(f"sessions/analysis_{session_id:03}.json", analysis)
        self._log_io(session_id, 0, "analysis", prompt, analysis, raw_text=ret["raw"])
        return analysis

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

    def infer_user_profile_from_session(self, conversation_history) -> dict:
        # (기존 코드 유지) 프로필 1회 분석
        prompt = self.agent.format_agent_1st_session_analysis_prompt()
        result = self.generate_response(
            role="agent",
            content=prompt,
            history=conversation_history,
            model_name=self.agent.model_name,
            api_url=self.agent.api_url,
            api_key=self.agent.api_key,
            headers={"Authorization": f"Bearer {self.agent.api_key}"}
        )
        self.agent.inferred_user_profile = result.get("inferred_attributes", {})
        self.agent.goal_behavior = result.get("goal_behavior", 4.0)

    def run_session(self, session_id: int, max_turns: int = 10, first_session: bool = False, planned_suggestion: float = None):
        self.conversation_history = []
        session_log = []

        for t in range(max_turns):
            print(f"\n[Session {session_id} | Turn {t + 1}] ------------------")

            self.agent.run_context = {
                "session_id": session_id,
                "current_turn": t + 1,
                "max_turns": max_turns,
                "total_sessions": self.total_steps,
                "compliance_summary": self._compliance_summary(window=10)
            }

            # --- Agent Turn ---
            agent_result = self._generate_agent_turn(session_id, first_session, planned_suggestion)
            agent_utterance = agent_result["utterance"]
            agent_monologue = agent_result["monologue"]
            agent_end = agent_result["endkey"]

            self.conversation_history.append({"role": "assistant", "content": agent_utterance})

            # --- User Turn ---
            user_result = self._generate_user_turn(session_id)
            user_utterance = user_result["utterance"]
            user_end = user_result["endkey"]
            ground_truth_action = user_result.get("action", None)

            self.conversation_history.append({"role": "user", "content": user_utterance})

            # --- Log Entry ---
            log_entry = {
                "turn": t + 1,
                "agent_utterance": agent_utterance,
                "agent_monologue": agent_monologue,
                "agent_endkey": agent_end,
                "user_utterance": user_utterance,
                "user_endkey": user_end,
                "planned_suggestion": planned_suggestion
            }

            # 실제 action은 세션 파일에 기록(평가/시각화용)하되, 에이전트 업데이트에는 사용하지 않음
            if not first_session:
                log_entry["ground_truth_action"] = ground_truth_action

            session_log.append(log_entry)

            if agent_end or user_end:
                break

        # --- After Session ---
        if first_session:
            self.infer_user_profile_from_session(self.conversation_history)
            inferred_action = None
            compliance_est = None
        else:
            analysis = self._analyze_session(self.conversation_history, session_id, last_suggestion=planned_suggestion)
            inferred_action = analysis.get("user_action_estimate", None)
            try:
                inferred_action = float(inferred_action) if inferred_action is not None else None
            except Exception:
                inferred_action = None

            compliance_est = analysis.get("compliance_estimate", None)
            try:
                compliance_est = float(compliance_est) if compliance_est is not None else None
            except Exception:
                compliance_est = None

            # 안전망: 분석 JSON에 compliance_estimate가 없거나 이상하면 재계산
            if compliance_est is None:
                compliance_est = self.compute_compliance(planned_suggestion, inferred_action)

            # 세션 로그에 추론 결과 저장
            if len(session_log) > 0:
                session_log[-1]["inferred_action"] = inferred_action
                session_log[-1]["compliance_estimate"] = compliance_est

        self._save_session_log(session_log, session_id, first_session)
        return session_log

    # ---------- Training ----------

    def train(self):
        """
        Main simulation loop:
        - Step 0: First Session (User Profiling)
        - Steps 1~N: RL-driven suggestion & feedback (action inferred via session analysis)
        """
        print("\n[Session 0] -------------------------------")
        session_log = self.run_session(session_id=0, first_session=True)

        print("\n[Goal Behavior Inference]")
        print(self.agent.goal_behavior)

        print("\n[Extracted User Profile]")
        for k, v in self.agent.inferred_user_profile.items():
            print(f"- {k}: {v}")

        # ---- Main Simulation Loop ----
        for session_id in range(1, self.total_steps):
            print(f"\n[Session {session_id}] -------------------------------")

            # 1) RL 제안(숫자) 선택
            suggestion, suggestion_idx, _ = self.agent.policy()
            self.suggestion_trace.append(suggestion)

            # 2) 세션 실행 (사용자는 실제 action을 내보내지만, 에이전트는 그 값에 접근하지 않음)
            session_log = self.run_session(session_id=session_id, planned_suggestion=suggestion)

            # 3) 파일로 저장된 분석 결과 로드
            analysis = self._load_json(f"sessions/analysis_{session_id:03}.json") or {}
            inferred_action = analysis.get("user_action_estimate", None)
            try:
                inferred_action = float(inferred_action) if inferred_action is not None else None
            except Exception:
                inferred_action = None

            compliance = analysis.get("compliance_estimate", None)
            try:
                compliance = float(compliance) if compliance is not None else None
            except Exception:
                compliance = None

            if compliance is None:
                compliance = self.compute_compliance(suggestion, inferred_action)

            # 4) 실제 action(평가용) 추출
            ground_truth_action = None
            if session_log and "ground_truth_action" in session_log[-1]:
                ground_truth_action = session_log[-1]["ground_truth_action"]

            # 5) 보상 계산 및 Q 업데이트 (추론 행동 사용)
            reward, _ = self.agent.reward(suggestion_idx, inferred_action if inferred_action is not None else suggestion, compliance)

            print(f"Suggestion: {suggestion:.2f}, "
                  f"Inferred Action: {('NA' if inferred_action is None else f'{inferred_action:.2f}')}, "
                  f"GT Action: {('NA' if ground_truth_action is None else f'{ground_truth_action:.2f}')}, "
                  f"Compliance(est): {compliance:.4f}, Reward: {reward:.4f}")

            # 6) 로그 적재
            self._log_after_session(suggestion, inferred_action, ground_truth_action, reward, compliance)

        # Save simulation log
        self.save_log(f"{'simulation_log.txt'}")

    def _log_after_session(self, suggestion, inferred_action, ground_truth_action, reward, compliance):
        self.inferred_action_trace.append(inferred_action)
        self.ground_truth_action_trace.append(ground_truth_action)
        self.reward_trace.append(reward)
        self.compliance_trace.append(compliance)
        self.estimated_compliance_trace.append(self.agent.estimated_compliance)
        self.true_mu_trace.append(getattr(self.user, "behavior_mean", np.nan))
        self.estimated_mu_trace.append(self.agent.estimated_behavior_mean)
        self.temperature_trace.append(self.agent.policy_temperature)
        self.q_value_trace.append(self.agent.q_values.copy())

    def save_log(self, filename="simulation_log.txt"):
        with open(filename, "w") as f:
            f.write("Step\tSuggestion\tInferredAction\tGTAction\tCompliance\tReward\tEstimated_mu\tTemperature\tQ_max\n")

            for i in range(len(self.suggestion_trace)):
                def fmt(val):
                    return f"{val:.4f}" if (val is not None and not (isinstance(val, float) and np.isnan(val))) else "NA"

                q_val_max = np.max(self.q_value_trace[i]) if i < len(self.q_value_trace) else None

                f.write(
                    f"{i+1}\t"
                    f"{fmt(self.suggestion_trace[i])}\t"
                    f"{fmt(self.inferred_action_trace[i] if i < len(self.inferred_action_trace) else None)}\t"
                    f"{fmt(self.ground_truth_action_trace[i] if i < len(self.ground_truth_action_trace) else None)}\t"
                    f"{fmt(self.compliance_trace[i] if i < len(self.compliance_trace) else None)}\t"
                    f"{fmt(self.reward_trace[i] if i < len(self.reward_trace) else None)}\t"
                    f"{fmt(self.estimated_mu_trace[i] if i < len(self.estimated_mu_trace) else None)}\t"
                    f"{fmt(self.temperature_trace[i] if i < len(self.temperature_trace) else None)}\t"
                    f"{fmt(q_val_max)}\n"
                )
