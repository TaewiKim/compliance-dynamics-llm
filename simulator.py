# simulator.py (with enhanced logging)
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
        self.RETRY_STATUS = {429, 500, 502, 503, 504}
        self._init_logs()

    def _init_logs(self):
        self.suggestion_trace = []
        self.inferred_action_trace = []
        self.ground_truth_action_trace = []
        self.reward_trace = []
        self.compliance_trace = []
        self.estimated_compliance_trace = []
        self.estimated_mu_trace = []
        self.temperature_trace = []
        self.q_value_trace = []
        self.io_dir = "io_logs"
        os.makedirs(self.io_dir, exist_ok=True)

    # ... [helper methods like _ensure_dir, _save_json, etc. remain the same] ...

    def _ensure_dir(self, d):
        os.makedirs(d, exist_ok=True)
        return d

    def _save_json(self, path, data):
        self._ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _log_io(self, session_id, turn, role, prompt_text, parsed_json, raw_text=None):
        fname = os.path.join(self.io_dir, f"session_{session_id:03}_turn_{turn:02}_{role}.json")
        payload = {"role": role, "prompt_text": prompt_text, "parsed_response": parsed_json}
        if raw_text is not None:
            payload["raw_text"] = raw_text
        self._save_json(fname, payload)

    def compute_compliance(self, suggestion, inferred_action):
        if suggestion is None or inferred_action is None:
            return 0.0
        rng = float(np.ptp(self.action_space)) or 5.0
        comp = 1.0 - abs(float(inferred_action) - float(suggestion)) / rng
        return float(np.clip(comp, 0.0, 1.0))

    def _compliance_summary(self, window: int = 10) -> dict:
        vals = [v for v in self.compliance_trace if v is not None]
        recent = vals[-window:] if vals else []
        to_float = (lambda x: None if x is None else float(x))
        return {
            "count": len(vals), "last": to_float(vals[-1]) if vals else None,
            "mean": (float(np.mean(vals)) if vals else None),
            "recent_mean": (float(np.mean(recent)) if recent else None),
            "estimated_by_agent": to_float(getattr(self.agent, "estimated_compliance", None))
        }
        
    def generate_response(self, role, content, history, model_name, api_url, api_key, headers, return_raw=False, retries: int = 5, backoff_base: float = 1.0, timeout: int = 60):
        assert role in ["user", "agent"]
        messages = history + [{"role": "user" if role == "user" else "assistant", "content": content}]
        payload = {"model": model_name, "messages": messages, "response_format": {"type": "json_object"}}
        last_err = None
        for attempt in range(retries + 1):
            try:
                response = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
                if response.status_code in self.RETRY_STATUS and attempt < retries:
                    raise HTTPError(f"Transient HTTP {response.status_code}", response=response)
                response.raise_for_status()
                result = response.json()
                message_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # --- DETAILED LOGGING ---
                print(f"\n<< RAW LLM Response from {role.upper()} >>\n{message_content}\n--------------------")
                
                cleaned = message_content.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                    cleaned = re.sub(r"\s*```$", "", cleaned)
                try:
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError:
                    m = re.search(r"(\{.*\}|\[.*\])", message_content, flags=re.DOTALL)
                    if not m: raise
                    parsed = json.loads(m.group(1))
                if return_raw:
                    return {"parsed": parsed, "raw": message_content, "messages": messages}
                return parsed
            except (Timeout, ReqConnectionError, HTTPError) as e:
                last_err = e
                if attempt < retries:
                    wait = backoff_base * (2 ** attempt) + random.uniform(0, 0.5)
                    print(f"[Retry {attempt+1}/{retries}] Network issue. Sleeping {wait:.2f}s...")
                    time.sleep(wait)
                else: raise
        raise RuntimeError(f"Failed to get response after {retries} retries") from last_err

    # ... [other methods like _generate_agent_turn, _generate_user_utterance] ...
    def _generate_agent_turn(self, session_id, first_session, planned_suggestion=None):
        if first_session:
            prompt = self.agent.format_agent_1st_session_prompt()
        else:
            prior_analysis = self._load_json(f"sessions/analysis_{session_id - 1:03}.json") or {}
            prompt = self.agent.format_agent_prompt(
                suggestion_score=planned_suggestion,
                suggestion_history=self.suggestion_trace[-10:],
                prior_analysis=prior_analysis,
                planned_suggestion=planned_suggestion if planned_suggestion is not None else self.agent.goal_behavior
            )
        # --- DETAILED LOGGING ---
        print(f"\n>> AGENT PROMPT for Session {session_id} Turn {len(self.conversation_history)//2 + 1} <<\n{prompt}\n--------------------")
        ret = self.generate_response("agent", prompt, self.conversation_history, self.agent.model_name, self.agent.api_url, self.agent.api_key, self.agent.headers, return_raw=True)
        response = ret["parsed"]
        self._log_io(session_id, len(self.conversation_history)//2 + 1, "agent", prompt, response, ret["raw"])
        return response

    def _generate_user_utterance(self, session_id):
        prompt = self.user.format_user_prompt(
            recommendation_history=self.suggestion_trace,
            action_history=self.ground_truth_action_trace
        )
        # --- DETAILED LOGGING ---
        print(f"\n>> USER PROMPT for Session {session_id} Turn {len(self.conversation_history)//2} <<\n{prompt}\n--------------------")
        ret = self.generate_response("user", prompt, self.conversation_history, self.user.model_name, self.user.api_url, self.user.api_key, self.user.headers, return_raw=True)
        response = ret["parsed"]
        self._log_io(session_id, len(self.conversation_history)//2, "user_utterance", prompt, response, ret["raw"])
        return response

    def _determine_user_action(self, session_id):
        print("\n[Determining User Action]...")
        prompt = self.user.format_user_action_prompt(
            conversation_history=self.conversation_history,
            recommendation_history=self.suggestion_trace,
            action_history=self.ground_truth_action_trace
        )
        # --- DETAILED LOGGING ---
        print(f"\n>> USER ACTION PROMPT for Session {session_id} <<\n{prompt}\n--------------------")
        ret = self.generate_response("user", prompt, [], self.user.model_name, self.user.api_url, self.user.api_key, self.user.headers, return_raw=True)
        response = ret["parsed"]
        self._log_io(session_id, 99, "user_action", prompt, response, ret["raw"])
        return response.get("action")

    # ... [other methods like _analyze_session, run_session] ...
    def _analyze_session(self, session_id, last_suggestion):
        prompt = self.agent.format_agent_session_analysis_prompt(last_suggestion=last_suggestion)
        # --- DETAILED LOGGING ---
        print(f"\n>> ANALYSIS PROMPT for Session {session_id} <<\n{prompt}\n--------------------")
        ret = self.generate_response("agent", prompt, self.conversation_history, self.agent.model_name, self.agent.api_url, self.agent.api_key, self.agent.headers, return_raw=True)
        analysis = ret["parsed"]
        print("\n[🧠 Session Analysis]:")
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
        self._save_json(f"sessions/analysis_{session_id:03}.json", analysis)
        self._log_io(session_id, 0, "analysis", prompt, analysis, ret["raw"])
        return analysis
    
    def _load_json(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_session_log(self, session_log, session_id, first_session):
        os.makedirs("sessions", exist_ok=True)
        path = f"sessions/{'profile' if first_session else 'session'}_{session_id:03}.json"
        self._save_json(path, session_log)

    def run_session(self, session_id: int, max_turns: int = 10, first_session: bool = False, planned_suggestion: float = None):
        self.conversation_history = []
        session_log = []

        for t in range(max_turns):
            print(f"\n[Session {session_id} | Turn {t + 1}] ------------------")
            self.agent.run_context = {
                "session_id": session_id, "current_turn": t + 1, "max_turns": max_turns,
                "total_sessions": self.total_steps, "compliance_summary": self._compliance_summary()
            }
            agent_result = self._generate_agent_turn(session_id, first_session, planned_suggestion)
            self.conversation_history.append({"role": "assistant", "content": agent_result.get("utterance", "")})
            user_result = self._generate_user_utterance(session_id)
            self.conversation_history.append({"role": "user", "content": user_result.get("utterance", "")})
            
            session_log.append({
                "turn": t + 1,
                "agent_utterance": agent_result.get("utterance", ""),
                "agent_monologue": agent_result.get("monologue", ""),
                "user_utterance": user_result.get("utterance", ""),
            })

            if user_result.get("endkey", False) or agent_result.get("endkey", False):
                break
        
        ground_truth_action = self._determine_user_action(session_id)
        if ground_truth_action is not None:
            ground_truth_action = float(ground_truth_action)
            if session_log:
                session_log[-1]["ground_truth_action"] = ground_truth_action

        if first_session:
            prompt = self.agent.format_agent_1st_session_analysis_prompt()
            analysis = self.generate_response("agent", prompt, self.conversation_history, self.agent.model_name, self.agent.api_url, self.agent.api_key, self.agent.headers)
            self.agent.inferred_user_profile = analysis.get("inferred_attributes", {})
            self.agent.goal_behavior = analysis.get("goal_behavior", 4.0)
        else:
            analysis = self._analyze_session(session_id, last_suggestion=planned_suggestion)
            inferred_action = analysis.get("user_action_estimate")
            compliance_est = analysis.get("compliance_estimate")
            try: inferred_action = float(inferred_action) if inferred_action is not None else None
            except (ValueError, TypeError): inferred_action = None
            try: compliance_est = float(compliance_est) if compliance_est is not None else None
            except (ValueError, TypeError): compliance_est = None

            if compliance_est is None and inferred_action is not None:
                compliance_est = self.compute_compliance(planned_suggestion, inferred_action)
            
            if session_log:
                session_log[-1]["inferred_action"] = inferred_action
                session_log[-1]["compliance_estimate"] = compliance_est

        self._save_session_log(session_log, session_id, first_session)
        return session_log
        
    def train(self):
        print("\n[Session 0] -------------------------------")
        self.run_session(session_id=0, first_session=True)
        print("\n[Goal Behavior Inference]:", self.agent.goal_behavior)
        print("\n[Extracted User Profile]:", json.dumps(self.agent.inferred_user_profile, indent=2))

        for session_id in range(1, self.total_steps):
            print(f"\n[Session {session_id}] -------------------------------")
            suggestion, suggestion_idx, _ = self.agent.policy()
            self.suggestion_trace.append(suggestion)

            session_log = self.run_session(session_id=session_id, planned_suggestion=suggestion)
            
            analysis = self._load_json(f"sessions/analysis_{session_id:03}.json") or {}
            inferred_action = analysis.get("user_action_estimate")
            compliance = analysis.get("compliance_estimate")
            ground_truth_action = session_log[-1].get("ground_truth_action") if session_log else None
            
            try: inferred_action = float(inferred_action) if inferred_action is not None else None
            except (ValueError, TypeError): inferred_action = None
            try: compliance = float(compliance) if compliance is not None else None
            except (ValueError, TypeError): compliance = None

            if compliance is None and inferred_action is not None:
                compliance = self.compute_compliance(suggestion, inferred_action)

            reward, _ = self.agent.reward(suggestion_idx, inferred_action if inferred_action is not None else suggestion, compliance if compliance is not None else 0.0)

            # --- DETAILED LOGGING ---
            print("\n" + "="*50)
            print(f"SESSION {session_id} SUMMARY")
            print(f"  - Suggestion:           {suggestion:.2f}")
            print(f"  - Ground Truth Action:  {ground_truth_action:.2f}" if ground_truth_action is not None else "  - Ground Truth Action:  N/A")
            print(f"  - Inferred Action:      {inferred_action:.2f}" if inferred_action is not None else "  - Inferred Action:      N/A")
            print(f"  - Compliance:           {compliance:.4f}" if compliance is not None else "  - Compliance:           N/A")
            print(f"  - Reward:               {reward:.4f}")
            print(f"  - Agent Estimated Mean: {self.agent.estimated_behavior_mean:.2f}")
            print(f"  - Agent Temperature:    {self.agent.policy_temperature:.2f}")
            print("="*50 + "\n")


            self._log_after_session(suggestion, inferred_action, ground_truth_action, reward, compliance)

        self.save_log()
    
    def _log_after_session(self, suggestion, inferred_action, ground_truth_action, reward, compliance):
        self.inferred_action_trace.append(inferred_action)
        self.ground_truth_action_trace.append(ground_truth_action)
        self.reward_trace.append(reward)
        self.compliance_trace.append(compliance)
        self.estimated_compliance_trace.append(self.agent.estimated_compliance)
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