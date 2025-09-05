import os
import re
import json
import random
import types
import time
import numpy as np
import streamlit as st

from agent import Agent
from user import UserLlm
from simulator import Simulator


# -------------------------------
# Helpers
# -------------------------------

def parse_last_suggestion_from_agent_prompt(prompt: str, default=3.0) -> float:
    """
    Extract planned numeric suggestion from agent planning prompt.
    Looks for lines like: 'Planned Numeric Suggestion ...: 3.25'
    """
    m = re.search(r"Planned Numeric Suggestion.*?:\s*([0-9.]+)", prompt)
    if m:
        try:
            return float(m.group(1))
        except:
            return default
    return default

def parse_recent_recs_from_user_prompt(prompt: str) -> list:
    """
    In user prompt, there is a line like:
    - Recent agent numeric recommendations (last N): [1.00, 2.50, 3.75]
    This extracts the bracketed numbers.
    """
    # Take the last [...] bracket if multiple are present
    matches = re.findall(r"\[([^\]]+)\]", prompt)
    if not matches:
        return []
    nums_text = matches[-1]
    nums = []
    for tok in nums_text.split(","):
        tok = tok.strip()
        try:
            nums.append(float(tok))
        except:
            pass
    return nums

def clip1to5(x: float) -> float:
    return float(np.clip(x, 1.0, 5.0))

def compute_compliance(suggestion: float, action: float) -> float:
    # In simulator: 1 - |a - s| / range, with range=ptp(action_space) or 5.0
    rng = 5.0
    c = 1.0 - abs(float(action) - float(suggestion)) / rng
    return float(np.clip(c, 0.0, 1.0))

# -------------------------------
# UI State
# -------------------------------

def init_state():
    if "sim" not in st.session_state:
        st.session_state.sim = None
    if "logs" not in st.session_state:
        st.session_state.logs = {}        # session_id -> session_log(list)
    if "next_id" not in st.session_state:
        st.session_state.next_id = 1      # next session id after profiling
    if "last_step_info" not in st.session_state:
        st.session_state.last_step_info = {}
    if "inited" not in st.session_state:
        st.session_state.inited = False
    # Additional state for live chat streaming
    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs = []
    if "stop_flag" not in st.session_state:
        st.session_state.stop_flag = False


# -------------------------------
# Build simulator
# -------------------------------

def build_simulator(user_profile, model_name="gpt-5-nano"):
    action_space = np.linspace(0.0, 5.0, 100)
    # Important: Agent/User read OPENAI_API_KEY at __init__ time
    user = UserLlm(user_profile, model_name=model_name)
    agent = Agent(action_space=action_space,
                  user_age=user_profile.get("age"),
                  user_gender=user_profile.get("gender"),
                  model_name=model_name)
    sim = Simulator(user=user, agent=agent, action_space=action_space, total_steps=200)
    return sim


# -------------------------------
# One-step runner
# -------------------------------

def run_profiling(sim: Simulator):
    """Run Session 0 (profiling)."""
    session_log = sim.run_session(session_id=0, first_session=True)
    st.session_state.logs[0] = session_log
    st.session_state.next_id = 1
    return session_log

def run_next_session(sim: Simulator):
    """Run one RL step (one session), update traces and return a compact summary for UI."""
    sid = st.session_state.next_id
    suggestion, suggestion_idx, _ = sim.agent.policy()
    sim.suggestion_trace.append(suggestion)

    # Run one session with this suggestion
    session_log = sim.run_session(session_id=sid, planned_suggestion=suggestion)

    # After run_session, there is analysis saved; extract inferred action & compliance
    analysis_path = f"sessions/analysis_{sid:03}.json"
    if os.path.exists(analysis_path):
        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)
    else:
        analysis = {}

    inferred_action = analysis.get("user_action_estimate", None)
    try:
        inferred_action = float(inferred_action) if inferred_action is not None else None
    except:
        inferred_action = None

    compliance = analysis.get("compliance_estimate", None)
    try:
        compliance = float(compliance) if compliance is not None else None
    except:
        compliance = None

    if compliance is None and inferred_action is not None:
        compliance = sim.compute_compliance(suggestion, inferred_action)

    # GT action (if any) captured in session_log last row
    gt_action = session_log[-1].get("ground_truth_action") if session_log else None

    # Reward + agent updates (Q, estimates) using inferred action
    reward, _ = sim.agent.reward(
        suggestion_idx,
        inferred_action if inferred_action is not None else suggestion,
        compliance if compliance is not None else 0.0
    )
    sim._log_after_session(suggestion, inferred_action, gt_action, reward, compliance)

    # Persist
    st.session_state.logs[sid] = session_log
    st.session_state.next_id = sid + 1

    # Return brief for UI
    return {
        "session_id": sid,
        "suggestion": suggestion,
        "inferred_action": inferred_action,
        "gt_action": gt_action,
        "compliance": compliance,
        "reward": reward,
        "agent_monologue": session_log[-1].get("agent_monologue") if session_log else None
    }


def run_next_session_live(sim: Simulator, max_turns: int = 10):
    """
    Run one RL session with real-time chat output.
    This streams agent and user messages to st.session_state.chat_msgs and breaks early if st.session_state.stop_flag is True.
    Returns a summary dictionary similar to run_next_session.
    """
    sid = st.session_state.next_id
    suggestion, suggestion_idx, _ = sim.agent.policy()
    sim.suggestion_trace.append(suggestion)

    # Reset chat messages and stop flag for this live session
    st.session_state.chat_msgs = []
    st.session_state.stop_flag = False

    # Prepare conversation history and session log
    sim.conversation_history = []
    session_log = []
    planned_suggestion = suggestion

    for t in range(max_turns):
        # update agent context similar to Simulator.run_session
        sim.agent.run_context = {
            "session_id": sid,
            "current_turn": t + 1,
            "max_turns": max_turns,
            "total_sessions": sim.total_steps,
            "compliance_summary": sim._compliance_summary(window=10)
        }

        # Agent turn
        agent_result = sim._generate_agent_turn(sid, first_session=False, planned_suggestion=planned_suggestion)
        agent_utterance = agent_result.get("utterance", "")
        agent_monologue = agent_result.get("monologue", "")
        agent_end = agent_result.get("endkey", False)

        sim.conversation_history.append({"role": "assistant", "content": agent_utterance})

        log_entry = {
            "turn": t + 1,
            "agent_utterance": agent_utterance,
            "agent_monologue": agent_monologue,
            "agent_endkey": agent_end,
            "planned_suggestion": planned_suggestion
        }

        # Append agent message to chat
        st.session_state.chat_msgs.append(("assistant", agent_utterance))

        # If agent signals end of conversation, append log and break
        if agent_end:
            session_log.append(log_entry)
            break

        # User turn
        user_result = sim._generate_user_turn(sid)
        user_utterance = user_result.get("utterance", "")
        user_end = user_result.get("endkey", False)
        ground_truth_action = user_result.get("action", None)

        sim.conversation_history.append({"role": "user", "content": user_utterance})

        log_entry["user_utterance"] = user_utterance
        log_entry["user_endkey"] = user_end
        if ground_truth_action is not None:
            log_entry["ground_truth_action"] = ground_truth_action

        # Append user message to chat
        st.session_state.chat_msgs.append(("user", user_utterance))

        session_log.append(log_entry)

        # Break conditions: user or agent ended conversation
        if user_end:
            break

        # Stop flag: user clicked stop button
        if st.session_state.stop_flag:
            break

        # Short sleep to simulate real-time update
        time.sleep(0.2)

    # After live session: analyze and update estimates
    # Load analysis via sim._analyze_session
    analysis = sim._analyze_session(sim.conversation_history, sid, last_suggestion=planned_suggestion)

    inferred_action = analysis.get("user_action_estimate", None)
    try:
        inferred_action = float(inferred_action) if inferred_action is not None else None
    except:
        inferred_action = None

    compliance = analysis.get("compliance_estimate", None)
    try:
        compliance = float(compliance) if compliance is not None else None
    except:
        compliance = None

    if compliance is None:
        compliance = sim.compute_compliance(planned_suggestion, inferred_action)

    # Persist inferred action and compliance in last log entry
    if session_log:
        session_log[-1]["inferred_action"] = inferred_action
        session_log[-1]["compliance_estimate"] = compliance

    # Compute reward and update traces
    reward, _ = sim.agent.reward(
        suggestion_idx,
        inferred_action if inferred_action is not None else suggestion,
        compliance if compliance is not None else 0.0
    )
    sim._log_after_session(suggestion, inferred_action, session_log[-1].get("ground_truth_action") if session_log else None, reward, compliance)

    # Store session log
    st.session_state.logs[sid] = session_log
    st.session_state.next_id = sid + 1

    return {
        "session_id": sid,
        "suggestion": suggestion,
        "inferred_action": inferred_action,
        "gt_action": session_log[-1].get("ground_truth_action") if session_log else None,
        "compliance": compliance,
        "reward": reward,
        "agent_monologue": session_log[-1].get("agent_monologue") if session_log else None
    }


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Dietary Coaching Simulation", layout="wide")
init_state()

st.title("Dietary Coaching Simulation — Streamlit Dashboard")

with st.expander("🔧 실행 설정", expanded=True):
    col_b, col_c = st.columns([1, 1])
    with col_b:
        api_key_input = st.text_input("OPENAI_API_KEY", type="password", help="실제 API 사용 시 입력(선택)")
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
    with col_c:
        model_name = st.text_input("Model name", value="gpt-5-nano",
                                   help="실제 OpenAI 사용 시 지원되는 모델명으로 변경 가능")

# 기본 프로필(주신 main.py의 adaptive_user에 해당)
age_list = ["Teenager (10s)", "Young adult (20s)", "Adult (30s)", "Middle-aged (40s)", "Older adult (50+)"]
gender_list = ["Male", "Female", "Non-binary / Other", "Prefer not to say"]
condition_list = [
    "None", "Overeating (Hyperphagia)", "Binge eating disorder (BED)", "Anorexia nervosa",
    "Night eating syndrome", "Glycemic regulation issues", "Gastrointestinal disorders", "Other"
]
mu_list = [
    "Highly irregular eating patterns", "Somewhat irregular eating habits",
    "Moderately regular dietary routine", "Slightly structured meal schedule",
    "Strictly consistent eating habits"
]
beta_list = [
    "Highly resistant to dietary suggestions", "Somewhat resistant to behavioral influence",
    "Moderately compliant with guidance", "Easily influenced by suggestions",
    "Highly suggestible and reactive to guidance"
]
alpha_list = [
    "Extremely resistant to behavioral change", "Rarely adopts new eating behaviors",
    "Occasionally adapts eating habits", "Frequently adopts suggested behaviors",
    "Immediately responsive to new habits"
]
gamma_list = [
    "Insensitive to emotional or environmental stimuli", "Slightly responsive to contextual cues",
    "Moderately sensitive to external changes", "Highly influenced by situational factors",
    "Extremely vulnerable to emotional or environmental triggers"
]
memory_list = [
    "Poor recall of recent eating behaviors", "Able to recall patterns for about 1 week",
    "Able to recall for approximately 2 weeks", "Able to maintain pattern memory over 1 month",
    "Long-term retention of dietary routines"
]
delta_list = [
    "Highly reactive to small pattern changes", "Adapts with minimal stability required",
    "Moderately stable before behavior change", "Requires significant stability to change",
    "Changes only after long-term behavioral reinforcement"
]
epsilon_list = [
    "Behaves predictably with almost no deviations", "Rarely shows exceptions to routine",
    "Occasional deviation from typical patterns", "Frequently exhibits irregular behaviors",
    "Consistently unpredictable and erratic"
]

with st.expander("👤 사용자 프로파일 설정", expanded=not st.session_state.inited):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.selectbox("Age", age_list, index=2)
        gender = st.selectbox("Gender", gender_list, index=1)
    with c2:
        condition = st.selectbox("Condition", condition_list, index=2)
        mu = st.selectbox("μ (regularity)", mu_list, index=2)
    with c3:
        beta = st.selectbox("β (suggestion sensitivity)", beta_list, index=2)
        alpha = st.selectbox("α (adaptability)", alpha_list, index=2)
    with c4:
        gamma = st.selectbox("γ (emotional/env sensitivity)", gamma_list, index=2)
        memory = st.selectbox("Memory", memory_list, index=3)
    c5, c6 = st.columns(2)
    with c5:
        delta = st.selectbox("Δ (stability requirement)", delta_list, index=2)
    with c6:
        epsilon = st.selectbox("ε (irregularity tendency)", epsilon_list, index=2)

    do_init = st.button("사용자 프로파일 설정")
    if do_init:
        user_profile = {
            "name": "adaptive_user",
            "age": age,
            "gender": gender,
            "condition": condition,
            "mu": mu,
            "beta": beta,
            "alpha": alpha,
            "gamma": gamma,
            "memory": memory,
            "delta": delta,
            "epsilon": epsilon,
        }
        # Important: set API key BEFORE building instances
        st.session_state.sim = build_simulator(user_profile, model_name=model_name)
        st.session_state.logs = {}
        st.session_state.next_id = 1
        st.session_state.last_step_info = {}
        st.session_state.inited = True
        st.success("시뮬레이터가 초기화되었습니다.")

# Control panel
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 3])
with ctrl_col1:
    if st.button("프로파일링 세션 실행 (Session 0)"):
        if st.session_state.sim is None:
            st.warning("먼저 시뮬레이터를 초기화하세요.")
        else:
            run_profiling(st.session_state.sim)
            st.toast("Session 0 완료")

with ctrl_col2:
    # Traditional one-step execution
    if st.button("다음 세션 1스텝 실행"):
        if st.session_state.sim is None:
            st.warning("먼저 시뮬레이터를 초기화하세요.")
        elif 0 not in st.session_state.logs:
            st.warning("먼저 '프로파일링 세션'을 실행하세요.")
        else:
            info = run_next_session(st.session_state.sim)
            st.session_state.last_step_info = info
            st.toast(f"Session {info['session_id']} 완료")
    # Real-time live execution
    if st.button("실시간 세션 실행"):
        if st.session_state.sim is None:
            st.warning("먼저 시뮬레이터를 초기화하세요.")
        elif 0 not in st.session_state.logs:
            st.warning("먼저 '프로파일링 세션'을 실행하세요.")
        else:
            info = run_next_session_live(st.session_state.sim)
            st.session_state.last_step_info = info
            st.toast(f"실시간 Session {info['session_id']} 완료")

with ctrl_col3:
    steps = st.number_input("여러 스텝 연속 실행", min_value=1, max_value=200, value=5, step=1)
    if st.button("연속 실행"):
        if st.session_state.sim is None or 0 not in st.session_state.logs:
            st.warning("초기화 및 프로파일링 세션을 먼저 실행하세요.")
        else:
            prog = st.progress(0)
            for i in range(int(steps)):
                info = run_next_session(st.session_state.sim)
                st.session_state.last_step_info = info
                prog.progress((i + 1) / steps)
                # 짧은 sleep으로 UI 갱신 느낌(모의)
                time.sleep(0.05)
            st.toast(f"{steps} 스텝 실행 완료")
    # Button to stop live conversation
    if st.button("대화 중지"):
        st.session_state.stop_flag = True

st.markdown("---")

# Layout: Left (profile, metrics, charts) | Right (chat)
left, right = st.columns([1, 1])

# -------------------------------
# LEFT: Profile, latest metrics, compliance chart
# -------------------------------
with left:
    st.subheader("📇 사용자 프로파일 (설정값)")
    if st.session_state.sim is not None:
        st.json(st.session_state.sim.user.user_profile)

        # Inferred profile after Session 0
        if 0 in st.session_state.logs and hasattr(st.session_state.sim.agent, "inferred_user_profile"):
            st.subheader("🧭 에이전트 추론 프로파일 (Session 0 이후)")
            st.json(st.session_state.sim.agent.inferred_user_profile or {})

        # Latest metrics
        st.subheader("📊 최신 지표")
        last = st.session_state.last_step_info
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("최근 제안(숫자)", f"{last.get('suggestion', np.nan):.2f}" if last else "—")
        with m2:
            ia = last.get("inferred_action") if last else None
            st.metric("추론 행동", f"{ia:.2f}" if ia is not None else "—")
        with m3:
            comp = last.get("compliance") if last else None
            st.metric("컴플라이언스", f"{comp:.3f}" if comp is not None else "—")

        # Agent monologue
        if last and last.get("agent_monologue"):
            st.caption("Agent Monologue (내부 독백)")
            st.write(last["agent_monologue"])

        st.subheader("📈 Compliance (실시간)")
        comp_trace = st.session_state.sim.compliance_trace
        if comp_trace:
            st.line_chart(comp_trace)
        else:
            st.info("아직 컴플라이언스 데이터가 없습니다. 세션을 실행해 주세요.")

# -------------------------------
# RIGHT: Chat (Agent ↔ User)
# -------------------------------
with right:
    st.subheader("💬 대화 뷰")
    # If live chat messages exist, display them directly
    if st.session_state.chat_msgs:
        for role, msg in st.session_state.chat_msgs:
            # Role stored as 'assistant' or 'user'
            st.chat_message("assistant" if role == "assistant" else "user").write(msg)
    else:
        # Choose which session to display
        max_sid = max(st.session_state.logs.keys()) if st.session_state.logs else 0
        view_sid = st.number_input("세션 선택", min_value=0, max_value=int(max_sid), value=int(max_sid), step=1)

        if st.session_state.logs and view_sid in st.session_state.logs:
            log = st.session_state.logs[view_sid]
            st.caption(f"Session {view_sid} — 턴 로그")
            for row in log:
                st.chat_message("assistant").write(row.get("agent_utterance", ""))
                st.chat_message("user").write(row.get("user_utterance", ""))

            # Action snapshot for this session (if any)
            last_row = log[-1] if log else {}
            gt = last_row.get("ground_truth_action", None)
            inf = last_row.get("inferred_action", None)
            comp = last_row.get("compliance_estimate", None)
            st.markdown("—")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("GT Action(사용자 실제)", f"{gt:.2f}" if isinstance(gt, (int, float)) else "—")
            with c2:
                st.metric("Inferred Action(추론)", f"{inf:.2f}" if isinstance(inf, (int, float)) else "—")
            with c3:
                st.metric("Compliance", f"{comp:.3f}" if isinstance(comp, (int, float)) else "—")
        else:
            st.info("표시할 세션 로그가 없습니다. 먼저 세션을 실행해 주세요.")