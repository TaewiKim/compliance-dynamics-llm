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
# Mock mode patching
# -------------------------------

def patch_mock_mode(sim: Simulator):
    """
    Monkey-patch Simulator.generate_response and Simulator._analyze_session
    so the app can run without real API calls. This preserves the JSON
    formats expected by your existing code.
    """

    def mock_generate_response(self, role, content, history, model_name, api_url, api_key, headers,
                               return_raw=False, retries=5, backoff_base=1.0, timeout=60):
        # Build a minimal 'messages' just to keep shape similar
        messages = history + [{"role": ("assistant" if role == "agent" else "user"), "content": content}]

        # Agent (first session) — ask one simple question
        if role == "agent" and "first session" in content.lower():
            parsed = {
                "utterance": "최근 식사 패턴을 한 가지만 알려주실래요? 예: 아침은 거르는 편이에요.",
                "monologue": "첫 세션: 라포 형성과 현재 습관 탐색.",
                "endkey": False
            }

        # Agent (ongoing session) — supportive suggestion aligned with planned numeric suggestion
        elif role == "agent":
            planned = parse_last_suggestion_from_agent_prompt(content, default=3.2)
            # Keep numeric suggestion private; utterance is qualitative
            parsed = {
                "utterance": "오늘은 저녁에 채소 반찬 한 가지를 먼저 접시에 담아보는 건 어때요? 너무 부담스럽지 않게 작은 한 걸음부터요.",
                "monologue": "사용자 순응성 중간, 작은 행동 유도. 계획된 숫자 제안에 맞춰 강도 조절.",
                "endkey": False
            }

        # User — sometimes takes an action (endkey=True) and returns action ~ last suggestion
        elif role == "user":
            recs = parse_recent_recs_from_user_prompt(content)
            last_rec = recs[-1] if recs else 3.0
            will_act = random.random() < 0.35
            if will_act:
                action = clip1to5(random.gauss(last_rec, 0.6))
                parsed = {
                    "utterance": "좋아요, 오늘은 말씀하신 대로 조금 더 신경 써볼게요.",
                    "endkey": True,
                    "action": action
                }
                # expose to analyzer
                self._mock_last_action = action
            else:
                parsed = {
                    "utterance": "음… 오늘은 일이 많아서 크게 바꾸기는 어려울 것 같아요. 내일은 시도해볼게요.",
                    "endkey": False
                }
                self._mock_last_action = None
        else:
            parsed = {"utterance": "OK", "endkey": False}

        raw = json.dumps(parsed, ensure_ascii=False)
        return {"parsed": parsed, "raw": raw, "messages": messages} if return_raw else parsed

    def mock_analyze_session(self, conversation_history, session_id, last_suggestion):
        # Prefer the last 'ground truth' action if user took one in the mocked user turn
        action_est = self._mock_last_action if getattr(self, "_mock_last_action", None) is not None \
                     else clip1to5(random.gauss(last_suggestion, 0.8))
        comp = compute_compliance(last_suggestion, action_est)
        analysis = {
            "user_action_estimate": round(float(action_est), 3),
            "compliance_estimate": round(float(comp), 3),
            "confidence": 0.65 if getattr(self, "_mock_last_action", None) is not None else 0.5,
            "basis": "대화의 수용적 톤과 최근 제안 강도 대비 반응을 근거로 추정.",
            "cognitive_dissonance": "변화 의지는 있으나 일정/피로로 인해 망설임.",
            "negative_thought_patterns": "일시적 회피, 부담감.",
            "emotional_triggers": "업무 스트레스, 시간 부족.",
            "effective_reinforcement": ["Empathy", "Small wins", "Specific next step"],
            "coaching_notes": "작은 행동부터, 구체적 실행조건(시간/장소) 제안."
        }
        # Save like the original
        os.makedirs("sessions", exist_ok=True)
        with open(f"sessions/analysis_{session_id:03}.json", "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        return analysis

    sim.generate_response = types.MethodType(mock_generate_response, sim)
    sim._analyze_session = types.MethodType(mock_analyze_session, sim)


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
    if "mock_mode" not in st.session_state:
        st.session_state.mock_mode = True
    if "inited" not in st.session_state:
        st.session_state.inited = False


# -------------------------------
# Build simulator
# -------------------------------

def build_simulator(user_profile, model_name="gpt-5-nano", mock_mode=True):
    action_space = np.linspace(0.0, 5.0, 100)
    # Important: Agent/User read OPENAI_API_KEY at __init__ time
    user = UserLlm(user_profile, model_name=model_name)
    agent = Agent(action_space=action_space,
                  user_age=user_profile.get("age"),
                  user_gender=user_profile.get("gender"),
                  model_name=model_name)
    sim = Simulator(user=user, agent=agent, action_space=action_space, total_steps=200)
    if mock_mode:
        patch_mock_mode(sim)
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


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Dietary Coaching Simulation", layout="wide")
init_state()

st.title("Dietary Coaching Simulation — Streamlit Dashboard")

with st.expander("🔧 실행 설정", expanded=True):
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        st.session_state.mock_mode = st.toggle("Mock 모드(오프라인 실행)", value=st.session_state.mock_mode,
                                               help="OFF로 두고 OPENAI_API_KEY를 설정하면 실제 API로 동작합니다.")
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

    do_init = st.button("시뮬레이터 초기화")
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
        st.session_state.sim = build_simulator(user_profile, model_name=model_name,
                                               mock_mode=st.session_state.mock_mode)
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
    if st.button("다음 세션 1스텝 실행"):
        if st.session_state.sim is None:
            st.warning("먼저 시뮬레이터를 초기화하세요.")
        elif 0 not in st.session_state.logs:
            st.warning("먼저 '프로파일링 세션'을 실행하세요.")
        else:
            info = run_next_session(st.session_state.sim)
            st.session_state.last_step_info = info
            st.toast(f"Session {info['session_id']} 완료")

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
