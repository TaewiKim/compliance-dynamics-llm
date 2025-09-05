# compliance_simulator_standalone.py (최종 수정 버전)
import os
import re
import json
import random
import time
import numpy as np
import streamlit as st
import requests
from requests.exceptions import HTTPError, Timeout, ConnectionError as ReqConnectionError

# agent.py와 user.py는 import하여 사용합니다.
from agent import Agent
from user import UserLlm

# -------------------------------
# 시뮬레이션 로직
# -------------------------------

def generate_response(role, content, history, model_name, api_url, api_key, headers, retries=5, backoff_base=1.0, timeout=60):
    """LLM API를 호출하여 응답을 생성하는 함수"""
    if not api_key:
        st.error("OPENAI_API_KEY가 설정되지 않았습니다. API 키를 입력하고 시뮬레이터를 다시 초기화해주세요.")
        return None
    messages = history + [{"role": "user" if role == "user" else "assistant", "content": content}]
    payload = {"model": model_name, "messages": messages, "response_format": {"type": "json_object"}}
    last_err = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
            RETRY_STATUS = {429, 500, 502, 503, 504}
            if response.status_code in RETRY_STATUS and attempt < retries:
                raise HTTPError(f"일시적인 HTTP 오류: {response.status_code}", response=response)
            response.raise_for_status()
            result = response.json()
            message_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
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
            return parsed
        except (Timeout, ReqConnectionError, HTTPError) as e:
            last_err = e
            if attempt < retries:
                wait = backoff_base * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(wait)
            else:
                st.error(f"API 호출에 최종 실패했습니다: {last_err}")
                return None
        except Exception as e:
            st.error(f"응답 처리 중 예외가 발생했습니다: {e}")
            return None


def run_full_session(session_id, agent, user, first_session=False, planned_suggestion=None, max_turns=7):
    """한 번의 전체 세션(대화)을 실행하고, 끝난 뒤 행동을 결정하는 함수"""
    conversation_history = []
    session_log = []
    
    # --- 1. 대화 세션 진행 ---
    # 실시간 대화 출력을 위한 컨테이너 선택
    chat_container = st.session_state.get('chat_container')

    with chat_container:
        st.info(f"**[Session {session_id}]** 대화 시작...")

    for t in range(max_turns):
        # Agent Turn
        prompt_context = {"session_id": session_id, "current_turn": t + 1, "max_turns": max_turns}
        if first_session:
            prompt = agent.format_agent_1st_session_prompt()
        else:
            prompt = agent.format_agent_prompt(
                suggestion_score=planned_suggestion,
                suggestion_history=st.session_state.suggestion_trace[-10:],
                prior_analysis=st.session_state.get("last_analysis", {}),
                planned_suggestion=planned_suggestion
            )
        agent_res = generate_response("agent", prompt, conversation_history, agent.model_name, agent.api_url, agent.api_key, agent.headers)
        if agent_res is None: return None
        agent_utterance = agent_res.get("utterance", "")
        conversation_history.append({"role": "assistant", "content": agent_utterance})
        with chat_container:
            st.chat_message("assistant").write(agent_utterance) # 실시간 출력

        # User Turn
        user_prompt = user.format_user_prompt(
            recommendation_history=st.session_state.suggestion_trace,
            action_history=st.session_state.ground_truth_action_trace
        )
        user_res = generate_response("user", user_prompt, conversation_history, user.model_name, user.api_url, user.api_key, user.headers)
        if user_res is None: return None
        user_utterance = user_res.get("utterance", "")
        user_end = user_res.get("endkey", False)
        conversation_history.append({"role": "user", "content": user_utterance})
        with chat_container:
            st.chat_message("user").write(user_utterance) # 실시간 출력
        
        log_entry = {"turn": t + 1, "agent_utterance": agent_utterance, "user_utterance": user_utterance}
        session_log.append(log_entry)

        if user_end:
            break
    
    with chat_container:
        st.info(f"**[Session {session_id}]** 대화가 종료되었습니다. 사용자 행동 및 에이전트 분석을 시작합니다.")
    ground_truth_action = None

    # --- 2. 사용자 행동 결정 (After Conversation) ---
    action_prompt = user.format_user_action_prompt(conversation_history)
    action_res = generate_response("user", action_prompt, [], user.model_name, user.api_url, user.api_key, user.headers)
    if action_res and "action" in action_res:
        ground_truth_action = float(action_res["action"])
        with chat_container:
            st.success(f"↳ 사용자가 최종 행동으로 **{ground_truth_action:.2f}** 을(를) 결정했습니다.")
        if session_log:
            session_log[-1]["ground_truth_action"] = ground_truth_action

    # --- 3. 에이전트의 세션 분석 (After Session Analysis) ---
    if first_session:
        analysis_prompt = agent.format_agent_1st_session_analysis_prompt()
        analysis = generate_response("agent", analysis_prompt, conversation_history, agent.model_name, agent.api_url, agent.api_key, agent.headers)
        if analysis:
            agent.inferred_user_profile = analysis.get("inferred_attributes", {})
            agent.goal_behavior = analysis.get("goal_behavior", 4.0)
    else:
        analysis_prompt = agent.format_agent_session_analysis_prompt(last_suggestion=planned_suggestion)
        analysis = generate_response("agent", analysis_prompt, conversation_history, agent.model_name, agent.api_url, agent.api_key, agent.headers)
        if analysis:
            st.session_state.last_analysis = analysis
            if session_log:
                session_log[-1]["inferred_action"] = analysis.get("user_action_estimate")
                session_log[-1]["compliance_estimate"] = analysis.get("compliance_estimate")
    
    with chat_container:
        st.info(f"**[Session {session_id}]** 분석이 완료되었습니다.")

    return session_log


def compute_compliance(suggestion, action):
    if suggestion is None or action is None: return 0.0
    rng = 5.0
    c = 1.0 - abs(float(action) - float(suggestion)) / rng
    return float(np.clip(c, 0.0, 1.0))

def init_app_state():
    if "inited" not in st.session_state:
        st.session_state.inited = False
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "user" not in st.session_state:
        st.session_state.user = None
    if "logs" not in st.session_state:
        st.session_state.logs = {}
    if "next_id" not in st.session_state:
        st.session_state.next_id = 1
    if "last_step_info" not in st.session_state:
        st.session_state.last_step_info = {}
    trace_keys = ["suggestion_trace", "inferred_action_trace", "ground_truth_action_trace", "reward_trace", "compliance_trace"]
    for key in trace_keys:
        if key not in st.session_state:
            st.session_state[key] = []
init_app_state()

st.title("Dietary Coaching Simulation (실시간 대화)")

with st.expander("🔧 실행 설정", expanded=True):
    api_key_input = st.text_input("OPENAI_API_KEY", type="password", help="OpenAI API 사용 시 키를 입력하세요.")
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
    model_name = st.text_input("Model name", value="gpt-4-turbo", help="실제 OpenAI 사용 시 지원되는 모델명으로 변경 가능")

age_list = ["Teenager (10s)", "Young adult (20s)", "Adult (30s)", "Middle-aged (40s)", "Older adult (50+)"]
gender_list = ["Male", "Female", "Non-binary / Other", "Prefer not to say"]
condition_list = ["None", "Overeating (Hyperphagia)", "Binge eating disorder (BED)", "Anorexia nervosa"]
mu_list = ["Highly irregular eating patterns", "Moderately regular dietary routine", "Strictly consistent eating habits"]
beta_list = ["Highly resistant to dietary suggestions", "Moderately compliant with guidance", "Highly suggestible"]
alpha_list = ["Extremely resistant to behavioral change", "Occasionally adapts eating habits", "Immediately responsive"]
gamma_list = ["Insensitive to emotional stimuli", "Moderately sensitive", "Extremely vulnerable to triggers"]
memory_list = ["Poor recall", "Able to recall for approximately 2 weeks", "Long-term retention"]
delta_list = ["Highly reactive to small pattern changes", "Moderately stable before change", "Requires long-term reinforcement"]
epsilon_list = ["Behaves predictably", "Occasional deviation", "Consistently unpredictable"]

with st.expander("👤 사용자 프로파일 설정", expanded=not st.session_state.inited):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.selectbox("Age", age_list, index=2)
        gender = st.selectbox("Gender", gender_list, index=1)
        condition = st.selectbox("Condition", condition_list, index=2)
    with c2:
        mu = st.selectbox("μ (regularity)", mu_list, index=1)
        beta = st.selectbox("β (suggestion sensitivity)", beta_list, index=1)
        alpha = st.selectbox("α (adaptability)", alpha_list, index=1)
    with c3:
        gamma = st.selectbox("γ (emotional/env sensitivity)", gamma_list, index=1)
        memory = st.selectbox("Memory", memory_list, index=1)
        delta = st.selectbox("Δ (stability requirement)", delta_list, index=1)
        epsilon = st.selectbox("ε (irregularity tendency)", epsilon_list, index=1)
    
    if st.button("사용자 프로파일로 시뮬레이터 초기화"):
        user_profile = {
            "name": "custom_user", "age": age, "gender": gender, "condition": condition,
            "mu": mu, "beta": beta, "alpha": alpha, "gamma": gamma,
            "memory": memory, "delta": delta, "epsilon": epsilon,
        }
        action_space = np.linspace(0.0, 5.0, 100)
        st.session_state.agent = Agent(action_space=action_space, user_age=age, user_gender=gender, model_name=model_name)
        st.session_state.user = UserLlm(user_profile, model_name=model_name)
        init_app_state() 
        st.session_state.inited = True
        st.success("시뮬레이터가 성공적으로 초기화되었습니다.")
        st.rerun()

def run_next_simulation_step():
    agent = st.session_state.agent
    user = st.session_state.user
    sid = st.session_state.next_id
    suggestion, suggestion_idx, _ = agent.policy()
    st.session_state.suggestion_trace.append(suggestion)
    session_log = run_full_session(sid, agent, user, planned_suggestion=suggestion)
    if not session_log:
        st.error(f"Session {sid} 실행 중 API 오류가 발생했습니다.")
        st.session_state.suggestion_trace.pop()
        return None
    last_log = session_log[-1] if session_log else {}
    inferred_action = last_log.get("inferred_action")
    compliance = last_log.get("compliance_estimate")
    try: inferred_action = float(inferred_action) if inferred_action is not None else None
    except (ValueError, TypeError): inferred_action = None
    try: compliance = float(compliance) if compliance is not None else None
    except (ValueError, TypeError): compliance = None
    if compliance is None and inferred_action is not None:
        compliance = compute_compliance(suggestion, inferred_action)
    reward, _ = agent.reward(
        suggestion_idx,
        inferred_action if inferred_action is not None else suggestion,
        compliance if compliance is not None else 0.0
    )
    st.session_state.reward_trace.append(reward)
    st.session_state.compliance_trace.append(compliance)
    st.session_state.inferred_action_trace.append(inferred_action)
    st.session_state.ground_truth_action_trace.append(last_log.get("ground_truth_action"))
    st.session_state.logs[sid] = session_log
    st.session_state.next_id += 1
    info = {
        "session_id": sid, "suggestion": suggestion, "inferred_action": inferred_action,
        "compliance": compliance, "reward": reward,
        "agent_monologue": st.session_state.get("last_analysis", {}).get("coaching_notes", "")
    }
    st.session_state.last_step_info = info
    return info

col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    if st.button("프로파일링 (Session 0)"):
        if not st.session_state.inited:
            st.warning("먼저 시뮬레이터를 초기화하세요.")
        else:
            run_full_session(0, st.session_state.agent, st.session_state.user, first_session=True)

with col2:
    if st.button("다음 1스텝 실행"):
        if not st.session_state.inited:
            st.warning("먼저 시뮬레이터를 초기화하세요.")
        elif 0 not in st.session_state.logs and st.session_state.next_id == 1:
             st.warning("먼저 '프로파일링 세션'을 실행하세요.")
        else:
            run_next_simulation_step()

with col3:
    steps_to_run = st.number_input("연속 실행 스텝 수", min_value=1, max_value=50, value=5, step=1)
    if st.button("연속 실행"):
        if not st.session_state.inited or (0 not in st.session_state.logs and st.session_state.next_id == 1):
            st.warning("초기화 및 프로파일링 세션을 먼저 실행하세요.")
        else:
            prog_bar = st.progress(0, text="연속 실행 준비 중...")
            for i in range(int(steps_to_run)):
                info = run_next_simulation_step()
                if not info: break
                prog_bar.progress((i + 1) / steps_to_run, text=f"Session {info['session_id']} 실행 완료")
            st.toast(f"{steps_to_run} 스텝 실행 완료")

st.markdown("---")

left, right = st.columns([1, 1])
with left:
    st.subheader("📇 사용자 프로파일")
    if st.session_state.user:
        st.json(st.session_state.user.user_profile)
    if st.session_state.agent and st.session_state.agent.inferred_user_profile:
        st.subheader("🧭 에이전트 추론 프로파일 (Session 0 이후)")
        st.json(st.session_state.agent.inferred_user_profile)
    st.subheader("📊 최신 지표")
    last = st.session_state.last_step_info
    m1, m2, m3 = st.columns(3)
    m1.metric("최근 제안", f"{last.get('suggestion', 0):.2f}" if last else "—")
    m2.metric("추론 행동", f"{last.get('inferred_action', 0):.2f}" if last.get('inferred_action') is not None else "—")
    m3.metric("순응도", f"{last.get('compliance', 0):.3f}" if last.get('compliance') is not None else "—")
    if last.get("agent_monologue"):
        with st.container(border=True):
            st.caption("Agent Coaching Notes")
            st.write(last["agent_monologue"])
    st.subheader("📈 Compliance Trace")
    if st.session_state.compliance_trace:
        st.line_chart(st.session_state.compliance_trace)
    else:
        st.info("세션을 실행하면 순응도 그래프가 표시됩니다.")

with right:
    st.subheader("💬 대화 로그")
    # 실시간 대화를 표시할 컨테이너
    chat_container = st.container(height=500)
    st.session_state['chat_container'] = chat_container

    # 이전 로그를 표시하고 싶을 경우
    show_history = st.checkbox("이전 세션 로그 보기")
    if show_history:
        if st.session_state.logs:
            max_sid = max(st.session_state.logs.keys())
            view_sid = st.number_input("표시할 세션 선택", min_value=0, max_value=int(max_sid), value=int(max_sid), step=1)
            log = st.session_state.logs.get(view_sid, [])
            for row in log:
                # 여기서 에이전트 발화 키 수정
                st.chat_message("assistant").write(row.get("agent_utterance", ""))
                st.chat_message("user").write(row.get("user_utterance", ""))
        else:
            st.info("표시할 이전 로그가 없습니다.")