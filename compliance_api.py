"""
FastAPI-based backend for the dietary compliance simulator.

This module exposes RESTful endpoints so a separate front-end
can initialize the simulator, run profiling, start a new session,
step through a session turn by turn, retrieve chat messages, and
stop an in-progress session. It uses the existing Agent/User/Simulator
classes from the project. Sessions beyond the profiling phase use
reinforcement learning suggestions and track compliance and rewards.

Example API usage:

1. POST /init with a user profile to initialize the simulator.
2. POST /profiling to run the initial profiling session (Session 0).
3. POST /start_session to begin a new RL-driven session; returns the
   planned numeric suggestion.
4. POST /step to generate one agent/user exchange; repeat until
   `done` is True. When the session ends, the response includes
   summary fields (inferred action, compliance, reward, etc.).
5. POST /stop to abort the current session before completion.

The API maintains minimal internal state and does not depend on
Streamlit. Front-end code can call these endpoints to display
conversations in real time and control execution flow.
"""

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import json
import time

from agent import Agent
from user import UserLlm
from simulator import Simulator


app = FastAPI(title="Compliance Simulator API")


# ---------------------------------------------------------------------------
# Pydantic models for request bodies
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    name: str = "adaptive_user"
    age: str
    gender: str
    condition: str
    mu: str
    beta: str
    alpha: str
    gamma: str
    memory: str
    delta: str
    epsilon: str

class InitRequest(BaseModel):
    user_profile: UserProfile
    model_name: str = "gpt-5-nano"

class StartSessionResponse(BaseModel):
    session_id: int
    suggestion: float

class StepResponse(BaseModel):
    session_id: int
    turn: int
    agent_utterance: str
    user_utterance: Optional[str] = None
    done: bool
    summary: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Internal state variables
# These maintain the active simulator and session context across API calls.
# ---------------------------------------------------------------------------

sim: Optional[Simulator] = None
# Next session ID (1-based; Session 0 is profiling)
next_id: int = 1
session_logs: Dict[int, List[Dict[str, Any]]] = {}

# Variables for an in-progress RL session
current_session_id: Optional[int] = None
planned_suggestion: Optional[float] = None
turn_counter: int = 0
conversation_history: List[Dict[str, str]] = []
current_log: List[Dict[str, Any]] = []
stop_flag: bool = False
MAX_TURNS: int = 10


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_simulator(user_profile: Dict[str, Any], model_name: str = "gpt-5-nano") -> Simulator:
    """Instantiate a Simulator given a user profile and model name."""
    action_space = np.linspace(0.0, 5.0, 100)
    user = UserLlm(user_profile, model_name=model_name)
    agent = Agent(action_space=action_space,
                  user_age=user_profile.get("age"),
                  user_gender=user_profile.get("gender"),
                  model_name=model_name)
    return Simulator(user=user, agent=agent, action_space=action_space, total_steps=200)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.post("/init")
def init_simulator(req: InitRequest) -> Dict[str, Any]:
    """
    Initialize the simulator with a user profile and optional model name.
    Resets all internal session state and returns a status message.
    """
    global sim, next_id, session_logs
    global current_session_id, planned_suggestion, turn_counter
    global conversation_history, current_log, stop_flag

    # Build simulator
    sim = build_simulator(req.user_profile.dict(), model_name=req.model_name)
    next_id = 1
    session_logs = {}
    # Reset current session variables
    current_session_id = None
    planned_suggestion = None
    turn_counter = 0
    conversation_history = []
    current_log = []
    stop_flag = False
    return {"status": "initialized"}


@app.post("/profiling")
def run_profiling() -> Dict[str, Any]:
    """
    Run the profiling session (Session 0) to infer the user's profile.
    Returns the conversation log of the profiling session.
    """
    global sim, session_logs, next_id
    if sim is None:
        raise HTTPException(status_code=400, detail="Simulator is not initialized.")
    # Run session 0
    log = sim.run_session(session_id=0, first_session=True)
    session_logs[0] = log
    next_id = 1  # reset to 1 after profiling
    # Build a simple list of utterances for convenience
    chat = []
    for row in log:
        chat.append({"role": "assistant", "content": row.get("agent_utterance", "")})
        chat.append({"role": "user", "content": row.get("user_utterance", "")})
    return {"session_id": 0, "log": log, "chat": chat}


@app.post("/start_session", response_model=StartSessionResponse)
def start_session() -> StartSessionResponse:
    """
    Begin a new RL-driven session. Returns the session ID and the numeric suggestion
    selected by the agent's policy.
    """
    global sim, next_id
    global current_session_id, planned_suggestion, turn_counter
    global conversation_history, current_log, stop_flag

    if sim is None:
        raise HTTPException(status_code=400, detail="Simulator is not initialized.")
    # Determine session ID
    sid = next_id
    # Choose a suggestion using the agent's policy
    suggestion, suggestion_idx, _ = sim.agent.policy()
    # Record the suggestion for the simulator's trace
    sim.suggestion_trace.append(suggestion)
    # Prepare session state
    current_session_id = sid
    planned_suggestion = suggestion
    turn_counter = 0
    conversation_history = []
    current_log = []
    stop_flag = False
    return StartSessionResponse(session_id=sid, suggestion=float(suggestion))


@app.post("/step", response_model=StepResponse)
def step_session() -> StepResponse:
    """
    Execute a single turn (agent + user) in the current session and return the
    resulting utterances. When the session finishes, the response includes
    a summary of inferred action, compliance estimate, reward, etc.
    """
    global sim, current_session_id, planned_suggestion, turn_counter
    global conversation_history, current_log, session_logs, next_id
    global stop_flag, MAX_TURNS

    if sim is None or current_session_id is None:
        raise HTTPException(status_code=400, detail="No active session. Call /start_session first.")

    if stop_flag:
        # Session was externally stopped
        current_session_id = None
        return StepResponse(session_id=-1, turn=turn_counter, agent_utterance="", user_utterance=None, done=True, summary=None)

    # Limit the number of turns
    if turn_counter >= MAX_TURNS:
        # End the session due to reaching maximum turns
        summary = _finalize_session(current_session_id, planned_suggestion, conversation_history, current_log)
        current_session_id = None
        return StepResponse(session_id=summary["session_id"], turn=turn_counter, agent_utterance="", user_utterance=None, done=True, summary=summary)

    # Set run context like Simulator.run_session does
    sim.agent.run_context = {
        "session_id": current_session_id,
        "current_turn": turn_counter + 1,
        "max_turns": MAX_TURNS,
        "total_sessions": sim.total_steps,
        "compliance_summary": sim._compliance_summary(window=10)
    }

    # Agent turn
    agent_result = sim._generate_agent_turn(current_session_id, first_session=False, planned_suggestion=planned_suggestion)
    agent_utterance = agent_result.get("utterance", "")
    agent_monologue = agent_result.get("monologue", "")
    agent_end = agent_result.get("endkey", False)

    conversation_history.append({"role": "assistant", "content": agent_utterance})
    log_entry = {
        "turn": turn_counter + 1,
        "agent_utterance": agent_utterance,
        "agent_monologue": agent_monologue,
        "agent_endkey": agent_end,
        "planned_suggestion": planned_suggestion
    }

    # If agent indicates session should end before user responds
    if agent_end:
        current_log.append(log_entry)
        # Finalize session
        summary = _finalize_session(current_session_id, planned_suggestion, conversation_history, current_log)
        turn_counter += 1
        current_session_id = None
        return StepResponse(session_id=summary["session_id"], turn=turn_counter, agent_utterance=agent_utterance, user_utterance=None, done=True, summary=summary)

    # User turn
    user_result = sim._generate_user_turn(current_session_id)
    user_utterance = user_result.get("utterance", "")
    user_end = user_result.get("endkey", False)
    ground_truth_action = user_result.get("action", None)

    conversation_history.append({"role": "user", "content": user_utterance})
    log_entry["user_utterance"] = user_utterance
    log_entry["user_endkey"] = user_end
    if ground_truth_action is not None:
        log_entry["ground_truth_action"] = ground_truth_action

    current_log.append(log_entry)
    turn_counter += 1

    # If user ends the conversation, finalize
    if user_end:
        summary = _finalize_session(current_session_id, planned_suggestion, conversation_history, current_log)
        current_session_id = None
        return StepResponse(session_id=summary["session_id"], turn=turn_counter, agent_utterance=agent_utterance, user_utterance=user_utterance, done=True, summary=summary)

    # Continue session
    return StepResponse(session_id=current_session_id, turn=turn_counter, agent_utterance=agent_utterance, user_utterance=user_utterance, done=False, summary=None)


@app.post("/stop")
def stop_session() -> Dict[str, Any]:
    """
    Abort the current session. The current session state is cleared without
    updating the simulator's logs or traces. Subsequent calls to /step will
    indicate there is no active session.
    """
    global current_session_id, planned_suggestion, turn_counter
    global conversation_history, current_log, stop_flag
    if current_session_id is None:
        return {"status": "no-active-session"}
    # Signal stop and clear session context
    stop_flag = True
    current_session_id = None
    planned_suggestion = None
    turn_counter = 0
    conversation_history = []
    current_log = []
    return {"status": "stopped"}


@app.get("/chat")
def get_chat() -> Dict[str, Any]:
    """
    Retrieve the conversation history for the current session. If no session
    is active, returns an empty list.
    """
    return {"chat": conversation_history}


# ---------------------------------------------------------------------------
# Internal helper to finalize a session and compute summary
# ---------------------------------------------------------------------------

def _finalize_session(sid: int, suggestion: float, history: List[Dict[str, str]], log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform session analysis, compute inferred action, compliance, reward, and
    update global simulator logs/traces. Returns a summary dictionary.
    """
    global sim, session_logs, next_id
    # Perform analysis on the conversation history
    analysis = sim._analyze_session(history, sid, last_suggestion=suggestion)
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
        compliance = sim.compute_compliance(suggestion, inferred_action)
    # Attach estimates to last log row
    if log:
        log[-1]["inferred_action"] = inferred_action
        log[-1]["compliance_estimate"] = compliance
    # Calculate reward and update traces
    reward, _ = sim.agent.reward(
        sim.action_space.tolist().index(suggestion) if hasattr(sim, "action_space") else 0,
        inferred_action if inferred_action is not None else suggestion,
        compliance if compliance is not None else 0.0
    )
    gt_action = log[-1].get("ground_truth_action") if log else None
    sim._log_after_session(suggestion, inferred_action, gt_action, reward, compliance)
    # Save session log and increment next_id
    session_logs[sid] = log.copy()
    next_id = max(next_id, sid + 1)
    # Build summary
    summary = {
        "session_id": sid,
        "suggestion": suggestion,
        "inferred_action": inferred_action,
        "gt_action": gt_action,
        "compliance": compliance,
        "reward": reward,
        "agent_monologue": log[-1].get("agent_monologue") if log else None
    }
    return summary