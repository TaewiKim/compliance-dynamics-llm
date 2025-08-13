# Compliance Dynamics Simulator

A psychologically-grounded simulation framework where adaptive user models interact with a reinforcement learning (RL) agent.  
It explores how varying levels of user compliance, adaptation, and noise impact agent policy learning and behavioral outcomes.

---

## 🚀 Overview

This project implements:

- **AdaptiveUser**:  
  A user model with an internal behavior mean (μ) that probabilistically follows suggestions, with parameters controlling compliance sensitivity, adaptation, and noise.
  
- **Agent**:  
  A reinforcement learning agent that suggests behaviors to the user, estimates user compliance, learns Q-values, and adjusts its softmax policy temperature.

- **Simulator**:  
  Orchestrates the warm-up phase, main training loop, logging, and generates multi-panel plots visualizing:
  - Suggestions vs. user actions
  - Rewards
  - Compliance tracking (Real vs Estimated)
  - Temperature evolution
  - Estimated vs. true behavior mean
  - Q-value dynamics

- **Streamlit Dashboard**:  
  Interactive UI to:
  - Configure user profile and agent settings
  - Run profiling (Session 0) and step-by-step sessions
  - View real-time metrics and compliance charts
  - Inspect chat transcripts between agent and user
  - Switch between **Mock mode** (offline) and **Live mode** (real API)

---

## ⚙️ Features

✅ **Multi-user scenarios**:
- Independent user (low compliance)
- Highly compliant user
- Adaptive user (adjusts μ over time)
- High-noise user
- Resistant user (slow adaptation)

✅ **Logging**:
- Suggestions, actions, rewards
- Compliance probability (real & estimated)
- Estimated μ, policy temperature, Q-values

✅ **Visualizations**:
- Multi-panel plots saved under `plots/`
- Live compliance chart in Streamlit
- Session-level conversation logs

✅ **New Live Mode Support**:
- Agent/User/Simulator subclasses run full process without mock mode
- Passes current session/turn counts and compliance history to Agent prompts

---

## 🧩 Installation

```bash
git clone https://github.com/TaewiKim/compliance-dynamics-llm.git
cd compliance-dynamics-llm
pip install -r requirements.txt
