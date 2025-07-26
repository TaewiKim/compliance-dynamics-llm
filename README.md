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
  - Compliance tracking
  - Temperature evolution
  - Estimated vs. true behavior mean
  - Q-value dynamics

---

## ⚙️ Features

✅ Multi-user scenarios:
- Independent user (low compliance)
- Highly compliant user
- Adaptive user (adjusts μ over time)
- High-noise user
- Resistant user (slow adaptation)

✅ Logs stepwise data:
- Suggestions, actions, rewards, compliance probability
- Estimated μ (behavior mean), estimated compliance
- Policy temperature, Q-values

✅ Auto-generates plots saved under `plots/`.

---

## 🧩 Installation

```bash
git clone https://github.com/TaewiKim/compliance-dynamics.git
cd compliance-dynamics
pip install -r requirements.txt