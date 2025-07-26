import numpy as np
import torch
import matplotlib.pyplot as plt
from agent import PPOAgent
from user import AdaptiveUser
from simulator import PPOTrainerSimulator

if __name__ == '__main__':
    state_dim = 2  # 입력 상태 벡터 크기

    # Continuous action space 범위
    action_low = 2.0
    action_high = 5.0

    # 사용자 모델 정의
    user = AdaptiveUser(mu=2.0, beta=0.5, alpha=0.1, gamma=0.1)

    # 에이전트 정의
    agent = PPOAgent(
        state_dim=state_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 시뮬레이터 실행
    sim = PPOTrainerSimulator(
        user=user,
        ppo_agent=agent,
        total_steps=1000,
        update_interval=20,
        action_low=action_low,
        action_high=action_high
    )

    sim.run()

    # ─────────────────────────────
    # 시각화: 4-패널 subplot 구성
    # ─────────────────────────────
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # 1. Suggestion vs User Action
    axs[0].plot(sim.logs['suggestions'], label='Agent Suggestion', color='green', linestyle='--', alpha=0.7)
    axs[0].scatter(range(len(sim.logs['actions'])), sim.logs['actions'], label='User Action', color='blue', s=10)
    axs[0].axhline(y=2.0, color='gray', linestyle='--', label='Initial Mu')
    axs[0].axhline(y=4.0, color='red', linestyle=':', label='Goal (G)')
    axs[0].set_ylabel("Action Value")
    axs[0].set_title("Agent Suggestion vs User Action")
    axs[0].legend()
    axs[0].grid(True)

    # 2. Reward
    axs[1].plot(sim.logs['rewards'], label='Reward', color='orange')
    axs[1].set_ylabel("Reward")
    axs[1].set_title("Reward over Time")
    axs[1].legend()
    axs[1].grid(True)

    # 3. 평균 Reward per PPO update
    axs[2].plot(sim.logs['update_steps'], sim.logs['avg_rewards'], label='Avg Reward per Update', marker='o')
    axs[2].set_ylabel("Avg Reward")
    axs[2].set_title("Average Reward per PPO Update")
    axs[2].legend()
    axs[2].grid(True)

    # 4. 사용자 평균 행동 μ
    axs[3].plot(sim.logs['true_mu'], label='True Mu (User)', color='purple')
    axs[3].axhline(y=2.0, color='gray', linestyle='--', label='Initial Mu')
    axs[3].axhline(y=4.0, color='red', linestyle=':', label='Goal (G)')
    axs[3].set_xlabel("Step")
    axs[3].set_ylabel("Behavior Mean")
    axs[3].set_title("User Behavior Mean (True Mu) Over Time")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()
