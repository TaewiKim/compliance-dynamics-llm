import numpy as np

if __name__ == "__main__":
    from user import AdaptiveUser
    from agent import Agent
    from simulator import Simulator

    action_space = np.linspace(0.0, 5.0, 100)

    user_cases = [
        {"name": "independent_user", "mu": 2.0, "beta": 10.0, "alpha": 0.00, "gamma": 0.1},
        {"name": "compliant_user", "mu": 2.0, "beta": 0.2, "alpha": 0.00, "gamma": 0.1},
        {"name": "adaptive_user", "mu": 2.0, "beta": 0.5, "alpha": 0.10, "gamma": 0.1},
        {"name": "high_noise_user", "mu": 2.0, "beta": 0.5, "alpha": 0.00, "gamma": 0.8},
        {"name": "resistant_user", "mu": 2.0, "beta": 1.5, "alpha": 0.001, "gamma": 0.1},
    ]

    saved_images = []

    for case in user_cases:
        print(f"\n===== Running Simulation: {case['name']} =====")
        user = AdaptiveUser(
            mu=case["mu"],
            beta=case["beta"],
            alpha=case["alpha"],
            gamma=case["gamma"]
        )
        agent = Agent(action_space=action_space)
        sim = Simulator(user=user, agent=agent, action_space=action_space,
                        total_steps=400, warmup_steps=10)
        sim.warmup()
        sim.train()
        png_path = sim.plot(save=True, filename=f"{case['name']}")
        saved_images.append(png_path)

    print("\nSaved plot images:", saved_images)

