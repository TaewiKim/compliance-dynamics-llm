# buffer.py
import torch
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.state_values = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.state_values.clear()

    def to_tensors(self, device):
        return {
            'states': torch.tensor(np.array(self.states), dtype=torch.float32, device=device),
            'actions': torch.tensor(self.actions, dtype=torch.float32, device=device),
            'log_probs': torch.tensor(self.log_probs, dtype=torch.float32, device=device),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32, device=device),
            'dones': torch.tensor(self.dones, dtype=torch.float32, device=device),
            'state_values': torch.tensor(self.state_values, dtype=torch.float32, device=device),
        }
