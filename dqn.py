import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim=9, action_dim=4):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.network(x)