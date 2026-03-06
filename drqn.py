import torch
import torch.nn as nn


class DRQN(nn.Module):
    def __init__(self, input_size=25, hidden=64, actions=4):
        super().__init__()

        # stronger function approximator: Linear -> LSTM -> Linear
        self.fc1 = nn.Linear(input_size, 64)
        self.lstm = nn.LSTM(64, hidden, batch_first=True)
        self.fc2 = nn.Linear(hidden, actions)

    def forward(self, x, h):
        # x: (batch, seq_len, input_size)
        x = self.fc1(x)
        out, h = self.lstm(x, h)
        q = self.fc2(out[:, -1, :])
        return q, h