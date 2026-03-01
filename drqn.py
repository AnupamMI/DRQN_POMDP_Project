import torch
import torch.nn as nn

class DRQN(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, action_dim=4):
        super(DRQN, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        q_values = self.fc(out[:, -1, :])
        return q_values, hidden