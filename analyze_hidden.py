import torch
import numpy as np
import matplotlib.pyplot as plt

from env import GridWorld
from drqn import DRQN

# Load trained model
model = DRQN()
model.load_state_dict(torch.load("drqn_model.pth"))
model.eval()

env = GridWorld()

state = env.reset()
state = state.flatten()

sequence_length = 4
state_sequence = [state for _ in range(sequence_length)]

hidden = (
    torch.zeros(1, 1, 64),
    torch.zeros(1, 1, 64)
)

hidden_norms = []
done = False
step_count = 0
max_steps = 50

while not done and step_count < max_steps:

    state_tensor = torch.FloatTensor(np.array(state_sequence)).unsqueeze(0)

    with torch.no_grad():
        q_values, hidden = model(state_tensor, hidden)

    # Compute LSTM hidden state norm
    h = hidden[0]
    norm = torch.norm(h).item()
    hidden_norms.append(norm)

    action = torch.argmax(q_values).item()

    next_state, reward, done = env.step(action)
    next_state = next_state.flatten()

    state_sequence = state_sequence[1:] + [next_state]

    step_count += 1

plt.plot(hidden_norms)
plt.xlabel("Time Step")
plt.ylabel("LSTM Hidden State Norm")
plt.title("Hidden State Dynamics (DRQN)")
plt.show()