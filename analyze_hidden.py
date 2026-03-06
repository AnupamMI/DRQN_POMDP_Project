import torch
import numpy as np
import matplotlib.pyplot as plt

from env import GridWorld
from drqn import DRQN

# Load trained model (try common filenames)
model = DRQN(input_size=25, hidden=64, actions=4)
for fname in ("drqn_model.pth", "drqn_model_seed1.pth", "drqn_model_seed0.pth"):
    try:
        model.load_state_dict(torch.load(fname, map_location=torch.device("cpu")))
        break
    except Exception:
        continue
model.eval()

env = GridWorld(grid_size=8, dynamic_obstacles=True, obstacle_prob=0.15)

state = env.reset()
state = state.flatten()

sequence_length = 16
# initialize with zero-padding and current state at the end
state_sequence = [np.zeros_like(state) for _ in range(sequence_length - 1)] + [state]

# use model LSTM hidden size
hsize = getattr(model.lstm, "hidden_size", 64)
hidden = (
    torch.zeros(1, 1, hsize),
    torch.zeros(1, 1, hsize)
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