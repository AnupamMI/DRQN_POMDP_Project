import torch
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

from env import GridWorld
from drqn import DRQN


# ================= SEED FUNCTION =================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ================= READ SEED =================
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
set_seed(seed)
print("Running DRQN with seed:", seed)


# ================= ENV + MODEL =================
env = GridWorld(grid_size=8, dynamic_obstacles=False, obstacle_prob=0.12)
# DRQN expects flattened 5x5 observations (25) as input vectors
model = DRQN(input_size=25, hidden=64, actions=4)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()


# ================= TRAINING PARAMETERS =================
episodes = 600
gamma = 0.99
epsilon = 1.0
# slower decay so DRQN explores longer
epsilon_decay = 0.995
epsilon_min = 0.05
# memory depth for 8x8 maps
sequence_length = 8

reward_history = []


# ================= TRAIN LOOP =================
for episode in range(episodes):

    # env.reset() returns a 5x5 array; flatten to a vector
    state = env.reset().flatten()

    # start with empty history and pad when forming sequences
    history = []

    total_reward = 0
    done = False
    step_count = 0
    max_steps = 100

    # initial hidden state is None so LSTM will use zeros internally
    h = None

    # accumulate losses for truncated BPTT (update every `sequence_length` steps)
    accum_losses = []

    while not done and step_count < max_steps:
        step_count += 1

        # append current state and form fixed-length input
        history.append(state)
        if len(history) < sequence_length:
            pad = [np.zeros_like(state)] * (sequence_length - len(history))
            seq_list = pad + history
        else:
            seq_list = history[-sequence_length:]

        state_tensor = torch.FloatTensor(np.array(seq_list)).unsqueeze(0)

        # Epsilon-greedy (use model for greedy decisions)
        # preserve previous hidden for consistent learning
        h_prev = h
        if random.random() < epsilon:
            action = random.randint(0, 3)
            # still run forward to advance hidden (no-grad)
            with torch.no_grad():
                _, h = model(state_tensor, h_prev)
        else:
            q_values, h = model(state_tensor, h_prev)
            action = torch.argmax(q_values).item()

        next_state, reward, done = env.step(action)
        next_state = next_state.flatten()

        # prepare next sequence (pad at front with next_state if short)
        next_hist = history + [next_state]
        if len(next_hist) < sequence_length:
            next_seq_list = [next_state] * (sequence_length - len(next_hist)) + next_hist
        else:
            next_seq_list = next_hist[-sequence_length:]

        next_tensor = torch.FloatTensor(np.array(next_seq_list)).unsqueeze(0)

        # compute q(s,a) using the hidden before the forward (h_prev)
        # if we used random action and didn't compute q_values, compute it now for learning
        if 'q_values' not in locals():
            q_values, _ = model(state_tensor, h_prev)

        q_sa = q_values[0, action]

        # bootstrap target: use h_prev so target conditions match training state
        with torch.no_grad():
            next_q, _ = model(next_tensor, h_prev)
            if done:
                target = torch.tensor(reward, dtype=q_sa.dtype)
            else:
                target = torch.tensor(reward, dtype=q_sa.dtype) + gamma * torch.max(next_q)

        target_q = q_values.clone()
        target_q[0][action] = target

        loss = loss_fn(q_values, target_q)

        accum_losses.append(loss)

        # perform optimizer step every `sequence_length` steps or at episode end
        if len(accum_losses) >= sequence_length or done:
            optimizer.zero_grad()
            total_loss = torch.stack(accum_losses).mean()
            total_loss.backward()
            optimizer.step()
            accum_losses = []

            # detach hidden to truncate BPTT
            if h is not None:
                h = (h[0].detach(), h[1].detach())

        # cleanup for next iteration
        if 'q_values' in locals():
            del q_values

        history = next_hist
        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    reward_history.append(total_reward)

    print(f"Episode {episode+1}, Reward: {total_reward}")


# ================= SAVE SEED-SPECIFIC FILES =================
np.save(f"drqn_rewards_seed{seed}.npy", np.array(reward_history))
torch.save(model.state_dict(), f"drqn_model_seed{seed}.pth")


# ================= OPTIONAL PLOT =================
plt.plot(reward_history)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title(f"DRQN Training (Seed {seed})")
plt.show()

print("Model saved successfully.")