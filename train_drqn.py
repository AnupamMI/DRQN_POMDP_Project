import torch
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

from env import GridWorld
from drqn import DRQN

env = GridWorld()
model = DRQN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

episodes = 300
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

sequence_length = 4
reward_history = []

for episode in range(episodes):

    state = env.reset()
    state = state.flatten()

    # Store last N observations
    state_sequence = [state for _ in range(sequence_length)]

    total_reward = 0
    done = False

    step_count = 0
    max_steps = 100

    # Initialize hidden state
    hidden = (
        torch.zeros(1, 1, 64),
        torch.zeros(1, 1, 64)
    )

    while not done and step_count < max_steps:
        step_count += 1

        state_tensor = torch.FloatTensor(np.array(state_sequence)).unsqueeze(0)

        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                q_values, hidden = model(state_tensor, hidden)
                action = torch.argmax(q_values).item()

        next_state, reward, done = env.step(action)
        next_state = next_state.flatten()

        next_sequence = state_sequence[1:] + [next_state]
        next_tensor = torch.FloatTensor(np.array(next_sequence)).unsqueeze(0)

        target = reward

        if not done:
            with torch.no_grad():
                next_hidden = (
                    torch.zeros(1, 1, 64),
                    torch.zeros(1, 1, 64)
                )
                next_q, _ = model(next_tensor, next_hidden)
                target += gamma * torch.max(next_q).item()

        hidden = (hidden[0].detach(), hidden[1].detach())
        q_values, hidden = model(state_tensor, hidden)
        target_q = q_values.clone()
        target_q[0][action] = target

        loss = loss_fn(q_values, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state_sequence = next_sequence
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    reward_history.append(total_reward)
    print(f"Episode {episode+1}, Reward: {total_reward}")

np.save('drqn_rewards.npy', np.array(reward_history))

plt.plot(reward_history)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("DRQN Training Performance (POMDP)")
plt.show()

def moving_average(data, window_size=20):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

smoothed = moving_average(reward_history)

plt.plot(smoothed)
plt.xlabel("Episodes")
plt.ylabel("Smoothed Reward")
plt.title("Smoothed Training Curve")
plt.show()

# save trained model weights
torch.save(model.state_dict(), "drqn_model.pth")
print("Model saved successfully.")