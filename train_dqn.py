import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
import random

from env import GridWorld
from dqn import DQN

env = GridWorld()
model = DQN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

episodes = 300
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
reward_history = []

for episode in range(episodes):
    state = env.reset()
    state = torch.FloatTensor(state.flatten()).unsqueeze(0)

    total_reward = 0
    done = False

    step_count = 0
    max_steps = 100

    while not done and step_count < max_steps:
        step_count += 1
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                q_values = model(state)
                action = torch.argmax(q_values).item()

        next_state, reward, done = env.step(action)
        next_state = torch.FloatTensor(next_state.flatten()).unsqueeze(0)

        target = reward
        if not done:
            with torch.no_grad():
                target += gamma * torch.max(model(next_state)).item()

        q_values = model(state)
        target_q = q_values.clone()
        target_q[0][action] = target

        loss = loss_fn(q_values, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    reward_history.append(total_reward)
print(f"Episode {episode+1}, Reward: {total_reward}")

np.save('dqn_rewards.npy', np.array(reward_history))

plt.plot(reward_history)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("DQN Training Performance (POMDP)")
plt.show()

def moving_average(data, window_size=20):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

smoothed = moving_average(reward_history)

plt.plot(smoothed)
plt.xlabel("Episodes")
plt.ylabel("Smoothed Reward")
plt.title("Smoothed Training Curve")
plt.show()