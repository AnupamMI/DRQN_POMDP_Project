import random
import numpy as np
import torch
import torch.optim as optim

from env import GridWorld
from dqn import DQN
from drqn import DRQN


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_dqn(seed: int) -> list:
    """Train a DQN agent for a given seed and return reward history."""
    set_seed(seed)

    env = GridWorld(grid_size=8, dynamic_obstacles=False, obstacle_prob=0.12)
    model = DQN(input_size=25, actions=4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    episodes = 600
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    reward_history = []

    loss_fn = torch.nn.MSELoss()

    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state.flatten()).unsqueeze(0)

        total_reward = 0
        done = False
        step_count = 0
        max_steps = 100

        while not done and step_count < max_steps:
            step_count += 1

            if np.random.rand() < epsilon:
                action = np.random.randint(0, 4)
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

    return reward_history


def train_drqn(seed: int, seq_len: int) -> list:
    """Train a DRQN agent with memory depth `seq_len` and return rewards."""
    set_seed(seed)
    env = GridWorld(grid_size=8, dynamic_obstacles=False, obstacle_prob=0.12)
    model = DRQN(input_size=25, hidden=64, actions=4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    episodes = 600
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    reward_history = []

    for episode in range(episodes):
        state = env.reset().flatten()

        # start with empty history; we'll pad to `seq_len` when forming inputs
        history = []

        total = 0
        done = False

        # initial hidden state None so LSTM uses zeros
        h = None

        accum_losses = []

        for step in range(100):
            # append current observation
            history.append(state)

            # form a fixed-length sequence (zero-pad at front if too short)
            if len(history) < seq_len:
                pad = [np.zeros_like(state)] * (seq_len - len(history))
                seq_list = pad + history
            else:
                seq_list = history[-seq_len:]

            seq = torch.FloatTensor(np.array(seq_list, dtype=np.float32)).unsqueeze(0)

            # preserve previous hidden for consistent transitions
            h_prev = h
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 4)
                with torch.no_grad():
                    _, h = model(seq, h_prev)
            else:
                q, h = model(seq, h_prev)
                action = torch.argmax(q).item()

            next_state, reward, done = env.step(action)
            next_state = next_state.flatten()

            # prepare next sequence for bootstrap (pad similarly)
            next_hist = history + [next_state]
            if len(next_hist) < seq_len:
                next_seq_list = [np.zeros_like(state)] * (seq_len - len(next_hist)) + next_hist
            else:
                next_seq_list = next_hist[-seq_len:]

            next_seq = torch.FloatTensor(np.array(next_seq_list, dtype=np.float32)).unsqueeze(0)

            # compute Q(s,a) using h_prev (the hidden before the forward)
            q_values, _ = model(seq, h_prev)
            q_sa = q_values[0, action]

            with torch.no_grad():
                q_next, _ = model(next_seq, h_prev)
                if done:
                    target = torch.tensor(reward, dtype=q_sa.dtype)
                else:
                    target = torch.tensor(reward, dtype=q_sa.dtype) + gamma * torch.max(q_next)

            loss = (q_sa - target) ** 2

            accum_losses.append(loss)

            if len(accum_losses) >= seq_len or done:
                optimizer.zero_grad()
                total_loss = torch.stack(accum_losses).mean()
                total_loss.backward()
                optimizer.step()
                accum_losses = []

                if h is not None:
                    h = tuple([x.detach() for x in h])

            state = next_state
            total += reward

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        reward_history.append(total)

    return reward_history
