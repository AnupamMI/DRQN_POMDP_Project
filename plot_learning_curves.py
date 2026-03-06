import numpy as np
import matplotlib.pyplot as plt

NUM_SEEDS = 16

dqn_runs = []
drqn_runs = []

# load reward files
for seed in range(NUM_SEEDS):
    dqn_runs.append(np.load(f"dqn_rewards_seed{seed}.npy"))
    drqn_runs.append(np.load(f"drqn_rewards_seed{seed}.npy"))

dqn_runs = np.array(dqn_runs)
drqn_runs = np.array(drqn_runs)

# compute mean and std
dqn_mean = dqn_runs.mean(axis=0)
dqn_std  = dqn_runs.std(axis=0)

drqn_mean = drqn_runs.mean(axis=0)
drqn_std  = drqn_runs.std(axis=0)

episodes = np.arange(len(dqn_mean))

plt.figure(figsize=(8,5))

plt.plot(episodes, dqn_mean, label="DQN", color="#00b4d8")
plt.fill_between(episodes,
                 dqn_mean-dqn_std,
                 dqn_mean+dqn_std,
                 alpha=0.2,
                 color="#00b4d8")

plt.plot(episodes, drqn_mean, label="DRQN", color="#ff6b6b")
plt.fill_between(episodes,
                 drqn_mean-drqn_std,
                 drqn_mean+drqn_std,
                 alpha=0.2,
                 color="#ff6b6b")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Learning Curve: DQN vs DRQN")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()