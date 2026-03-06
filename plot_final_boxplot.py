import numpy as np
import matplotlib.pyplot as plt

NUM_SEEDS = 16
LAST_K = 50

dqn_final = []
drqn_final = []

# Load reward files
for seed in range(NUM_SEEDS):
    dqn_rewards = np.load(f"dqn_rewards_seed{seed}.npy")
    drqn_rewards = np.load(f"drqn_rewards_seed{seed}.npy")

    dqn_final.append(dqn_rewards[-LAST_K:].mean())
    drqn_final.append(drqn_rewards[-LAST_K:].mean())

dqn_final = np.array(dqn_final)
drqn_final = np.array(drqn_final)

# Create boxplot
plt.figure(figsize=(6,5))

plt.boxplot(
    [dqn_final, drqn_final],
    labels=["DQN", "DRQN"],
    patch_artist=True
)

plt.ylabel("Final Average Reward (Last 50 Episodes)")
plt.title("Final Reward Distribution Across Seeds")

plt.grid(alpha=0.3)

plt.show()
