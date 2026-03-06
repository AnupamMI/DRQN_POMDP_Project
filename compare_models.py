import numpy as np
import matplotlib.pyplot as plt

# Load aggregated experiment results
dqn = np.load("dqn.npy")
drqn = np.load("drqn_16.npy")

# Average across seeds
dqn_rewards = dqn.mean(axis=0)
drqn_rewards = drqn.mean(axis=0)

def moving_average(data, window=20):
    return np.convolve(data, np.ones(window)/window, mode='valid')

dqn_smooth = moving_average(dqn_rewards)
drqn_smooth = moving_average(drqn_rewards)

plt.figure(figsize=(12,7))
plt.plot(dqn_smooth, label="DQN", linewidth=2)
plt.plot(drqn_smooth, label="DRQN (seq=16)", linewidth=2)

plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("DQN vs DRQN Training Performance")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("\nFinal 50 Episode Average:")
print("DQN:", np.mean(dqn_rewards[-50:]))
print("DRQN:", np.mean(drqn_rewards[-50:]))

print("\nReward Std Dev:")
print("DQN:", np.std(dqn_rewards))
print("DRQN:", np.std(drqn_rewards))

print("\nMax Reward:")
print("DQN:", np.max(dqn_rewards))
print("DRQN:", np.max(drqn_rewards))