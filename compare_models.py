import numpy as np
import matplotlib.pyplot as plt

# Load reward histories from both models
dqn_rewards = np.load('dqn_rewards.npy')
drqn_rewards = np.load('drqn_rewards.npy')

# Smoothing function
def moving_average(data, window=20):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# Calculate smoothed curves
dqn_smooth = moving_average(dqn_rewards)
drqn_smooth = moving_average(drqn_rewards)

# Plot both raw and smoothed curves
plt.figure(figsize=(12, 7))
plt.plot(dqn_smooth, label="DQN (Smoothed)", linewidth=2.5, alpha=0.8)
plt.plot(drqn_smooth, label="DRQN (Smoothed)", linewidth=2.5, alpha=0.8)
plt.xlabel("Episodes", fontsize=12)
plt.ylabel("Total Reward", fontsize=12)
plt.title("DQN vs DRQN Training Performance (Smoothed)", fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Quantitative Metrics
print("=" * 50)
print("QUANTITATIVE PERFORMANCE METRICS")
print("=" * 50)
print("\nFinal 50 Episode Average:")
print(f"DQN:  {np.mean(dqn_rewards[-50:]):.2f}")
print(f"DRQN: {np.mean(drqn_rewards[-50:]):.2f}")

print("\nReward Std Dev (Stability):")
print(f"DQN:  {np.std(dqn_rewards):.2f}")
print(f"DRQN: {np.std(drqn_rewards):.2f}")

print("\nMax Reward Achieved:")
print(f"DQN:  {np.max(dqn_rewards):.2f}")
print(f"DRQN: {np.max(drqn_rewards):.2f}")
print("=" * 50)
