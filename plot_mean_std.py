import numpy as np
import matplotlib.pyplot as plt

# ---------------- SETTINGS ----------------
NUM_SEEDS = 5
SMOOTH_WINDOW = 20


# ---------------- SMOOTH FUNCTION ----------------
def smooth(data, window):
    return np.convolve(data, np.ones(window) / window, mode='valid')


# ---------------- LOAD REWARD FILES ----------------
dqn_runs = []
drqn_runs = []

for seed in range(NUM_SEEDS):
    dqn_runs.append(np.load(f"dqn_rewards_seed{seed}.npy"))
    drqn_runs.append(np.load(f"drqn_rewards_seed{seed}.npy"))

dqn_runs = np.array(dqn_runs)
drqn_runs = np.array(drqn_runs)


# ---------------- COMPUTE MEAN & STD ----------------
dqn_mean = dqn_runs.mean(axis=0)
dqn_std = dqn_runs.std(axis=0)

drqn_mean = drqn_runs.mean(axis=0)
drqn_std = drqn_runs.std(axis=0)


# ---------------- SMOOTH CURVES ----------------
dqn_mean_s = smooth(dqn_mean, SMOOTH_WINDOW)
dqn_std_s = smooth(dqn_std, SMOOTH_WINDOW)

drqn_mean_s = smooth(drqn_mean, SMOOTH_WINDOW)
drqn_std_s = smooth(drqn_std, SMOOTH_WINDOW)

episodes = np.arange(len(dqn_mean_s))


# ---------------- PLOT ----------------
plt.figure(figsize=(10, 6))

plt.plot(episodes, dqn_mean_s, label="DQN", linewidth=2)
plt.fill_between(
    episodes,
    dqn_mean_s - dqn_std_s,
    dqn_mean_s + dqn_std_s,
    alpha=0.2
)

plt.plot(episodes, drqn_mean_s, label="DRQN", linewidth=2)
plt.fill_between(
    episodes,
    drqn_mean_s - drqn_std_s,
    drqn_mean_s + drqn_std_s,
    alpha=0.2
)

plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Mean ± Standard Deviation over 5 Seeds (Smoothed)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ---------------- FINAL PERFORMANCE PRINT ----------------
final_window = 50

dqn_final = dqn_mean[-final_window:].mean()
drqn_final = drqn_mean[-final_window:].mean()

print("\nFinal 50-Episode Average Reward:")
print(f"DQN   : {dqn_final:.2f}")
print(f"DRQN  : {drqn_final:.2f}")

improvement = ((drqn_final - dqn_final) / abs(dqn_final)) * 100
print(f"Relative Improvement of DRQN: {improvement:.2f}%")