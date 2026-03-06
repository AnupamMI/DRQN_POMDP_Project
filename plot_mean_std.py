import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------- SETTINGS ----------------
NUM_SEEDS = 5
SMOOTH_WINDOW = 20


# ---------------- SMOOTH FUNCTION ----------------
def smooth(data, window):
    return np.convolve(data, np.ones(window) / window, mode='valid')


# ---------------- LOAD REWARD FILES ----------------
dqn_runs = []
drqn_runs = []

# Collect available per-seed files for DQN and DRQN
available_dqn_seeds = []
available_drqn_seeds = []
for seed in range(NUM_SEEDS):
    dqn_path = f"dqn_rewards_seed{seed}.npy"
    drqn_path = f"drqn_rewards_seed{seed}.npy"
    if os.path.exists(dqn_path):
        available_dqn_seeds.append(seed)
        dqn_runs.append(np.load(dqn_path))
    if os.path.exists(drqn_path):
        available_drqn_seeds.append(seed)
        drqn_runs.append(np.load(drqn_path))

print(f"Found DQN seed files: {available_dqn_seeds}")
print(f"Found DRQN seed files: {available_drqn_seeds}")

# If DRQN per-seed files are missing, try aggregated fallbacks (drqn.npy or drqn_*.npy)
if len(drqn_runs) == 0:
    # try common aggregated names
    for alt in ("drqn.npy", "drqn_4.npy", "drqn_8.npy", "drqn_16.npy"):
        if os.path.exists(alt):
            try:
                agg = np.load(alt, allow_pickle=True)
                agg = np.array(agg)
                # if agg is 2D: (seeds, episodes)
                if agg.ndim == 2:
                    # take up to NUM_SEEDS rows
                    rows = min(agg.shape[0], NUM_SEEDS)
                    for i in range(rows):
                        drqn_runs.append(agg[i])
                    print(f"Loaded DRQN aggregated data from {alt} (using {rows} seeds)")
                    break
            except Exception:
                continue

if len(dqn_runs) == 0:
    raise FileNotFoundError("No DQN seed files found. Please run `train_dqn.py` for seeds 0..4 to produce dqn_rewards_seed{seed}.npy files.")
if len(drqn_runs) == 0:
    raise FileNotFoundError("No DRQN runs found. Place per-seed files `drqn_rewards_seed{seed}.npy` or an aggregated `drqn.npy` in the working directory.")

# Align lengths by truncating to the shortest run among the two groups
min_len_dqn = min(len(r) for r in dqn_runs)
min_len_drqn = min(len(r) for r in drqn_runs)
min_len = min(min_len_dqn, min_len_drqn)

if SMOOTH_WINDOW >= min_len:
    # reduce smoothing window if it's too large
    SMOOTH_WINDOW = max(1, min_len // 10)
    print(f"Adjusted SMOOTH_WINDOW to {SMOOTH_WINDOW} due to short runs (min_len={min_len})")

dqn_runs = np.array([r[:min_len] for r in dqn_runs])
drqn_runs = np.array([r[:min_len] for r in drqn_runs])


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