import numpy as np
from scipy.stats import ttest_rel

LAST_K = 50
NUM_SEEDS = 16


def load_final_rewards(prefix):
    values = []

    for seed in range(NUM_SEEDS):
        file = f"{prefix}_seed{seed}.npy"
        data = np.load(file)

        # mean reward of last 50 episodes
        mean_reward = data[-LAST_K:].mean()
        values.append(mean_reward)

    return np.array(values)


def main():

    # Load rewards
    dqn_final = load_final_rewards("dqn_rewards")
    drqn_final = load_final_rewards("drqn_rewards")

    print("Seeds used:", list(range(NUM_SEEDS)))

    print("\nDQN Final Rewards (last 50 episodes):")
    print(dqn_final)

    print("\nDRQN Final Rewards (last 50 episodes):")
    print(drqn_final)

    # Mean and std
    dqn_mean = dqn_final.mean()
    drqn_mean = drqn_final.mean()

    dqn_std = dqn_final.std(ddof=1)
    drqn_std = drqn_final.std(ddof=1)

    print("\n---- Performance ----")
    print(f"DQN  Mean: {dqn_mean:.3f} ± {dqn_std:.3f}")
    print(f"DRQN Mean: {drqn_mean:.3f} ± {drqn_std:.3f}")

    # Paired t-test
    t_stat, p_value = ttest_rel(dqn_final, drqn_final)

    print("\n---- Statistical Test ----")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Result: Statistically Significant (p < 0.05)")
    else:
        print("Result: Not Statistically Significant")

    # Effect size
    diff = drqn_final - dqn_final
    cohen_d = diff.mean() / diff.std(ddof=1)

    print("\n---- Effect Size ----")
    print(f"Cohen's d: {cohen_d:.4f}")


if __name__ == "__main__":
    main()