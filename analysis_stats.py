import numpy as np
from pathlib import Path
from scipy import stats

LAST_K = 50


def load_per_seed(prefix):
    """
    Load reward files automatically for all seeds found.
    """
    values = []
    seeds = []

    for seed in range(100):  # search many possible seeds
        file = Path(f"{prefix}_rewards_seed{seed}.npy")

        if file.exists():
            rewards = np.load(file)
            values.append(rewards[-LAST_K:].mean())
            seeds.append(seed)

    return np.array(values), seeds


def main():

    # ── load reward data ──────────────────────────────────
    dqn_final, dqn_seeds = load_per_seed("dqn")
    drqn_final, drqn_seeds = load_per_seed("drqn")

    common_seeds = sorted(set(dqn_seeds).intersection(drqn_seeds))

    if len(common_seeds) == 0:
        raise RuntimeError("No matching seed files found.")

    dqn_final = np.array([np.load(f"dqn_rewards_seed{s}.npy")[-LAST_K:].mean()
                          for s in common_seeds])

    drqn_final = np.array([np.load(f"drqn_rewards_seed{s}.npy")[-LAST_K:].mean()
                           for s in common_seeds])

    n = len(common_seeds)

    # ── statistics ─────────────────────────────────────────
    diff = drqn_final - dqn_final

    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1)

    cohens_d = mean_diff / std_diff if std_diff > 0 else np.nan

    # confidence interval
    se = std_diff / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)

    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se

    # paired t-test
    t_stat, p_val = stats.ttest_rel(dqn_final, drqn_final)

    # Wilcoxon test
    try:
        w_stat, w_p = stats.wilcoxon(dqn_final, drqn_final)
    except Exception:
        w_stat, w_p = (np.nan, np.nan)

    # relative improvement
    rel_improvement = (drqn_final.mean() - dqn_final.mean()) / abs(dqn_final.mean()) * 100

    # ── output ─────────────────────────────────────────────
    print("\n==== DRQN vs DQN Statistical Comparison ====\n")

    print(f"Seeds used                 : {common_seeds}")
    print(f"Number of runs             : {n}")

    print("\nPerformance (last 50 episodes):")
    print(f"DQN  : {dqn_final.mean():.3f} ± {dqn_final.std(ddof=1):.3f}")
    print(f"DRQN : {drqn_final.mean():.3f} ± {drqn_final.std(ddof=1):.3f}")

    print(f"\nMean difference (DRQN - DQN) : {mean_diff:.3f}")
    print(f"Relative improvement        : {rel_improvement:.2f}%")

    print(f"\nCohen's d (effect size)     : {cohens_d:.3f}")
    print(f"95% CI for difference       : [{ci_low:.3f}, {ci_high:.3f}]")

    print("\nStatistical Tests:")
    print(f"Paired t-test  : t={t_stat:.3f}, p={p_val:.6f}")
    print(f"Wilcoxon test  : W={w_stat:.3f}, p={w_p:.6f}")

    if p_val < 0.05:
        print("\nResult: DRQN significantly outperforms DQN (p < 0.05)")
    else:
        print("\nResult: Difference not statistically significant")

    print("\n=============================================\n")


if __name__ == "__main__":
    main()