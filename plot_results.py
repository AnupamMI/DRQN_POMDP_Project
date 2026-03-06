import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def safe_load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path)


def plot_training_curve(dqn_path: Path, drqn_path: Path):
    dqn = safe_load(dqn_path)
    drqn = safe_load(drqn_path)

    # Expect shape (n_seeds, n_episodes)
    dqn_mean = dqn.mean(axis=0)
    drqn_mean = drqn.mean(axis=0)

    plt.figure(figsize=(8, 4.5))
    plt.plot(dqn_mean, label="DQN")
    plt.plot(drqn_mean, label="DRQN (seq=4)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    plt.show()


def plot_memory_depth(seqs, out_prefix="memory_depth"):
    means = []
    for s in seqs:
        path = Path(f"drqn_{s}.npy")
        if not path.exists():
            raise FileNotFoundError(f"Missing results file for seq {s}: {path}")
        data = np.load(path)  # shape (n_seeds, n_episodes)
        # compute per-seed mean over last 50 episodes, then average across seeds
        if data.ndim == 1:
            # single-run case, treat as one seed
            last_mean = data[-50:].mean()
        else:
            per_seed = data[:, -50:].mean(axis=1)
            last_mean = per_seed.mean()
        means.append(last_mean)

    plt.figure(figsize=(6, 4))
    plt.plot(seqs, means, marker="o")
    plt.xlabel("Sequence Length")
    plt.ylabel("Final Average Reward (mean over seeds)")
    plt.title("Effect of Memory Depth")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=150)
    plt.show()


def main():
    try:
        plot_training_curve(Path("dqn.npy"), Path("drqn_4.npy"))
    except Exception as e:
        print("Could not plot training curve:", e)

    seqs = [2, 4, 8, 16]
    try:
        plot_memory_depth(seqs)
    except Exception as e:
        print("Could not plot memory depth:", e)


if __name__ == "__main__":
    main()
