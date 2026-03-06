import numpy as np
from train_models import train_dqn, train_drqn


def main():
    seeds = [0, 1, 2, 3, 4]

    dqn_results = []
    drqn_results = {2: [], 4: [], 8: [], 16: []}

    for s in seeds:
        print(f"Running DQN seed={s}")
        dqn_results.append(train_dqn(s))

        for seq in [2, 4, 8, 16]:
            print(f"Running DRQN seed={s} seq={seq}")
            drqn_results[seq].append(train_drqn(s, seq))

    np.save("dqn.npy", dqn_results)
    for seq in drqn_results:
        np.save(f"drqn_{seq}.npy", drqn_results[seq])


if __name__ == "__main__":
    main()
