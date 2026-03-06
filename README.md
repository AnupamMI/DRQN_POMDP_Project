# DRQN_POMDP_Project

## Overview

This project implements and compares two reinforcement learning agents — **Deep Q-Network (DQN)** and **Deep Recurrent Q-Network (DRQN)** — in a **partially observable grid-world environment (POMDP)**.

The objective is to demonstrate how **temporal memory using an LSTM layer** helps an agent perform better when the environment cannot be fully observed at each timestep.

The project includes:

- Training multiple agents across **16 random seeds**
- **Statistical significance testing**
- **Learning curve analysis**
- **Memory depth experiments**
- **Visualization of agent behavior**

---

## Environment

A custom **GridWorld** environment was designed for the experiments.

| Property | Description |
|--------|-------------|
Grid Size | 8 × 8
Observation | Agent sees a **5×5 local region** around its position
Actions | Up, Down, Left, Right
Obstacles | Randomly generated
Goal | Reach the goal position
Rewards | +10 for reaching goal, -1 per step

Because the agent only observes a **local window**, the environment becomes a **Partially Observable Markov Decision Process (POMDP)**.

---

## Agents Implemented

### DQN (Baseline)

- Feedforward neural network
- Uses current observation only
- No memory of previous states

### DRQN (Proposed)

- LSTM-based Q-network
- Maintains **hidden state memory**
- Uses sequences of observations

This allows the agent to **infer hidden state information over time**.

---

## Repository Structure

| File | Description |
|------|-------------|
`env.py` | Custom GridWorld environment |
`dqn.py` | DQN network architecture |
`drqn.py` | DRQN (LSTM-based) architecture |
`train_dqn.py` | Training script for DQN |
`train_drqn.py` | Training script for DRQN |
`play_animation.py` | Side-by-side animation of DQN vs DRQN |
`compare_models.py` | Plot learning curves |
`stats_test.py` | Statistical comparison of models |
`analyze_hidden.py` | Analyze DRQN hidden state dynamics |
`plot_final_boxplot.py` | Reward distribution visualization |
`requirements.txt` | Python dependencies |

Saved experiment outputs:
dqn_rewards_seed*.npy
drqn_rewards_seed*.npy
dqn_model_seed*.pth
drqn_model_seed*.pth


---

## Installation

Clone the repository:

```bash
git clone https://github.com/AnupamMI/DRQN_POMDP_Project.git
cd DRQN_POMDP_Project
Create a virtual environment:

python -m venv .venv

Activate the environment:

Windows
.\.venv\Scripts\activate
Linux / Mac
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt
Training the Agents

Train the DQN agent:

python train_dqn.py

Train the DRQN agent:

python train_drqn.py

Multiple seeds can be used to ensure statistical robustness.

Evaluation and Analysis
Learning Curve

Compare DQN and DRQN performance:

python compare_models.py
Statistical Significance Test

Run paired statistical analysis:

python stats_test.py

Outputs include:

Mean performance

Relative improvement

Paired t-test

Wilcoxon signed-rank test

Effect size (Cohen's d)

Confidence intervals

Reward Distribution Visualization
python plot_final_boxplot.py

This shows the reward distribution across seeds.

Memory Depth Analysis

Different sequence lengths were tested:

Sequence lengths tested:
2
4
8
16

This experiment studies the impact of temporal memory length on DRQN performance.

Animation Demo

Visualize agents navigating the environment:

python play_animation.py

This animation compares:

Agent trajectories

Obstacle avoidance

Memory effects

Experimental Results

Across 16 training seeds:

Metric	DQN	DRQN
Mean Reward	-51.63	-39.67
Improvement	—	+23.17%

Statistical significance:

Paired t-test: p = 0.020
Wilcoxon test: p = 0.021

Effect size:

Cohen's d = 0.65

Result:

DRQN significantly outperforms DQN in partially observable environments.

Key Takeaways

DRQN handles partial observability better due to memory.

Performance improvement is statistically significant.

Memory depth influences agent performance.

Future Work

Possible extensions include:

Double DQN / Dueling DQN

Transformer-based RL memory

Larger environments

Continuous control tasks

Author

Anupam Mishra
