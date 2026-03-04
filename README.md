# DRQN_POMDP_Project

## Overview

This project implements and compares two reinforcement learning agents — **DQN** and **DRQN** — in a partially observable grid-world environment (POMDP). The goal of the project is to demonstrate how internal memory (via LSTM) improves agent performance under partial observability.

## Environment

- **Grid Size:** 5×5
- **Observation:** Agent sees a 3×3 local region around its position
- **Actions:** Up, Down, Left, Right
- **Rewards:**
  - +10 for reaching goal
  - -1 for each step taken

## Repository Contents

| File | Description |
|------|-------------|
| `env.py` | Custom grid-world environment |
| `dqn.py` | DQN network definition |
| `drqn.py` | DRQN (LSTM) network definition |
| `train_dqn.py` | DQN training script |
| `train_drqn.py` | DRQN training script |
| `compare_models.py` | Compare and plot reward curves |
| `analyze_hidden.py` | Visualize hidden-state dynamics |
| `requirements.txt` | Required Python packages |
| `.gitignore` | Files excluded from Git |
| `*.npy`, `*.pth` | Saved results & models |

## Installation

```bash
git clone https://github.com/AnupamMI/DRQN_POMDP_Project.git
cd DRQN_POMDP_Project
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
