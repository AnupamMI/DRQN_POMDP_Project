import numpy as np
from typing import List, Optional, Tuple


class GridWorld:
    """Small grid-world offering a 3x3 local observation around the agent.

    - Values in the full grid: 0 empty, -1 obstacle, 1 agent, 2 goal
    - `agent_pos` and `goal_pos` are (row, col) tuples to match the rest
      of the project (`play_animation.py` expects this).

    New features:
    - Optional per-episode random obstacles (controlled by `dynamic_obstacles`).
    - Obstacle density or explicit count.
    - Connectivity check: regenerates obstacle layout until a path exists.
    """

    def __init__(
        self,
        grid_size: int = 5,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        seed: Optional[int] = None,
        dynamic_obstacles: bool = False,
        obstacle_prob: float = 0.15,
        obstacle_count: Optional[int] = None,
        max_regen_attempts: int = 12,
    ):
        self.grid_size = grid_size
        self.goal_pos = (grid_size - 1, grid_size - 1)

        # default obstacles (kept only if provided explicitly)
        default_obs = [(2, 2), (3, 4), (5, 1), (6, 6)]
        self.obstacles = [] if obstacles is None else [
            o for o in obstacles if 0 <= o[0] < grid_size and 0 <= o[1] < grid_size and o != self.goal_pos
        ]

        # dynamic obstacle generation settings
        self.dynamic_obstacles = dynamic_obstacles
        self.obstacle_prob = float(obstacle_prob)
        self.obstacle_count = int(obstacle_count) if obstacle_count is not None else None
        self.max_regen_attempts = int(max_regen_attempts)

        # reproducible randomness when `seed` provided
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

        # agent / grid state
        self.agent_pos: Tuple[int, int] = (0, 0)

        # initialise
        self.reset()

    def reset(self) -> np.ndarray:
        """Place the agent at a random free cell (not obstacle or goal).

        Returns the local 3x3 observation (not flattened) to stay compatible
        with the project's expectation of a (3,3) array.
        """
        # if dynamic obstacles are enabled, generate a new obstacle layout
        if self.dynamic_obstacles:
            self._generate_obstacles()

        # pick a random valid start (not on obstacle or goal), retry until reachable
        attempts = 0
        while True:
            r = int(self.rng.randint(0, self.grid_size))
            c = int(self.rng.randint(0, self.grid_size))
            if (r, c) in self.obstacles or (r, c) == self.goal_pos:
                attempts += 1
                if attempts > 100:
                    # fallback to (0,0)
                    self.agent_pos = (0, 0)
                    break
                continue

            # quick connectivity check: ensure a path exists from start -> goal
            if self._is_reachable((r, c), self.goal_pos):
                self.agent_pos = (r, c)
                break
            attempts += 1
            if attempts >= 50:
                # if too many failures, accept placement anyway (avoid infinite loops)
                self.agent_pos = (r, c)
                break

        return self._get_obs()

    def _full_grid(self) -> np.ndarray:
        """Return the full grid (useful for rendering / debugging)."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for o in self.obstacles:
            grid[o] = -1
        grid[self.goal_pos] = 2
        grid[self.agent_pos] = 1
        return grid

    def _generate_obstacles(self) -> None:
        """Generate a random obstacle layout (updates self.obstacles).

        Respects `obstacle_prob` or `obstacle_count`. Ensures goal cell is free.
        Retries until start-goal connectivity exists (up to `max_regen_attempts`).
        """
        attempts = 0
        while True:
            attempts += 1
            obs = []
            if self.obstacle_count is not None:
                # sample unique obstacle positions
                all_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
                all_cells.remove(self.goal_pos)
                # sample without replacement
                k = min(self.obstacle_count, len(all_cells) - 1)
                if k > 0:
                    chosen = self.rng.choice(len(all_cells), size=k, replace=False)
                    obs = [all_cells[i] for i in chosen]
            else:
                # random by probability
                for r in range(self.grid_size):
                    for c in range(self.grid_size):
                        if (r, c) == self.goal_pos:
                            continue
                        if self.rng.rand() < self.obstacle_prob:
                            obs.append((r, c))

            # ensure goal not in obstacles
            obs = [o for o in obs if o != self.goal_pos]
            self.obstacles = obs

            # quick reachability test from a default start (0,0) or any free cell
            # pick a provisional start not in obstacles
            start = (0, 0)
            if start in self.obstacles or start == self.goal_pos:
                # find a different provisional start
                for r in range(self.grid_size):
                    for c in range(self.grid_size):
                        if (r, c) not in self.obstacles and (r, c) != self.goal_pos:
                            start = (r, c)
                            break
                    else:
                        continue
                    break

            if self._is_reachable(start, self.goal_pos):
                break
            if attempts >= self.max_regen_attempts:
                # accept current obstacle layout
                break

    def _is_reachable(self, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """Return True if goal reachable from start using 4-neighbour moves avoiding obstacles."""
        from collections import deque

        if start == goal:
            return True
        seen = set()
        q = deque([start])
        seen.add(start)
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                    continue
                if (nr, nc) in self.obstacles:
                    continue
                if (nr, nc) in seen:
                    continue
                if (nr, nc) == goal:
                    return True
                seen.add((nr, nc))
                q.append((nr, nc))
        return False

    def _get_obs(self) -> np.ndarray:
        """Return a 3x3 local observation centered on the agent (rows, cols)."""
        grid = self._full_grid()
        r, c = self.agent_pos
        padded = np.pad(grid, 2, mode="constant", constant_values=0)
        r_p, c_p = r + 2, c + 2
        obs = padded[r_p - 2 : r_p + 3, c_p - 2 : c_p + 3]
        return obs

    def get_flat_obs(self) -> np.ndarray:
        """Convenience: flattened 3x3 observation (length 9)."""
        return self._get_obs().flatten()

    def step(self, action: int) -> Tuple[np.ndarray, int, bool]:
        """Apply `action` and return (obs, reward, done).

        Actions: 0=up, 1=down, 2=left, 3=right. If the move would hit an
        obstacle the agent stays in place and receives a larger penalty.
        """
        r, c = self.agent_pos

        if action == 0:
            nr, nc = r - 1, c
        elif action == 1:
            nr, nc = r + 1, c
        elif action == 2:
            nr, nc = r, c - 1
        elif action == 3:
            nr, nc = r, c + 1
        else:
            nr, nc = r, c

        # clip to grid
        nr = int(np.clip(nr, 0, self.grid_size - 1))
        nc = int(np.clip(nc, 0, self.grid_size - 1))

        # collision with obstacle -> stay and stronger negative reward
        if (nr, nc) in self.obstacles:
            nr, nc = r, c
            reward = -2
        else:
            reward = -1

        self.agent_pos = (nr, nc)

        done = self.agent_pos == self.goal_pos
        if done:
            reward = 10

        return self._get_obs(), reward, bool(done)

    def render(self) -> None:
        """Pretty-print the full grid to the console."""
        grid = np.full((self.grid_size, self.grid_size), ".", dtype="U2")
        for (orow, ocol) in self.obstacles:
            grid[orow, ocol] = "X"
        gr, gc = self.goal_pos
        grid[gr, gc] = "G"
        ar, ac = self.agent_pos
        grid[ar, ac] = "A"

        for row in grid:
            print(" ".join(row))
        print()