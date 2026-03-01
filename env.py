import numpy as np

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]

        self.grid = np.zeros((self.size, self.size))
        self.grid[self.goal_pos[0], self.goal_pos[1]] = 2  # goal

        return self._get_obs()

    def step(self, action):
        # 0=up, 1=down, 2=left, 3=right
        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.size - 1:
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.size - 1:
            self.agent_pos[1] += 1

        done = self.agent_pos == self.goal_pos
        reward = 10 if done else -1

        return self._get_obs(), reward, done

    def _get_obs(self):
        obs = np.zeros((3, 3))

        x, y = self.agent_pos

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx = x + dx
                ny = y + dy

                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if [nx, ny] == self.goal_pos:
                        obs[dx + 1, dy + 1] = 2
                    elif [nx, ny] == self.agent_pos:
                        obs[dx + 1, dy + 1] = 1
                    else:
                        obs[dx + 1, dy + 1] = 0

        return obs

    def render(self):
        display = self._get_obs()
        print(display)