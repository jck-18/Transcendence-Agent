import numpy as np
import gym
from gym import spaces

class TranscendenceGridEnv(gym.Env):
    def __init__(self, grid_size=10, max_steps=200):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(5)  # 0=up, 1=down, 2=left, 3=right, 4=opt-out
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.uint8)

        self.start = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)
        self.agent_pos = self.start
        self.steps = 0

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self._generate_walls()
        self.agent_pos = self.start
        self.steps = 0
        return self._get_obs()

    def _generate_walls(self):
        # 30% chance of placing a wall in each cell, except start/goal
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) != self.start and (i, j) != self.goal:
                    self.grid[i][j] = 1 if np.random.rand() < 0.3 else 0

    def _get_obs(self):
        obs = np.copy(self.grid)
        obs[self.agent_pos] = 9  # mark agent
        return obs

    def _is_valid(self, pos):
        x, y = pos
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return self.grid[x][y] == 0
        return False

    def step(self, action):
        self.steps += 1
        reward = -0.01
        done = False

        if action == 4:  # opt-out
            if not self._is_reachable():
                reward = 1.0  # transcendence bonus
            else:
                reward = -1.0  # gave up too early
            done = True
            return self._get_obs(), reward, done, {}

        # move logic
        x, y = self.agent_pos
        if action == 0 and self._is_valid((x - 1, y)):
            x -= 1
        elif action == 1 and self._is_valid((x + 1, y)):
            x += 1
        elif action == 2 and self._is_valid((x, y - 1)):
            y -= 1
        elif action == 3 and self._is_valid((x, y + 1)):
            y += 1

        self.agent_pos = (x, y)

        if self.agent_pos == self.goal:
            reward = 1.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def _is_reachable(self):
        # Breadth-first search from agent to goal
        from collections import deque
        visited = set()
        queue = deque([self.agent_pos])

        while queue:
            pos = queue.popleft()
            if pos == self.goal:
                return True
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                new_pos = (pos[0]+dx, pos[1]+dy)
                if self._is_valid(new_pos) and new_pos not in visited:
                    visited.add(new_pos)
                    queue.append(new_pos)
        return False
