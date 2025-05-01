import numpy as np

class FrustrationAgent:
    def __init__(self, env, patience=20):
        self.env = env
        self.patience = patience
        self.reset()

    def reset(self):
        self.prev_distance = None
        self.frustration = 0
        self.path_history = set()

    def manhattan(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def choose_action(self, obs, agent_pos, goal_pos):
        # Compute Manhattan distance
        current_distance = self.manhattan(agent_pos, goal_pos)

        # Initialize distance tracker
        if self.prev_distance is None:
            self.prev_distance = current_distance

        # Compare with previous step
        if current_distance >= self.prev_distance:
            self.frustration += 1
        else:
            self.frustration = 0  # Reset if progress made

        self.prev_distance = current_distance

        # If frustration exceeds patience, opt out
        if self.frustration >= self.patience:
            return 4  # opt-out action

        # Otherwise: move greedily toward goal
        x, y = agent_pos
        gx, gy = goal_pos

        moves = []
        if gx < x: moves.append(0)  # up
        if gx > x: moves.append(1)  # down
        if gy < y: moves.append(2)  # left
        if gy > y: moves.append(3)  # right

        if not moves:
            return np.random.randint(0, 4)

        return np.random.choice(moves)
