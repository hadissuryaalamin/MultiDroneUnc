from math import log, sqrt
from time import perf_counter

import numpy as np
from multi_drone import MultiDroneUnc


class Node:
    def __init__(self, state: np.ndarray, num_actions: int):
        self.state = state.copy()
        self.N = np.zeros(num_actions)
        self.Q = np.zeros(num_actions)
        self.children = {}


class Planner:
    def __init__(self, env: MultiDroneUnc, a_param: float = 1.0, b_param: int = 1.0):
        # a_param and b_param are only kept for compatibility with run_planner.py.
        self.env = env
        config = env.get_config()
        self.gamma = config.discount_factor
        self.max_depth = config.max_num_steps
        self.c = sqrt(2.0)

        # Information for the greedy rollout, taken from the config only (README rule:
        # the planner may only use simulate, get_config and num_actions).
        self.rng = np.random.default_rng(config.seed)
        dzs = (-1, 0, 1) if config.change_altitude else (0,)
        self.moves = np.array([
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in dzs
            if (dx, dy, dz) != (0, 0, 0)
        ])  # same order as the simulator's per-drone actions
        self.goals = np.array(config.goal_positions)
        self.grid_size = np.array(config.grid_size)
        self.obstacles = np.zeros(config.grid_size, dtype=bool)
        for cell in config.obstacle_cells or []:
            self.obstacles[tuple(cell)] = True

    def plan(self, state: np.ndarray, planning_time_per_step: float) -> int:
        num_actions = self.env.num_actions
        root = Node(state, num_actions)
        self.deadline = perf_counter() + planning_time_per_step

        while perf_counter() < self.deadline:
            # One MCTS simulation. path records the visited (node, action, reward) triples.
            path = []
            node, depth, G = root, self.max_depth, 0.0

            while depth > 0 and perf_counter() < self.deadline:
                # 1. Selection
                action = self._selection(node)
                next_state, reward, done, _ = self.env.simulate(node.state, action)
                path.append((node, action, reward))
                depth -= 1
                if done:
                    break

                key = (action, next_state.tobytes())
                child = node.children.get(key)
                if child is None:
                    # 2. Expansion: add a new node, then 3. Rollout from that node.
                    node.children[key] = Node(next_state, num_actions)
                    G = self._rollout(next_state, depth)
                    break
                node = child

            # 4. Backpropagation: from bottom to top, G = reward + gamma * G.
            for node, action, reward in reversed(path):
                G = reward + self.gamma * G
                node.N[action] += 1
                node.Q[action] += (G - node.Q[action]) / node.N[action]

        if not root.N.any():
            return self._greedy(state)
        # Choose the action with the highest mean return among the tried actions.
        tried = root.N > 0                              # actions that have been tried
        scores = np.where(tried, root.Q, -np.inf)       # untried actions get -inf
        best_action = np.argmax(scores)                 # index with the highest Q
        return int(best_action)

    def _selection(self, node: Node) -> int:
        """Try untried actions first, then use UCB."""
        untried = np.flatnonzero(node.N == 0)
        if untried.size:
            return int(self.rng.choice(untried))
        ucb = node.Q + self.c * np.sqrt(log(node.N.sum()) / node.N)
        return int(np.argmax(ucb))

    def _rollout(self, state: np.ndarray, depth: int) -> float:
        """Simulation: run the greedy policy and sum the discounted rewards."""
        G, discount = 0.0, 1.0
        for _ in range(depth):
            if perf_counter() >= self.deadline:
                break
            state, reward, done, _ = self.env.simulate(state, self._greedy(state))
            G += discount * reward
            if done:
                break
            discount *= self.gamma
        return G

    def _greedy(self, state: np.ndarray) -> int:
        """Each drone picks the valid move closest to its goal."""
        m = len(self.moves)
        joint_action = 0

        for i, drone in enumerate(state):
            action = 0  # drone already at its goal: the simulator ignores its action
            if not drone[3]:
                dest = drone[:3] + self.moves
                valid = np.all((dest >= 0) & (dest < self.grid_size), axis=1)
                valid[valid] = ~self.obstacles[tuple(dest[valid].T)]
                if valid.any():
                    dist = np.where(valid, ((dest - self.goals[i]) ** 2).sum(axis=1), np.inf)
                    action = self.rng.choice(np.flatnonzero(dist == dist.min()))
                else:
                    action = self.rng.integers(m)
            # Encode the joint action: sum a_i * m^i.
            joint_action += int(action) * m ** i

        return joint_action
