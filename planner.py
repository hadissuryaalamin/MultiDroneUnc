from collections import deque
from time import perf_counter

import numpy as np
from multi_drone import MultiDroneUnc

class Node:
    def __init__(self, state: np.ndarray, num_actions: int):
        self.state = state.copy()
        self.N = np.zeros(num_actions)
        self.Q = np.zeros(num_actions)
        self.children = {}  # (action, next_state.tobytes()) -> Node


class Planner:
    def __init__(self, env: MultiDroneUnc, a_param: float = 1.0, b_param: int = 1.0):
        self._env = env
        # The runner passes an environment that has already loaded the YAML config.
        self._config = env.get_config()
        self._rng = np.random.default_rng(self._config.seed)
        self._a_param = a_param
        self._b_param = b_param

        # Match the simulator's per-drone action ordering using config only.
        dzs = (-1, 0, 1) if self._config.change_altitude else (0,)
        self._moves = np.array([
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in dzs
            if (dx, dy, dz) != (0, 0, 0)
        ], dtype=np.int32)
        self._grid_size = np.array(self._config.grid_size)
        self._obstacles = np.zeros(self._config.grid_size, dtype=bool)
        for cell in self._config.obstacle_cells or []:
            self._obstacles[tuple(cell)] = True
        self._goal_distances = [
            self._build_goal_distances(tuple(goal))
            for goal in self._config.goal_positions
        ]

    def _build_goal_distances(self, goal: tuple[int, int, int]) -> np.ndarray:
        """BFS distances assuming intended moves always succeed, ignoring other drones."""
        distances = np.full(self._config.grid_size, np.inf, dtype=np.float32)
        if self._obstacles[goal]:
            return distances
        distances[goal] = 0
        queue = deque([goal])

        while queue:
            position = queue.popleft()
            neighbors = np.array(position) + self._moves
            in_bounds = np.all((neighbors >= 0) & (neighbors < self._grid_size), axis=1)
            for neighbor in neighbors[in_bounds]:
                cell = tuple(neighbor)
                if self._obstacles[cell] or np.isfinite(distances[cell]):
                    continue
                distances[cell] = distances[position] + 1
                queue.append(cell)

        return distances

    def _deterministic_action(self, state: np.ndarray) -> int:
        """Choose one shortest-path move per drone and encode their joint action."""
        num_moves = len(self._moves)
        joint_action = 0
        for i, drone in enumerate(state):
            action = 0  # Reached drones stay at their goal regardless of action.
            if not drone[3]:
                destinations = drone[:3] + self._moves
                valid = np.all(
                    (destinations >= 0) & (destinations < self._grid_size), axis=1
                )
                valid[valid] = ~self._obstacles[tuple(destinations[valid].T)]
                candidates = np.flatnonzero(valid)
                if candidates.size:
                    distances = self._goal_distances[i][tuple(destinations[candidates].T)]
                    # If the goal is unreachable, all free moves tie at infinity.
                    best = candidates[distances == distances.min()]
                    action = int(self._rng.choice(best))
                else:
                    # No safe intended move exists; there is no stay action.
                    action = int(self._rng.integers(num_moves))
            joint_action += action * num_moves ** i

        return joint_action

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        deadline = perf_counter() + planning_time_per_step
        root = Node(current_state, self._env.num_actions)
        gamma = self._config.discount_factor
        max_depth = self._config.max_num_steps

        while perf_counter() < deadline:
            node = root
            path = []
            depth = 0
            reward_sum = 0.0

            while depth < max_depth and perf_counter() < deadline:
                # 1. Selection: sample a fresh outcome for the selected action.
                action = self._selection(node)
                next_state, reward, done, info = self._env.simulate(node.state, action)
                path.append((node, action, reward))
                depth += 1

                # 2. Expansion: preserve each distinct outcome of this action.
                key = (action, next_state.tobytes())
                child = node.children.get(key)
                is_new = child is None
                if is_new:
                    child = Node(next_state, self._env.num_actions)
                    node.children[key] = child

                if done:
                    break

                if is_new:
                    # 3. Rollout from the new child, within the remaining depth.
                    reward_sum = self._rollout(child.state, max_depth - depth, deadline)
                    break

                # The outcome already exists: continue selection from its node.
                node = child

            # 4. Backpropagation: update every selected action along the path.
            for node, action, reward in reversed(path):
                reward_sum = reward + gamma * reward_sum
                node.N[action] += 1
                node.Q[action] += (reward_sum - node.Q[action]) / node.N[action]

        tried = root.N > 0
        if not tried.any():
            return 0  # No simulations completed within the time budget.
        scores = np.where(tried, root.Q, -np.inf)
        best_action = int(np.argmax(scores))
        return best_action

    def _rollout(self, state: np.ndarray, depth: int, deadline: float) -> float:
        """Estimate a state's value following deterministic shortest paths."""
        sim_state = state.copy()
        reward_sum = 0.0
        discount = 1.0

        for _ in range(depth):
            if perf_counter() >= deadline:
                break
            sim_action = self._deterministic_action(sim_state)
            sim_state, reward, done, info = self._env.simulate(sim_state, sim_action)
            reward_sum += discount * reward
            discount *= self._config.discount_factor
            if done:
                break

        return reward_sum


    def _selection(self, node: Node) -> int:
        """Select the action with the highest UCB value."""
        untried = np.flatnonzero(node.N == 0)
        if untried.size:
            return int(self._rng.choice(untried))
        ucb = node.Q + np.sqrt(2 * np.log(node.N.sum()) / node.N)
        return int(np.argmax(ucb))
