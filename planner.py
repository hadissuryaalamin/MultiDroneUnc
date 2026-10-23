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
    def __init__(self, env: MultiDroneUnc, a_param: float = 1.0, b_param: int = 1.0,
                 max_depth: int | None = None):
        self._env = env
        # The runner passes an environment that has already loaded the YAML config.
        self._config = env.get_config()
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be positive")
        self._max_depth = self._config.max_num_steps if max_depth is None else max_depth
        self._rng = np.random.default_rng(self._config.seed)
        self._a_param = a_param
        self._b_param = b_param
        # Match the simulator's per-drone action order for joint-action encoding.
        dz_values = (-1, 0, 1) if self._config.change_altitude else (0,)
        vectors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in dz_values:
                    if (dx, dy, dz) != (0, 0, 0):
                        vectors.append([dx, dy, dz])

        self._action_vectors = np.array(vectors, dtype=np.int32)
        self._goals = np.asarray(self._config.goal_positions, dtype=float)
        self._grid_size = np.asarray(self._config.grid_size)
        self._obstacles = np.zeros(self._config.grid_size, dtype=bool)
        for cell in self._config.obstacle_cells or []:
            self._obstacles[tuple(cell)] = True

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        deadline = perf_counter() + planning_time_per_step
        root = Node(current_state, self._env.num_actions)
        gamma = self._config.discount_factor
        max_depth = self._max_depth

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
        best_q = np.max(root.Q[tried])
        best_actions = np.flatnonzero(tried & (root.Q == best_q))
        # Break equal-Q ties randomly instead of favoring the lowest action ID.
        best_action = int(self._rng.choice(best_actions))
        return best_action

    def _rollout(self, state: np.ndarray, depth: int, deadline: float) -> float:
        """Estimate a state's value with a distance-greedy rollout policy."""
        sim_state = state.copy()
        reward_sum = 0.0
        discount = 1.0

        for _ in range(depth):
            if perf_counter() >= deadline:
                break
            sim_action = self._greedy_action(sim_state)
            sim_state, reward, done, info = self._env.simulate(sim_state, sim_action)
            reward_sum += discount * reward
            discount *= self._config.discount_factor
            if done:
                break

        return reward_sum

    def _greedy_action(self, state: np.ndarray) -> int:
        """Choose each drone's closest free neighbor to its goal, then encode."""
        joint_action = 0
        num_actions = len(self._action_vectors)

        for i, drone_state in enumerate(state):
            if drone_state[3]:  # Simulator locks drones that reached their goal.
                continue

            position = drone_state[:3]
            goal = self._goals[i]

            # Calculate candidate positions without calling simulate().
            candidates = position + self._action_vectors

            # Keep actions that stay inside the grid.
            in_bounds = np.all(
                (candidates >= 0) & (candidates < self._grid_size), axis=1
            )
            action_ids = np.flatnonzero(in_bounds)

            # Avoid obstacles if a free neighbor exists. There is no stay action.
            cells = candidates[action_ids]
            free = ~self._obstacles[cells[:, 0], cells[:, 1], cells[:, 2]]
            if free.any():
                action_ids = action_ids[free]

            # Choose the candidate with the smallest Euclidean distance to goal.
            distances = np.linalg.norm(candidates[action_ids] - goal, axis=1)
            best_action = int(action_ids[np.argmin(distances)])

            # Encode this drone's choice in the joint action.
            joint_action += best_action * (num_actions ** i)

        return joint_action


    def _selection(self, node: Node) -> int:
        """Select the action with the highest UCB value."""
        untried = np.flatnonzero(node.N == 0)
        if untried.size:
            return int(self._rng.choice(untried))
        ucb = node.Q + np.sqrt(2 * np.log(node.N.sum()) / node.N)
        return int(np.argmax(ucb))
