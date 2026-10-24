from time import perf_counter

import numpy as np
from multi_drone import MultiDroneUnc


class Node:
    """Search-tree node with per-action visit counts (N) and mean returns (Q)."""

    def __init__(self, state: np.ndarray, num_actions: int):
        self.state = state.copy()
        self.N = np.zeros(num_actions)
        self.Q = np.zeros(num_actions)
        self.children = {}  # (action, next_state.tobytes()) -> Node
        self.invalid_actions = np.zeros(num_actions, dtype=bool)
        self.fallback_action = None
        self.has_safe_action = False


class Planner:
    def __init__(self, env: MultiDroneUnc, a_param: float = 1.0, b_param: float = 1.0):
        self._env = env
        # The runner passes an environment that has already loaded the YAML config.
        self._config = env.get_config()
        self._rng = np.random.default_rng(self._config.seed)
        self._a_param = a_param
        self._b_param = b_param

        # Match the simulator's per-drone action order for joint-action encoding.
        dz_values = (-1, 0, 1) if self._config.change_altitude else (0,)
        self._action_vectors = np.array(
            [
                (dx, dy, dz)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in dz_values
                if (dx, dy, dz) != (0, 0, 0)
            ],
            dtype=np.int32,
        )
        self._num_drone_actions = len(self._action_vectors)

        self._goals = np.asarray(self._config.goal_positions, dtype=float)
        self._grid_size = np.asarray(self._config.grid_size)
        self._obstacles = np.zeros(self._config.grid_size, dtype=bool)
        for cell in self._config.obstacle_cells or []:
            self._obstacles[tuple(cell)] = True

    # ------------------------------------------------------------------ MCTS

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        deadline = perf_counter() + planning_time_per_step
        root = Node(current_state, self._env.num_actions)

        while perf_counter() < deadline:
            path, value = self._simulate(root, deadline)
            self._backpropagate(path, value)

        return self._best_action(root)

    def _simulate(self, root: Node, deadline: float):
        """Walk down the tree once; return the visited (node, action, reward) path
        and the estimated value after its last step."""
        max_depth = self._config.max_num_steps
        node = root
        path = []

        while len(path) < max_depth and perf_counter() < deadline:
            # 1. Selection: sample a fresh outcome for the selected action.
            action = self._selection(node, deadline)
            if perf_counter() >= deadline:
                break
            next_state, reward, done, _ = self._env.simulate(node.state, action)
            path.append((node, action, reward))

            # 2. Expansion: keep each distinct outcome of this action as a child.
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
                return path, self._rollout(child.state, max_depth - len(path), deadline)

            # The outcome already exists: continue selection from its node.
            node = child

        return path, 0.0

    def _backpropagate(self, path, value: float) -> None:
        """4. Backpropagation: update every selected action along the path."""
        gamma = self._config.discount_factor
        for node, action, reward in reversed(path):
            value = reward + gamma * value
            node.N[action] += 1
            node.Q[action] += (value - node.Q[action]) / node.N[action]

    def _best_action(self, root: Node) -> int:
        tried = root.N > 0
        if not tried.any():
            # Still choose a checked action if the search budget was too short.
            if root.fallback_action is not None:
                return root.fallback_action
            return self._greedy_action(root.state)
        best_q = np.max(root.Q[tried])
        best_actions = np.flatnonzero(tried & (root.Q == best_q))
        # Break equal-Q ties randomly instead of favoring the lowest action ID.
        return int(self._rng.choice(best_actions))

    def _selection(self, node: Node, deadline: float = float('inf')) -> int:
        """Select a checked untried action, or use UCB among tried actions."""
        if node.fallback_action is None:
            node.fallback_action = self._greedy_action(node.state)
            node.has_safe_action = self._is_safe_action(node.state, node.fallback_action)

        if not node.has_safe_action:
            return node.fallback_action

        untried = np.flatnonzero((node.N == 0) & ~node.invalid_actions)
        if untried.size:
            # Bound rejection sampling so filtering cannot consume the whole budget.
            for _ in range(min(64, len(untried))):
                if perf_counter() >= deadline:
                    return node.fallback_action
                index = int(self._rng.integers(len(untried)))
                action = int(untried[index])
                if self._is_safe_action(node.state, action):
                    return action
                node.invalid_actions[action] = True
                # Remove the unsafe action by swapping in the last one.
                untried[index] = untried[-1]
                untried = untried[:-1]
            return node.fallback_action

        tried = np.flatnonzero((node.N > 0) & ~node.invalid_actions)
        if not tried.size:
            return node.fallback_action
        ucb = node.Q[tried] + np.sqrt(2 * np.log(node.N[tried].sum()) / node.N[tried])
        return int(tried[np.argmax(ucb)])

    def _rollout(self, state: np.ndarray, depth: int, deadline: float) -> float:
        """Estimate a state's value with a distance-greedy rollout policy."""
        state = state.copy()  # Never let the rollout touch the tree node's state.
        reward_sum = 0.0
        discount = 1.0

        for _ in range(depth):
            if perf_counter() >= deadline:
                break
            action = self._greedy_action(state)
            state, reward, done, _ = self._env.simulate(state, action)
            reward_sum += discount * reward
            discount *= self._config.discount_factor
            if done:
                break

        return reward_sum

    # ------------------------------------------------- joint-action helpers

    def _encode(self, drone_actions) -> int:
        """Combine per-drone action IDs into one joint action (drone 0 = lowest digit)."""
        return sum(int(a) * self._num_drone_actions ** i for i, a in enumerate(drone_actions))

    def _decode(self, action: int, num_drones: int) -> list:
        """Split a joint action into per-drone action IDs."""
        drone_actions = []
        for _ in range(num_drones):
            drone_actions.append(action % self._num_drone_actions)
            action //= self._num_drone_actions
        return drone_actions

    def _is_safe_action(self, state: np.ndarray, action: int) -> bool:
        """Check intended endpoints; actual stochastic moves may still collide."""
        occupied = set()

        for drone_state, action_id in zip(state, self._decode(action, len(state))):
            position = drone_state[:3]
            if drone_state[3]:
                target = position  # Reached drones stay solid at their goals.
            else:
                target = position + self._action_vectors[action_id]

            if np.any(target < 0) or np.any(target >= self._grid_size):
                return False
            cell = tuple(target)
            if self._obstacles[cell] or cell in occupied:
                return False
            occupied.add(cell)

        return True

    def _greedy_action(self, state: np.ndarray) -> int:
        """Move each drone toward its goal while keeping intended endpoints distinct.

        Each drone ranks its in-bounds, obstacle-free moves by distance to its goal.
        A backtracking search then picks the first combination with no shared target
        cell. If none exists, every drone takes its closest move regardless of conflicts.
        """
        choices = []       # per drone: ranked list of (action_id, target_cell)
        fallback_ids = []  # per drone: closest move, ignoring other drones

        for i, drone_state in enumerate(state):
            position = drone_state[:3]
            if drone_state[3]:
                # Reached drones stay at their goal and occupy that cell.
                choices.append([(0, tuple(position))])
                fallback_ids.append(0)
                continue

            candidates = position + self._action_vectors
            in_bounds = np.all((candidates >= 0) & (candidates < self._grid_size), axis=1)
            action_ids = np.flatnonzero(in_bounds)
            if not action_ids.size:
                choices.append([])
                fallback_ids.append(0)
                continue

            cells = candidates[action_ids]
            free = ~self._obstacles[cells[:, 0], cells[:, 1], cells[:, 2]]
            has_free_move = free.any()
            if has_free_move:
                action_ids = action_ids[free]

            # Closest candidate first; stable sort keeps action order on ties.
            distances = np.linalg.norm(candidates[action_ids] - self._goals[i], axis=1)
            ranked = action_ids[np.argsort(distances, kind='stable')]
            fallback_ids.append(int(ranked[0]))

            if has_free_move:
                choices.append([(int(a), tuple(candidates[a])) for a in ranked])
            else:
                choices.append([])  # Every in-bounds move hits an obstacle.

        drone_actions = self._assign_distinct_cells(choices)
        if drone_actions is None:
            # No stay action exists; some states have no safe intended move.
            return self._encode(fallback_ids)
        return self._encode(drone_actions)

    @staticmethod
    def _assign_distinct_cells(choices):
        """Depth-first search for the first per-drone choice with no shared cells."""
        occupied = set()

        def assign(drone_index):
            if drone_index == len(choices):
                return []
            for action_id, cell in choices[drone_index]:
                if cell in occupied:
                    continue
                occupied.add(cell)
                rest = assign(drone_index + 1)
                occupied.remove(cell)
                if rest is not None:
                    return [action_id] + rest
            return None

        return assign(0)
