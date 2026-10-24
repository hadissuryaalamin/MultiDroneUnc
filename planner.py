from time import perf_counter

import numpy as np
from multi_drone import MultiDroneUnc

class Node:
    def __init__(self, state: np.ndarray, num_actions: int):
        self.state = state.copy()
        self.N = np.zeros(num_actions)
        self.Q = np.zeros(num_actions)
        self.children = {}  # (action, next_state.tobytes()) -> Node
        self.invalid_actions = np.zeros(num_actions, dtype=bool)
        self.fallback_action = None
        self.has_safe_action = False


class Planner:
    def __init__(self, env: MultiDroneUnc, a_param: float = 1.0, b_param: int = 1.0):
        self._env = env
        # The runner passes an environment that has already loaded the YAML config.
        self._config = env.get_config()
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
        max_depth = self._config.max_num_steps

        while perf_counter() < deadline:
            node = root
            path = []
            depth = 0
            reward_sum = 0.0

            while depth < max_depth and perf_counter() < deadline:
                # 1. Selection: sample a fresh outcome for the selected action.
                action = self._selection(node, deadline)
                if perf_counter() >= deadline:
                    break
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
            # Still choose a checked action if the search budget was too short.
            if root.fallback_action is not None:
                return root.fallback_action
            fallback_action, _ = self._greedy_action(current_state)
            return fallback_action
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
            sim_action, _ = self._greedy_action(sim_state)
            sim_state, reward, done, info = self._env.simulate(sim_state, sim_action)
            reward_sum += discount * reward
            discount *= self._config.discount_factor
            if done:
                break

        return reward_sum

    def _is_safe_action(self, state: np.ndarray, action: int) -> bool:
        """Check intended endpoints; actual stochastic moves may still collide."""
        num_actions = len(self._action_vectors)
        occupied = set()

        for drone_state in state:
            action_id = action % num_actions
            action //= num_actions
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

    def _greedy_action(self, state: np.ndarray) -> tuple[int, bool]:
        """Return a joint action and whether a safe intended assignment was found."""
        num_actions = len(self._action_vectors)
        choices = []
        fallback_action = 0

        for i, drone_state in enumerate(state):
            if drone_state[3]:
                choices.append([0])  # Reached drones stay at their goals.
                continue

            position = drone_state[:3]
            goal = self._goals[i]
            candidates = position + self._action_vectors

            # Retain in-bounds moves even if none avoids obstacles: the fallback
            # still prefers an in-bounds move when no safe assignment exists.
            in_bounds = np.all(
                (candidates >= 0) & (candidates < self._grid_size), axis=1
            )
            action_ids = np.flatnonzero(in_bounds)
            if not action_ids.size:
                choices.append([])
                continue

            distances = np.linalg.norm(candidates[action_ids] - goal, axis=1)
            order = np.argsort(distances, kind='stable')
            ordered_actions = [int(action_id) for action_id in action_ids[order]]
            choices.append(ordered_actions)

            # Check one drone with the same validator used by Selection.
            fallback_id = ordered_actions[0]
            for action_id in ordered_actions:
                if self._is_safe_action(state[i:i + 1], action_id):
                    fallback_id = action_id
                    break
            fallback_action += fallback_id * (num_actions ** i)

        def assign(drone_index, joint_action):
            if drone_index == len(state):
                return joint_action

            for action_id in choices[drone_index]:
                candidate_action = joint_action + action_id * (num_actions ** drone_index)
                # Validate the assigned prefix only; later drones have no action
                # yet. This checks obstacles, bounds and collisions in one place.
                if not self._is_safe_action(state[:drone_index + 1], candidate_action):
                    continue
                result = assign(drone_index + 1, candidate_action)
                if result is not None:
                    return result

            return None

        joint_action = assign(0, 0)
        if joint_action is None:
            # No stay action exists; some states have no safe intended move.
            return fallback_action, False
        return joint_action, True

    def _selection(self, node: Node, deadline: float = float('inf')) -> int:
        """Select a checked untried action, or use UCB among tried actions."""
        if node.fallback_action is None:
            node.fallback_action, node.has_safe_action = self._greedy_action(node.state)

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
                untried[index] = untried[-1]
                untried = untried[:-1]
            return node.fallback_action

        tried = np.flatnonzero((node.N > 0) & ~node.invalid_actions)
        if not tried.size:
            return node.fallback_action
        ucb = node.Q[tried] + np.sqrt(2 * np.log(node.N[tried].sum()) / node.N[tried])
        return int(tried[np.argmax(ucb)])
