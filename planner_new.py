from time import perf_counter

import numpy as np
from multi_drone import MultiDroneUnc

class Node:
    def __init__(self, state: np.ndarray, num_actions: int):
        self.state = state.copy()
        self.N = np.zeros(num_actions)
        self.Q = np.zeros(num_actions)
        self.children = [None] * num_actions


class Planner:
    def __init__(self, env: MultiDroneUnc, a_param: float = 1.0, b_param: int = 1.0):
        self._env = env
        # The runner passes an environment that has already loaded the YAML config.
        self._config = env.get_config()
        self._a_param = a_param
        self._b_param = b_param

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        deadline = perf_counter() + planning_time_per_step
        root = Node(current_state, self._env.num_actions)

        while perf_counter() < deadline:
            # 1. Selection
            action = self._selection(root)
            next_state, reward, done, info = self._env.simulate(root.state, action)
            # 2. Expansion
            child = Node(next_state, self._env.num_actions)
            root.children[action] = child
            # 3. Simulation from child node
            reward_sum = reward  # Include the reward from root to child.
            discount = self._config.discount_factor
            sim_state = next_state.copy()
            for _ in range(self._config.max_num_steps):  # Simulate for a fixed number of steps
                if perf_counter() >= deadline:
                    break
                sim_action = np.random.choice(self._env.num_actions) # random or we can use a policy based on sim_state eg heuristic distance to goal
                sim_state, rollout_reward, done, info = self._env.simulate(sim_state, sim_action)
                reward_sum += discount * rollout_reward
                discount *= self._config.discount_factor
                if done:
                    break
            # 4. Backpropagation
            root.N[action] += 1
            root.Q[action] += reward_sum

        best_action = int(np.argmax(root.Q))
        return best_action

    
    def _selection(self, node: Node) -> int:
        """Select the action with the highest UCB value."""
        untried = np.flatnonzero(node.N == 0)
        if untried.size:
            return np.random.choice(untried)
        ucb = node.Q + np.sqrt(2 * np.log(node.N.sum()) / node.N)
        return int(np.argmax(ucb))
