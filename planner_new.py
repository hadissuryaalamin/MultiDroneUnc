import numpy as np
from multi_drone import MultiDroneUnc

NUM_SIM_STEPS = 100

class Node:
    def __init__(self, state: np.ndarray, num_moves: int):
        self.state = state.copy()
        self.N = np.zeros(num_moves)
        self.Q = np.zeros(num_moves)
        self.children = [None] * num_moves


class Planner:
    def __init__(self, env: MultiDroneUnc, a_param: float = 1.0, b_param: int = 1.0):
        self._env = env
        self._a_param = a_param
        self._b_param = b_param

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        # This doesn't do anything useful. It simply returns the action 
        # representen by integer 0.
        hist = []
        # Inside Loop
        # 1. Selection
        root = Node(current_state, self._env.num_moves)
        action = self._selection(root)
        next_state, reward, done, info = self._env.step(action)
        if (current_state, action, next_state) not in hist:
            hist.append((current_state, action, next_state))
            # 2. Expansion
            child = Node(next_state, self._env.num_moves)
            root.children[action] = child
            # 3. Simulation from child node
            reward_sum = 0
            sim_state = next_state.copy()
            for _ in range(NUM_SIM_STEPS):  # Simulate for a fixed number of steps
                sim_action = np.random.choice(self._env.num_moves) # random or we can use a policy based on sim_state eg heuristic distance to goal
                sim_state, reward, done, info = self._env.simulate(sim_state, sim_action)
                reward_sum += reward
                if done:
                    break
            # 4. Backpropagation
            root.N[action] += 1
            root.Q[action] += reward_sum - STEP_COST # this from config
            best_action = int(np.argmax(root.Q))
            return best_action

    
    def _selection(self, node: Node) -> int:
        """Select the action with the highest UCB value."""
        untried = np.flatnonzero(node.N == 0)
        if untried.size:
            return np.random.choice(untried)
        ucb = node.Q + np.sqrt(2 * np.log(node.N.sum()) / node.N)
        return int(np.argmax(ucb))
