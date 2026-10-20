"""A version of planner.py with more explicit MCTS steps.

To use this version in run_planner.py or run_u8050066.py, import:
    from planner_sederhana import Planner

The state has one row per drone: [x, y, z, reached_goal].
One action is an integer representing the moves of ALL drones at once.
"""

from math import log, sqrt
from time import perf_counter

import numpy as np

from multi_drone import MultiDroneUnc


class Node:
    """One state and the results of trying actions from that state."""

    def __init__(self, state: np.ndarray):
        self.state = state.copy()
        self.visits = 0
        self.action_visits = {}  # Number of times each action has been tried.
        self.action_values = {}  # Average return for each action.
        self.children = {}       # Nodes reached by simulating each action.


class Planner:
    def __init__(self, env: MultiDroneUnc, a_param: float = 1.0, b_param: int = 1.0):
        # Accept a_param and b_param for compatibility with existing runners.
        # As in planner.py, neither parameter is used in the MCTS calculations.
        self._env = env
        config = env.get_config()
        self._discount_factor = config.discount_factor
        self._exploration_weight = sqrt(2.0)
        self._max_depth = config.max_num_steps
        self._deadline = 0.0
        self._root = None

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        """Run simulations, then choose the action with the highest average return."""
        self._root = Node(current_state)
        self._deadline = perf_counter() + planning_time_per_step

        while perf_counter() < self._deadline:
            self._search(self._root, self._max_depth)

        # Use a greedy action if time runs out before any result is recorded.
        if not self._root.action_values:
            return self._greedy_joint_action(current_state)

        best_action = None
        best_value = float("-inf")
        for action, average_return in self._root.action_values.items():
            if average_return > best_value:
                best_action = action
                best_value = average_return

        return best_action

    def _search(self, node: Node, remaining_depth: int) -> float:
        """One MCTS iteration: selection, expansion, rollout, backpropagation."""
        if remaining_depth <= 0 or perf_counter() >= self._deadline:
            return 0.0

        # 1. Selection: choose an action and simulate one step.
        action = self._select_action(node)
        next_state, reward, done, _ = self._env.simulate(node.state, action)

        # A terminal result means all drones reached their goals OR a collision occurred.
        # There are no future rewards, but this step's reward must still be recorded.
        future_return = 0.0
        if not done:
            # Actions are stochastic: one action can lead to different states.
            child_key = (action, next_state.tobytes())
            child = node.children.get(child_key)

            if child is None:
                # 2. Expansion: store a newly discovered state.
                child = self._expand(node, action, next_state)
                # 3. Rollout: estimate future rewards using greedy moves.
                future_return = self._rollout(child.state, remaining_depth - 1)
            else:
                # This state already exists: continue searching inside the tree.
                future_return = self._search(child, remaining_depth - 1)

        # Return = immediate reward + discounted future return.
        total_return = reward + self._discount_factor * future_return

        # 4. Backpropagation: always record the result, including when done=True.
        self._backpropagate(node, action, total_return)
        return total_return

    def _select_action(self, node: Node) -> int:
        """Try unvisited actions first, then use the upper confidence bound (UCB)."""
        untried_actions = []
        for action in range(self._env.num_actions):
            if node.action_visits.get(action, 0) == 0:
                untried_actions.append(action)

        if untried_actions:
            index = self._env.rng.integers(len(untried_actions))
            return untried_actions[index]

        best_action = None
        best_score = float("-inf")
        for action in range(self._env.num_actions):
            average_return = node.action_values[action]
            action_visits = node.action_visits[action]

            # Give a larger bonus to actions that have been tried less often.
            exploration_bonus = self._exploration_weight * sqrt(
                log(node.visits) / action_visits
            )
            score = average_return + exploration_bonus

            if score > best_score:
                best_action = action
                best_score = score

        return best_action

    def _expand(self, node: Node, action: int, next_state: np.ndarray) -> Node:
        """Add a child node for an action and its resulting state."""
        child = Node(next_state)
        child_key = (action, next_state.tobytes())
        node.children[child_key] = child
        return child

    def _rollout(self, state: np.ndarray, remaining_depth: int) -> float:
        """Simulate future steps without adding nodes to the search tree."""
        total_return = 0.0
        discount = 1.0

        for _ in range(remaining_depth):
            if perf_counter() >= self._deadline:
                break

            action = self._greedy_joint_action(state)
            state, reward, done, _ = self._env.simulate(state, action)
            total_return += discount * reward

            if done:
                break

            discount *= self._discount_factor

        return total_return

    def _greedy_joint_action(self, state: np.ndarray) -> int:
        """For each drone, choose a valid move closest to its goal.

        This is a rollout heuristic: it does not guarantee collision avoidance
        between drones, and actual moves remain subject to uncertainty.
        """
        env = self._env
        grid_size = np.array(env.obstacles.shape)
        drone_actions = np.zeros(len(state), dtype=np.int32)

        for drone_index, drone in enumerate(state):
            position = drone[:3]
            already_reached_goal = bool(drone[3])
            if already_reached_goal:
                # The simulator keeps this drone at its goal regardless of its action.
                continue

            goal = env.goals[drone_index]
            best_distance = float("inf")
            best_actions = []

            for action, movement in enumerate(env._action_vectors):
                destination = position + movement

                if np.any(destination < 0) or np.any(destination >= grid_size):
                    continue
                if env.obstacles[tuple(destination)]:
                    continue

                # Squared distance is sufficient to compare proximity to the goal.
                distance = np.sum((destination - goal) ** 2)
                if distance < best_distance:
                    best_distance = distance
                    best_actions = [action]
                elif distance == best_distance:
                    best_actions.append(action)

            if best_actions:
                index = env.rng.integers(len(best_actions))
                drone_actions[drone_index] = best_actions[index]
            else:
                # As in planner.py, try a random action if no valid move is available.
                drone_actions[drone_index] = env.rng.integers(
                    len(env._action_vectors)
                )

        return env._encode_action(drone_actions)

    def _backpropagate(self, node: Node, action: int, total_return: float) -> None:
        """Update the visit count and average return for the chosen action."""
        previous_visits = node.action_visits.get(action, 0)
        previous_average = node.action_values.get(action, 0.0)
        new_visits = previous_visits + 1

        # New average = (previous total return + new return) / number of visits.
        previous_total = previous_average * previous_visits
        new_average = (previous_total + total_return) / new_visits

        node.visits += 1
        node.action_visits[action] = new_visits
        node.action_values[action] = new_average
