"""New planner, built step by step. Currently implements selection only."""

import numpy as np

from multi_drone import MultiDroneUnc


class Node:
    def __init__(self, state: np.ndarray, num_actions_per_drone: int):
        self.state = state.copy()
        # Each drone has its own action statistics, not a joint-action table.
        shape = (len(state), num_actions_per_drone)
        self.N = np.zeros(shape, dtype=np.int64)
        self.Q = np.zeros(shape, dtype=float)


class Planner:
    def __init__(self, env: MultiDroneUnc):
        self._env = env
        self._config = env.get_config()
        self._rng = np.random.default_rng(self._config.seed)

        # Keep the same per-drone action order as the simulator.
        dz_values = (-1, 0, 1) if self._config.change_altitude else (0,)
        self._action_vectors = np.array(
            [[dx, dy, dz]
             for dx in (-1, 0, 1)
             for dy in (-1, 0, 1)
             for dz in dz_values
             if (dx, dy, dz) != (0, 0, 0)],
            dtype=np.int32,
        )
        self._grid_size = np.asarray(self._config.grid_size)
        self._obstacles = np.zeros(self._config.grid_size, dtype=bool)
        for cell in self._config.obstacle_cells or []:
            self._obstacles[tuple(cell)] = True

    def _selection(self, node: Node) -> int | None:
        """Choose per-drone UCB actions in order, then encode them together.

        Conflicts concern intended target cells, before stochastic movement.
        Earlier choices stay fixed; only the current drone tries another action.
        Return None if it runs out of valid candidates. Statistics are updated
        later by backpropagation, never by selection itself.
        """
        actions = np.zeros(len(node.state), dtype=np.int32)
        # Reached drones remain solid, including those later in selection order.
        occupied = {tuple(drone[:3]) for drone in node.state if drone[3]}

        for drone_index, drone_state in enumerate(node.state):
            if drone_state[3]:
                # Any action ID is ignored by the simulator for a reached drone.
                continue

            visits = node.N[drone_index]
            tried = visits > 0
            ucb = np.full(len(self._action_vectors), np.inf)
            total_visits = max(1, int(visits.sum()))
            ucb[tried] = node.Q[drone_index, tried] + np.sqrt(
                2 * np.log(total_visits) / visits[tried]
            )

            targets = drone_state[:3] + self._action_vectors
            in_bounds = np.all(
                (targets >= 0) & (targets < self._grid_size), axis=1
            )
            candidates = np.flatnonzero(in_bounds)
            targets_in_bounds = targets[candidates]
            candidates = candidates[
                ~self._obstacles[tuple(targets_in_bounds.T)]
            ]

            while candidates.size:
                best_ucb = np.max(ucb[candidates])
                ties = candidates[ucb[candidates] == best_ucb]
                action = int(self._rng.choice(ties))
                target = tuple(targets[action])
                if target in occupied:
                    # Reject only for this selection, not permanently.
                    candidates = candidates[candidates != action]
                    continue
                actions[drone_index] = action
                occupied.add(target)
                break
            else:
                return None

        return self._encode_action(actions)

    def _encode_action(self, actions: np.ndarray) -> int:
        """Encode drone 0 as the least significant base-A digit."""
        num_actions = len(self._action_vectors)
        joint_action = 0
        base = 1
        for action in actions:
            joint_action += int(action) * base
            base *= num_actions
        return joint_action
