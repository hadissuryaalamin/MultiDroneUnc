from math import log, sqrt
from time import perf_counter

import numpy as np
from multi_drone import MultiDroneUnc


class Node:
    """Satu state dalam pohon pencarian MCTS."""

    def __init__(self, state: np.ndarray):
        self.state = state.copy()
        self.visits = 0
        self.action_visits = {}  # N(s, a)
        self.action_values = {}  # Q(s, a): rata-rata return
        self.children = {}       # (action, successor_state_key) -> Node


class Planner:
    def __init__(self, env: MultiDroneUnc, a_param: float = 1.0, b_param: int = 1.0):
        self._env = env
        # Parameter placeholder dari starter; tetap cocok dengan run_planner.py.
        self._a_param = a_param
        self._b_param = b_param
        self._root = None

        config = env.get_config()
        self._gamma = config.discount_factor
        self._c = np.sqrt(2.0)
        self._H = config.max_num_steps
        self._deadline = 0.0

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        """Planning dari state saat ini untuk memilih SATU aksi nyata."""
        self._root = Node(current_state)
        self._deadline = perf_counter() + planning_time_per_step
        while perf_counter() < self._deadline:
            self._search(self._root, self._H)
        return max(self._root.action_values, key=self._root.action_values.get)

    def _search(self, node: Node, depth: int) -> float:
        """Satu putaran MCTS; mengembalikan return dari node ini."""
        if depth == 0 or perf_counter() >= self._deadline:
            return 0.0
        action = self._select_action(node)
        next_state, reward, done, _ = self._env.simulate(node.state, action)
        if done:
            return reward
        child = node.children.get((action, next_state.tobytes()))
        if child is None:
            child = self._expand(node, action, next_state)
            future_return = self._rollout(child.state, depth - 1)
        else:
            future_return = self._search(child, depth - 1)
        G = reward + self._gamma * future_return
        self._backpropagate(node, action, G)
        return G

    def _select_action(self, node: Node) -> int:
        """1. Selection: pilih aksi yang akan dicoba."""
        untried_actions = [
            action for action in range(self._env.num_actions)
            if node.action_visits.get(action, 0) == 0
        ]
        if untried_actions:
            return untried_actions[self._env.rng.integers(len(untried_actions))]
        exploration = self._c * sqrt(log(node.visits))
        return max(
            range(self._env.num_actions),
            key=lambda action: node.action_values[action]
            + exploration / sqrt(node.action_visits[action]),
        )

    def _expand(self, node: Node, action: int, next_state: np.ndarray) -> Node:
        """2. Expansion: tambahkan satu successor node baru."""
        key = (action, next_state.tobytes())
        child = Node(next_state)
        node.children[key] = child
        return child

    def _rollout(self, state: np.ndarray, depth: int) -> float:
        """3. Simulation: estimasi return memakai greedy rollout."""
        total_return = 0.0
        discount = 1.0
        for _ in range(depth):
            if perf_counter() >= self._deadline:
                break
            action = self._greedy_joint_action(state)
            state, reward, done, _ = self._env.simulate(state, action)
            total_return += discount * reward
            if done:
                break
            discount *= self._gamma
        return total_return

    def _greedy_joint_action(self, state: np.ndarray) -> int:
        """Susun aksi semua drone; untuk satu drone hanya berisi satu aksi."""
        # TODO: tiap drone memilih intended destination bebas obstacle,
        #       di dalam grid, yang paling dekat dengan goal; tie diacak.
        # TODO: gunakan indeks aksi 0 untuk drone yang sudah mencapai goal.
        # TODO: jika tidak ada kandidat, pilih aksi per-drone secara acak.
        # TODO: encode aksi per-drone menjadi satu integer joint action.
        raise NotImplementedError("TODO: implement greedy joint action")

    def _backpropagate(self, node: Node, action: int, total_return: float) -> None:
        """4. Backpropagation: perbarui statistik node dan aksi ini."""
        # TODO: increment N(s) dan N(s, a).
        # TODO: perbarui mean Q(s, a) menggunakan total_return.
        #       Pemanggilan _search yang rekursif meneruskannya sampai root.
        raise NotImplementedError("TODO: implement backpropagation")
