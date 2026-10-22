from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from planner_new import Planner


def state(value):
    return np.array([[value, 0, 0, 0]], dtype=np.int32)


class ScriptedEnv:
    num_actions = 1

    def __init__(self, transition, iterations, max_depth=3, gamma=0.5):
        self.transition = transition
        self.iterations = iterations
        self.completed = 0
        self.calls = []
        self.config = SimpleNamespace(max_num_steps=max_depth, discount_factor=gamma)

    def get_config(self):
        return self.config

    def simulate(self, current_state, action):
        value = int(current_state[0, 0])
        self.calls.append((value, action))
        next_value, reward, done = self.transition(value)
        if done:
            self.completed += 1
        return state(next_value), reward, done, {}

    def clock(self):
        return 2.0 if self.completed >= self.iterations else 0.0


class PlannerTreeTests(unittest.TestCase):
    def run_search(self, env):
        planner = Planner(env)
        initial = state(0)
        with patch('planner_new.perf_counter', side_effect=env.clock), patch.object(
            planner, '_selection', wraps=planner._selection
        ) as selection:
            action = planner.plan(initial, 1.0)
        np.testing.assert_array_equal(initial, state(0))
        self.assertEqual(action, 0)
        return selection.call_args_list[0].args[0], selection.call_args_list

    def test_stochastic_outcomes_are_preserved_and_reused(self):
        outcomes = iter([1, 2, 1])

        def transition(value):
            if value == 0:
                return next(outcomes), 10.0, False
            return 3, 20.0 * value, True

        env = ScriptedEnv(transition, iterations=3)
        root, selections = self.run_search(env)
        self.assertEqual(len(root.children), 2)
        child_a = root.children[(0, state(1).tobytes())]
        child_b = root.children[(0, state(2).tobytes())]
        self.assertIs(selections[-1].args[0], child_a)
        self.assertEqual(root.N[0], 3)
        self.assertAlmostEqual(root.Q[0], (20 + 30 + 20) / 3)
        self.assertEqual(child_a.N[0], 1)
        self.assertEqual(child_a.Q[0], 20)
        self.assertEqual(child_b.N[0], 0)

    def test_discounted_returns_update_every_selected_node(self):
        def transition(value):
            return value + 1, [10.0, 20.0, 40.0][value], value == 2

        for gamma, expected_root, expected_child in [
            (0.0, 10.0, 20.0),
            (0.5, 30.0, 40.0),
            (1.0, 70.0, 60.0),
        ]:
            with self.subTest(gamma=gamma):
                env = ScriptedEnv(transition, iterations=3, gamma=gamma)
                root, _ = self.run_search(env)
                child = root.children[(0, state(1).tobytes())]
                grandchild = child.children[(0, state(2).tobytes())]
                self.assertEqual(root.N[0], 3)
                self.assertEqual(child.N[0], 2)
                self.assertEqual(grandchild.N[0], 1)
                self.assertAlmostEqual(root.Q[0], expected_root)
                self.assertAlmostEqual(child.Q[0], expected_child)
                self.assertAlmostEqual(grandchild.Q[0], 40)

    def test_terminal_child_is_not_simulated_again(self):
        def transition(value):
            self.assertEqual(value, 0)
            return 1, -5.0, True

        env = ScriptedEnv(transition, iterations=2)
        root, _ = self.run_search(env)
        self.assertEqual(len(env.calls), 2)
        self.assertEqual(len(root.children), 1)
        self.assertEqual(root.N[0], 2)
        self.assertEqual(root.Q[0], -5)

    def test_tree_and_rollout_share_one_depth_limit(self):
        # The state repeats forever; the horizon must still bound each iteration.
        def transition(value):
            if len(env.calls) % 3 == 0:
                env.completed += 1
            return value, -1.0, False

        env = ScriptedEnv(transition, iterations=3, max_depth=3)
        root, _ = self.run_search(env)
        child = root.children[(0, state(0).tobytes())]
        grandchild = child.children[(0, state(0).tobytes())]
        self.assertEqual(len(env.calls), 9)
        self.assertEqual(root.Q[0], -1.75)
        self.assertEqual(child.Q[0], -1.5)
        self.assertEqual(grandchild.Q[0], -1)

    def test_expired_budget_stops_descent_and_keeps_sampled_reward(self):
        env = ScriptedEnv(lambda value: (value + 1, 10.0, False), iterations=1)
        env.clock = lambda: 2.0 if env.calls else 0.0
        root, _ = self.run_search(env)
        self.assertEqual(len(env.calls), 1)
        self.assertEqual(root.Q[0], 10)

    def test_zero_budget_does_not_simulate(self):
        env = ScriptedEnv(lambda value: self.fail('Unexpected simulation'), iterations=1)
        with patch('planner_new.perf_counter', return_value=0.0):
            self.assertEqual(Planner(env).plan(state(0), 0.0), 0)
        self.assertEqual(env.calls, [])


if __name__ == '__main__':
    unittest.main()
