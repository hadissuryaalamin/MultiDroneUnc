"""Run the new per-drone MCTS planner in the simulator."""

import argparse

from multi_drone import MultiDroneUnc
from planner_new import Planner


def run(env, planner, planning_time_per_step=1.0):
    current_state = env.reset()
    num_steps = 0
    total_discounted_reward = 0.0
    history = []

    while True:
        action = planner.plan(current_state, planning_time_per_step)
        next_state, reward, done, info = env.step(action)

        total_discounted_reward += (
            env.get_config().discount_factor ** num_steps
        ) * reward
        history.append((current_state, action, reward, next_state, done, info))

        current_state = next_state
        num_steps += 1
        if done or num_steps >= env.get_config().max_num_steps:
            break

    return total_discounted_reward, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the yaml configuration file')
    args = parser.parse_args()

    env = MultiDroneUnc(args.config)
    planner = Planner(env)
    total_discounted_reward, history = run(env, planner, planning_time_per_step=1.0)
    print(f"success: {history[-1][5]['success']}, Total discounted reward: {total_discounted_reward}")
    env.show()
