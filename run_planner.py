import argparse
from multi_drone import MultiDroneUnc

# Replace this with your own online planner
from dummy_planner import DummyPlanner

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help="Path to the yaml configuration file")
args = parser.parse_args()

def run(env, planner, planning_time_per_step=1.0):
    # Set the simulator to the initial state
    current_state = env.reset()
    num_steps = 0
    total_discounted_reward = 0.0
    history = []

    while True:
        # Use MCTS to plan an action from the current state
        action = planner.plan(current_state, planning_time_per_step)

        # Apply the action to the environment
        next_state, reward, done, info = env.step(action)        

        # Accumulate discounted reward
        total_discounted_reward += (env.get_config().discount_factor ** num_steps) * reward

        # Log trajectory
        history.append((current_state, action, reward, next_state, done, info))

        # Move forward
        current_state = next_state
        num_steps += 1

        if done or num_steps >= env.get_config().max_num_steps:
            break

    return total_discounted_reward, history

# Instantiate the environment with the given config
env = MultiDroneUnc(args.config)

# Instantiate the planner
planner = DummyPlanner(env, a_param=1.0, b_param=2)

# Run the planning loop
total_discounted_reward, history = run(env, planner, planning_time_per_step=1.0)
print(f"success: {history[-1][5]['success']}")
env.show()