import numpy as np
from multi_drone import MultiDroneUnc

class DummyPlanner:
    def __init__(self, env: MultiDroneUnc, a_param: float = 1.0, b_param: int = 1.0):
        self._env = env
        self._a_param = a_param
        self._b_param = b_param

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        # This doesn't do anything useful. It simply returns the action 
        # representen by integer 0.
        return 0
