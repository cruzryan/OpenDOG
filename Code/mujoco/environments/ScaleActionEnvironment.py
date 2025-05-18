import gymnasium as gym
import numpy as np 
from gymnasium import spaces

class ScaleActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.ctrl_ranges = np.array([
            [2.36, 3.14],    # FR_tigh_actuator
            [-2.6816, -1.2], # FR_knee_actuator
            [2.36, 3.14],    # BR_tigh_actuator
            [-2.6816, -1.2], # BR_knee_actuator
            [2.36, 3.14],    # FL_tigh_actuator
            [-2.6816, -1.2], # FL_knee_actuator
            [2.36, 3.14],    # BL_tigh_actuator
            [-2.6816, -1.2]  # BL_knee_actuator
        ], dtype=np.float32)
        
        # Ensure action space is [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,))

    def action(self, action):
        scaled_action = np.zeros_like(action)
        for i in range(8):
            low, high = self.ctrl_ranges[i]
            scaled_action[i] = low + (action[i] + 1) * (high - low) / 2.0
        return scaled_action
