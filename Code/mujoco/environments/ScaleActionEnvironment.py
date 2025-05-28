import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ScaleActionWrapper(gym.ActionWrapper):
	def __init__(self, env):
		super().__init__(env)
		self.ctrl_ranges = np.array([
			[2.36, 2.8],
			[-1.8, -1.20],
			[2.36, 2.8],
			[-1.8, -1.20],
			[2.36, 2.8],
			[-1.8, -1.20],
			[2.36, 2.8],
			[-1.8, -1.20],
		], dtype=np.float32)

		self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

	def action(self, action: np.ndarray) -> np.ndarray:
		scaled = self.ctrl_ranges[:,0] + (action + 1.0) * (self.ctrl_ranges[:,1] - self.ctrl_ranges[:,0]) / 2.0
		return scaled
