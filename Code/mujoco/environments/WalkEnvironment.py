from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces

import numpy as np
from rewards.walk_environment_reward_calc import WalkEnvironmentRewardCalc

"""
See mjvCamera_ structure in mujoco documentation
https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjvcamera
Azimuth angle https://en.wikipedia.org/wiki/Azimuth
See more in https://en.wikipedia.org/wiki/Spherical_coordinate_system
"""
DEFAULT_CAMERA_CONFIG = {
	"type": 0,
	"fixedcamid": 1,
	"trackbodyid": 0,

	"lookat": np.array([.07, 0., .10]),
	"distance": .80,
	"azimuth": 90,
	"elevation": -20,

	"orthographic": 0
}

class WalkEnvironmentV0(MujocoEnv):

	metadata = {
		"render_modes": ["human", "rgb_array", "depth_array"],
		"render_fps": 50,
	}

	def __init__(self, render_mode=None):
		super().__init__(
			"./our_robot/walking_scene.xml",
			frame_skip=10,
			observation_space=None,
			default_camera_config=DEFAULT_CAMERA_CONFIG,
		)

		self.utils = WalkEnvironmentRewardCalc( 
			gravity=self.model.opt.gravity,
			default_joint_position=self.model.key_ctrl[0],
			actuator_range=self.model.actuator_ctrlrange)
		
		self.observation_space = spaces.Box(
			low=-np.inf, high=np.inf, shape=self._get_obs().shape, dtype=np.float64
		)

		self.render_mode = render_mode
		self._last_render_time = -1.0
		self._max_episode_time_sec = 15.0
		self._step = 0
		self._debug = False  

	def step(self, action):
		self._step += 1
		self.do_simulation(action, self.frame_skip)
		observation = self._get_obs()
		reward, reward_info = self._calculate_rewards(action)
		
		terminated = not self.utils.is_healthy(self.data.qpos, self.data.qvel) and \
					not self.data.qvel[0] < self.utils.desired_velocity_min[0]
		truncated = self._step >= (self._max_episode_time_sec / self.dt)
		
		info = {
			"x_position": self.data.qpos[0],
			"y_position": self.data.qpos[1],
			"distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
			**reward_info,
			"paw_contact_forces": self.utils.get_paw_contact_forces(self.data, self.model),
			"patterns_matches": self.utils.diagonal_gait_reward(self.data, self.model)
		}

		if self.render_mode == "human" and (self.data.time - self._last_render_time) > (1.0 / self.metadata["render_fps"]):
			self.render()
			self._last_render_time = self.data.time

		self.utils.set_last_action(action)
		return observation, reward, terminated, truncated, info

	def _calculate_rewards(self, action):
		rewards = self._calculate_positive_rewards()
		costs = self._calculate_negative_costs(action)
		reward = max(0.0, rewards - costs)
		
		reward_info = {
			"linear_vel_tracking_reward": self.utils.get_linear_velocity_tracking_reward(self.data.qvel[:2], self.data.qpos[0]),
			"reward_ctrl": self.utils.torque_cost(self.data.qfrc_actuator[-8:]),
		}
		
		
		return reward, reward_info
	
	def _calculate_positive_rewards(self):
		return (
			+ self.utils.get_linear_velocity_tracking_reward(self.data.qvel[:2], self.data.qpos[0]) * self.utils.reward_weights["linear_vel_tracking"]
			+ self.utils.get_reward_safe_range(self.data.qpos, self.data.qvel) * self.utils.reward_weights["healthy"]
			+ self.utils.get_angular_velocity_tracking_reward(self.data.qvel[5]) * self.utils.reward_weights["angular_vel_tracking"]
			+ self.utils.diagonal_gait_reward(self.data, self.model) * self.utils.reward_weights["diagonal_gait_reward"]
			
		)

	#+ self.utils.non_flat_base_cost(self.data.qpos[3:7]) * self.utils.cost_weights["orientation"]
	def _calculate_negative_costs(self, action):
		return (
			+ self.utils.get_cost_distance(self.data.qpos[0]) * self.utils.cost_weights["cost_distance"]
			+ self.utils.vertical_velocity_cost(self.data.qvel[2]) * self.utils.cost_weights["vertical_vel"]
			+ self.utils.non_flat_base_cost(self.data.qpos[3:7]) * self.utils.cost_weights["orientation"]
		)

	def _debug_rewards_costs(self, rewards, costs):
		print("Rewards:", rewards)
		print("Costs:", costs)

	def _get_obs(self):
		dofs_position = self.data.qpos[7:].flatten() - self.model.key_ctrl[0, 7:]
		velocity = self.data.qvel.flatten()
		base_linear_velocity = velocity[:3]
		base_angular_velocity = velocity[3:6]
		dofs_velocity = velocity[6:]
		desired_vel = self.utils.desired_velocity
		last_action = self.utils.get_last_action()
		projected_gravity = self.utils.get_projected_gravity(self.data.qpos[3:7])
		
		curr_obs = np.concatenate(
			(
				base_linear_velocity * self.utils.obs_scale["linear_velocity"],
				base_angular_velocity * self.utils.obs_scale["angular_velocity"],
				desired_vel * self.utils.obs_scale["linear_velocity"],
				dofs_position * self.utils.obs_scale["dofs_position"],
				dofs_velocity * self.utils.obs_scale["dofs_velocity"],
				last_action
			)
		).clip(-self.utils.clip_obs_threshold, self.utils.clip_obs_threshold)
		#print (curr_obs)
		return curr_obs

	def reset_model(self):
		self.data.qpos[:] = self.model.key_qpos[0] + self.np_random.uniform(
			low=-self.utils.reset_noise_scale,
			high=self.utils.reset_noise_scale,
			size=self.model.nq,
		)
		self.data.ctrl[:] = self.model.key_ctrl[0] + self.utils.reset_noise_scale * self.np_random.standard_normal(self.data.ctrl.shape)
		self._desired_velocity = self.utils.sample_desired_vel()
		self._step = 0
		self.utils.set_last_action(np.zeros(8))
		self._feet_air_time = np.zeros(4)
		self._last_contacts = np.zeros(4)
		self._last_render_time = -1.0
		return self._get_obs()

	def _get_reset_info(self):
		return {
			"x_position": self.data.qpos[0],
			"y_position": self.data.qpos[1],
			"distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
		}
