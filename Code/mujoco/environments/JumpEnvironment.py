from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
import mujoco
import numpy as np
from rewards.jump_environment_reward_calc import JumpEnvironmentRewardCalc

DEFAULT_CAMERA_CONFIG = {
	"azimuth": 90.0,
	"distance": 3.0,
	"elevation": -25.0,
	"lookat": np.array([0., 0., 0.]),
	"fixedcamid": 0,
	"trackbodyid": -1,
	"type": 2,
}

class JumpEnvironmentV0(MujocoEnv):

	metadata = {
		"render_modes": [
			"human",
			"rgb_array",
			"depth_array",
		],
		"render_fps": 50,
	}

	def __init__(self, render_mode):
		MujocoEnv.__init__(
			self,
			"./unitree_go1/jump_scene.xml",
			frame_skip=10,
			observation_space=None,
			default_camera_config=DEFAULT_CAMERA_CONFIG,
		)

		self.utils = JumpEnvironmentRewardCalc(self.model, self.data)

		self.obs_scale = {
			"horizontal_distance_to_cube": 1.0,
			"vertical_distance_to_cube": 1.0,
			"linear_velocity": 1.0,
			"launch_angle": 1.0
		}

		self._last_render_time = -1.0
		self._max_episode_time_sec = 15.0
		self._step = 0

		self.observation_space = spaces.Box(
			low=-np.inf, high=np.inf, shape=self._get_obs().shape, dtype=np.float64
		)

		feet_site = [
			"FR",
			"FL",
			"RR",
			"RL",
		]
		self._feet_site_name_to_id = {
			f: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f)
			for f in feet_site
		}
		self._main_body_id = mujoco.mj_name2id(
			self.model, mujoco.mjtObj.mjOBJ_BODY.value, "trunk"
		)
		if render_mode == 'human':
			self.render_mode = render_mode

	def step(self, action):
		self._step += 1
		self.do_simulation(action, self.frame_skip)
		observation = self._get_obs()
		self.utils.data = self.data
		reward, reward_info = self.utils.compute_rewards(action, self.dt)
		terminated = not self.utils.static_stability()
		truncated = self._step >= (self._max_episode_time_sec / self.dt)
		info = {
			"x_position": self.data.qpos[0],
			"y_position": self.data.qpos[1],
			"z_position": self.data.qpos[1],
			"distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
			**reward_info,
		}

		if self.render_mode == "human" and (self.data.time - self._last_render_time) > (
			1.0 / self.metadata["render_fps"]
		):
			self.render()
			self._last_render_time = self.data.time

		self._last_action = action
		return observation, reward, terminated, truncated, info

	def _calc_reward(self, action):
		reward, reward_info = self.utils.compute_rewards(action, self.dt)
		return reward, reward_info

	def _get_obs(self):
		velocity = self.data.qvel.flatten()
		horizontal_distance_to_cube = np.array([0.3 - self.data.qpos[0]])
		vertical_distance_to_cube = np.array([0.3 - self.data.qpos[2]])
		base_linear_velocity = velocity[:3]
		vertical_velocity = np.array([velocity[2]])
		last_action = self.utils.last_action
		projected_gravity = self.utils.get_projected_gravity()

		curr_obs = np.concatenate(
			(
				horizontal_distance_to_cube * self.obs_scale["horizontal_distance_to_cube"],
				vertical_distance_to_cube * self.obs_scale["vertical_distance_to_cube"],
				base_linear_velocity * self.obs_scale["linear_velocity"],
				vertical_velocity,
				projected_gravity, 
				last_action,
			)
		).clip(-self.utils.clip_obs_threshold, self.utils.clip_obs_threshold)

		return curr_obs

	def reset_model(self):
		self.data.qpos[:] = self.model.key_qpos[0] + self.np_random.uniform(
			low=-self.utils.reset_noise_scale,
			high=self.utils.reset_noise_scale,
			size=self.model.nq,
		)
		self.data.ctrl[:] = self.model.key_ctrl[
			0
		] + self.utils.reset_noise_scale * self.np_random.standard_normal(
			*self.data.ctrl.shape
		)
		self._desired_velocity = self.utils.sample_desired_vel()
		self._step = 0
		self._last_action = np.zeros(12)
		self._feet_air_time = np.zeros(4)
		self._last_contacts = np.zeros(4)
		self._last_render_time = -1.0
		observation = self._get_obs()
		return observation

	def _get_reset_info(self):
		return {
			"x_position": self.data.qpos[0],
			"y_position": self.data.qpos[1],
			"distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
		}
