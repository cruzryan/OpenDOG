from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
import mujoco
import numpy as np
from environments.landing_environment_reward_calc import LandingEnvironmentRewardCalc

DEFAULT_CAMERA_CONFIG = {
    "azimuth": 90.0,
    "distance": 3.0,
    "elevation": -25.0,
    "lookat": np.array([0., 0., 0.]),
    "fixedcamid": 0,
    "trackbodyid": -1,
    "type": 2,
}

class LandingEnvironmentV0(MujocoEnv):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None):
        super().__init__(
            "./unitree_go1/landing_scene.xml",
            frame_skip=10,
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
        )

        self.utils = LandingEnvironmentRewardCalc(
            gravity=self.model.opt.gravity,
            default_joint_position=self.model.key_ctrl[0],
            actuator_range=self.model.actuator_ctrlrange)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._get_obs().shape, dtype=np.float64
        )

        self._last_render_time = -1.0
        self._max_episode_time_sec = 15.0
        self._step = 0
        self._debug = False

        # IDs de los sitios de los pies y del cuerpo principal
        feet_site = ["FR", "FL", "RR", "RL"]
        self._feet_site_name_to_id = {
            f: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        }
        self._main_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY.value, "trunk"
        )

        self.render_mode = render_mode if render_mode == 'human' else None

    def step(self, action):
        self._step += 1
        self.do_simulation(action, self.frame_skip)
        
        observation = self._get_obs()
        reward, reward_info = self._calculate_rewards(action)
        
        terminated = not self.utils.is_healthy(self.data.qpos, self.data.qvel)
        truncated = self._step >= (self._max_episode_time_sec / self.dt)
        
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            **reward_info,
        }

        if self.render_mode == "human" and (self.data.time - self._last_render_time) > (1.0 / self.metadata["render_fps"]):
            self.render()
            self._last_render_time = self.data.time

        self._last_action = action
        return observation, reward, terminated, truncated, info

    def _calculate_rewards(self, action):
        rewards = self._calculate_positive_rewards()
        costs = self._calculate_negative_costs(action)
        reward = max(0.0, rewards - costs)
        
        reward_info = {
            "reward_phase_sync": self.utils.phase_sync_reward(self.data),
            "reward_front_then_back_contact": self.utils.front_then_back_contact_reward(self.data),
            "reward_weight_distribution": self.utils.weight_distribution_reward(self.data),
        }
        
        if self._debug:
            self._debug_rewards_costs(rewards, costs)
        
        return reward, reward_info

    def _calculate_positive_rewards(self):
        return (
            + self.utils.phase_sync_reward(self.data)
            + self.utils.front_then_back_contact_reward(self.data)
            + self.utils.weight_distribution_reward(self.data)
        )

    def _calculate_negative_costs(self, action):
        return (
            + self.utils.impact_force_cost(self.data)
            + self.utils.imbalance_cost(self.data)
            + self.utils.lack_of_flexion_cost(self.data)
        )

    def _debug_rewards_costs(self, rewards, costs):
        print("Rewards:", rewards)
        print("Costs:", costs)

    def _get_obs(self):
        dofs_position = self.data.qpos[7:].flatten() - self.model.key_qpos[0, 7:]
        velocity = self.data.qvel.flatten()
        base_linear_velocity = velocity[:3]
        base_angular_velocity = velocity[3:6]
        dofs_velocity = velocity[6:]
        projected_gravity = self.utils.get_projected_gravity(self.data.qpos[3:7])

        curr_obs = np.concatenate(
            (
                base_linear_velocity,
                base_angular_velocity,
                projected_gravity,
                dofs_position,
                dofs_velocity,
                self._last_action,
            )
        ).clip(-self.utils.clip_obs_threshold, self.utils.clip_obs_threshold)

        return curr_obs

    def reset_model(self):
        self.data.qpos[:] = self.model.key_qpos[0] + self.np_random.uniform(
            low=-self.utils.reset_noise_scale,
            high=self.utils.reset_noise_scale,
            size=self.model.nq,
        )
        self.data.ctrl[:] = self.model.key_ctrl[0] + self.utils.reset_noise_scale * self.np_random.standard_normal(self.data.ctrl.shape)
        self._step = 0
        self._last_action = np.zeros(12)
        self._last_render_time = -1.0
        return self._get_obs()

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }
