import numpy as np

class JumpEnvironmentRewardCalc:
	def __init__(self, model, data):
		self.model = model
		self.data = data

		self.reward_weights = {
			'feet_air_time': 1.0,
			'height_clearance': .2,
			'phase_sync': 0.8,
			'landing_angle': 1.0,
			'jump_velocity': 1.0,
			'landing_precision': 3.0,
			'landing_orientation': 2.0,
			'control_velocity_horizontal': 1.0
		}

		self.cost_weights = {
			'collision': 1.0,
			'distance_on_liftoff': 2.0,
			'vertical_velocity_on_landing': 1.5,
			'out_of_bounds': 3.0
		}

		self.cube_height = 0.5
		self.cube_position = np.array([1, 0, 0.5])

		self.curriculum_base = 0.3
		self.gravity_vector = np.array(self.model.opt.gravity)
		self.default_joint_position = np.array(self.model.key_ctrl[0])

		self.tracking_velocity_sigma = 0.45
		self.desired_velocity_min = np.array([1.20, -0.0, 1.20])
		self.desired_velocity_max = np.array([1.25, 0.0, 1.25])
		self.desired_velocity = self.sample_desired_vel()

		self.pitch_landing_angle = 30
		self.tolerange_landing_angle = (-np.deg2rad(5), np.deg2rad(5))

		self.healthy_yaw_range = (-np.deg2rad(20), np.deg2rad(20))
		self.healthy_pitch_range = (-np.deg2rad(20), np.deg2rad(20))
		self.healthy_roll_range = (-np.deg2rad(20), np.deg2rad(20))

		self.feet_air_time = np.zeros(4)
		self.last_contacts = np.zeros(4)
		self.cfrc_ext_feet_indices = [4, 7, 10, 13]
		self.cfrc_ext_contact_indices = [2, 3, 5, 6, 8, 9, 11, 12]
		self.soft_joint_range = np.copy(self.model.actuator_ctrlrange)
		self.reset_noise_scale = 0.1
		self.last_action = np.zeros(12)
		self.clip_obs_threshold = 100.0


	def compute_rewards(self, action, dt):
		landing_precision_reward = self.get_landing_precision_reward() * self.reward_weights['landing_precision']
		landing_orientation_reward = self.get_landing_orientation_reward() * self.reward_weights['landing_orientation']
		control_velocity_horizontal_reward = self.get_control_velocity_horizontal_reward() * self.reward_weights['control_velocity_horizontal']
		height_clearance_reward = self.get_height_clearance_reward() * self.reward_weights['height_clearance']
		phase_sync_reward = self.get_phase_sync_reward() * self.reward_weights['phase_sync']
		jump_velocity_reward = self.get_jump_velocity_reward() * self.reward_weights['jump_velocity']

		rewards = (landing_precision_reward + landing_orientation_reward + control_velocity_horizontal_reward +
		           height_clearance_reward + phase_sync_reward + jump_velocity_reward)

		distance_on_liftoff_cost = self.get_distance_on_liftoff_cost() * self.cost_weights['distance_on_liftoff']
		vertical_velocity_on_landing_cost = self.get_vertical_velocity_on_landing_cost() * self.cost_weights['vertical_velocity_on_landing']
		out_of_bounds_cost = self.get_out_of_bounds_cost() * self.cost_weights['out_of_bounds']
		collision_cost = self.get_collision_cost() * self.cost_weights['collision']

		costs = distance_on_liftoff_cost + vertical_velocity_on_landing_cost + out_of_bounds_cost + collision_cost

		reward = max(0.0, rewards - costs)
		reward_info = {
			'landing_precision': landing_precision_reward,
			'landing_orientation': landing_orientation_reward,
			'control_velocity_horizontal': control_velocity_horizontal_reward,
			'height_clearance': height_clearance_reward,
			'phase_sync': phase_sync_reward,
			'jump_velocity': jump_velocity_reward,
			'distance_on_liftoff': distance_on_liftoff_cost,
			'vertical_velocity_on_landing': vertical_velocity_on_landing_cost,
			'out_of_bounds': out_of_bounds_cost,
			'collision_cost': collision_cost
		}

		return reward, reward_info


	def get_landing_precision_reward(self):
		distance_to_cube = np.linalg.norm(self.cube_position[:2] - self.data.qpos[:2])
		return np.exp(-distance_to_cube) if self.data.qpos[2] >= self.cube_height else 0

	def get_landing_orientation_reward(self):
		roll, pitch, yaw = self.euler_from_quaternion(self.data.qpos[3:7])
		return np.exp(- (abs(roll) + abs(pitch) + abs(yaw)))

	def get_control_velocity_horizontal_reward(self):
		horizontal_velocity = np.linalg.norm(self.data.qvel[:2])
		return np.exp(-horizontal_velocity)

	def get_height_clearance_reward(self):
		return max(0, self.data.qpos[2] - self.cube_height)

	def get_phase_sync_reward(self):
		phase_reward = 0.0
		for i in range(0, len(self.feet_air_time), 2):
			phase_reward += abs(self.feet_air_time[i] - self.feet_air_time[i + 1])
		return -phase_reward

	def get_jump_velocity_reward(self):
		vel_sqr_error = np.sum(np.square(self.desired_velocity - self.data.qvel[:3]))
		return np.exp(-vel_sqr_error / self.tracking_velocity_sigma)

	def get_distance_on_liftoff_cost(self):
		distance_to_cube = np.linalg.norm(self.cube_position[:2] - self.data.qpos[:2])
		return np.exp(distance_to_cube) if self.data.qpos[2] < self.cube_height else 0

	def get_vertical_velocity_on_landing_cost(self):
		return self.data.qvel[2]**2 if self.data.qpos[2] >= self.cube_height else 0

	def get_out_of_bounds_cost(self):
		return 1.0 if np.linalg.norm(self.cube_position[:2] - self.data.qpos[:2]) > 1.0 else 0

	def get_collision_cost(self):
		return np.sum(1.0 * (np.linalg.norm(self.data.cfrc_ext[self.cfrc_ext_contact_indices]) > 0.1))

	def get_state_vector(self): 
		return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

	def feet_contact_forces(self):
		feet_contact_forces = self.data.cfrc_ext[self.cfrc_ext_feet_indices]
		return np.linalg.norm(feet_contact_forces, axis=1)

	def sample_desired_vel(self):
		return np.random.default_rng().uniform(
			low=self.desired_velocity_min, high=self.desired_velocity_max
		)
	
	def static_stability(self):
		state = self.get_state_vector()
		roll_x, _, yaw_z = self.euler_from_quaternion(state[3:7])

		min_yaw, max_yaw = self.healthy_yaw_range
		min_roll, max_roll = self.healthy_roll_range

		static_stability = np.isfinite(state).all() and min_yaw <= yaw_z <= max_yaw
		static_stability = static_stability and min_roll <= roll_x <= max_roll

		return static_stability

	def get_projected_gravity(self):
		euler_orientation = np.array(self.euler_from_quaternion(self.data.qpos[3:7]))
		projected_gravity_not_normalized = (
			np.dot(self.gravity_vector, euler_orientation) * euler_orientation
		)
		if np.linalg.norm(projected_gravity_not_normalized) == 0:
			return projected_gravity_not_normalized
		else:
			return projected_gravity_not_normalized / np.linalg.norm(
				projected_gravity_not_normalized
			)

	def euler_from_quaternion(self, quat):
		w = quat[0]
		x = quat[1]
		y = quat[2]
		z = quat[3]
		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		roll_x = np.arctan2(t0, t1)

		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch_y = np.arcsin(t2)

		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw_z = np.arctan2(t3, t4)

		return roll_x, pitch_y, yaw_z
