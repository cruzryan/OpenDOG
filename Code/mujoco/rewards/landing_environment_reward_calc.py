import json
import numpy as np

#self.model.opt.gravity, self.model.key_ctrl[0], self.model.actuator_ctrlrange

class LandingEnvironmentRewardCalc:
	def __init__(self, gravity, default_joint_position, actuator_range):
			
		self.reward_weights = {
			"limit_hips_mobility_reward": 1.0,
			"linear_vel_tracking": 2.0,
			"angular_vel_tracking": 1.0,
			"healthy": 0,
			"feet_airtime": 1.0,
			"diagonal_gait_reward": 2.0
		}

		self.cost_weights = {
			"torque": 0.0002,
			"vertical_vel": 2.0,
			"xy_angular_vel": 0.05,
			"action_rate": 0.01,
			"joint_limit": 10.0,
			"joint_velocity": 0.01,
			"joint_acceleration": 2.5e-7,
			"orientation": 1.0,
			"collision": 1.0,
			"default_joint_position": 0.1,
			"diagonal_gait_cost": 0.5
		}

		self.curriculum_base = 0.3
		self.gravity_vector = np.array(gravity)
		self.default_joint_position = np.array(default_joint_position)
		self.desired_velocity_min = np.array([0.5, -0.0, -0.0])
		self.desired_velocity_max = np.array([0.8, 0.0, 0.0])
		self.desired_velocity = self.sample_desired_vel()
		self.obs_scale = {
			"linear_velocity": 2.0,
			"angular_velocity": 0.25,
			"dofs_position": 1.0,
			"dofs_velocity": 0.05,
		}
		self.tracking_velocity_sigma = 0.25
		
		#Control deviation of a rect line
		self.healthy_z_range = (0.22, 0.65)
		self.healthy_yaw_range = (-np.deg2rad(10), np.deg2rad(10))
		self.healthy_pitch_range = (-np.deg2rad(10), np.deg2rad(10))
		self.healthy_roll_range = (-np.deg2rad(10), np.deg2rad(10))

		self.feet_air_time = np.zeros(4)
		self.last_contacts = np.zeros(4)
		self.cfrc_ext_feet_indices = [4, 7, 10, 13]  # 4:FR, 7:FL, 10:RR, 13:RL
		self.cfrc_ext_contact_indices = [2, 3, 5, 6, 8, 9, 11, 12]
		
		dof_position_limit_multiplier = 0.9
		ctrl_range_offset = ( 0.5 * (1 - dof_position_limit_multiplier)*( actuator_range[:, 1] - actuator_range[:, 0] ))
		self.soft_joint_range = np.copy(actuator_range)
		self.soft_joint_range[:, 0] += ctrl_range_offset
		self.soft_joint_range[:, 1] -= ctrl_range_offset
		self.reset_noise_scale = 0.1
		self.last_action = np.zeros(12)
		self.clip_obs_threshold = 100.0

	# data.qpos, data.qvel
	def get_state_vector(self, positions_vector, velocities_vector): 
		return np.concatenate([positions_vector.flat, velocities_vector.flat])

	def is_healthy(self, positions_vector, velocities_vector):
		state = self.get_state_vector(positions_vector, velocities_vector)
		roll_x, pitch_y, yaw_z = self.euler_from_quaternion(state[3:7])
		min_z, max_z = self.healthy_z_range
		is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z

		min_yaw, max_yaw = self.healthy_yaw_range
		is_healthy = is_healthy and min_yaw <= yaw_z <= max_yaw

		min_roll, max_roll = self.healthy_roll_range
		is_healthy = is_healthy and min_roll <= roll_x <= max_roll

		min_pitch, max_pitch = self.healthy_pitch_range
		is_healthy = is_healthy and min_pitch <= pitch_y <= max_pitch

		return is_healthy

	# self.data.qpos[3:7]
	def get_projected_gravity(self, orientation_vector):
		euler_orientation = np.array(self.euler_from_quaternion(orientation_vector))
		projected_gravity_not_normalized = (
			np.dot(self.gravity_vector, euler_orientation) * euler_orientation
		)
		if np.linalg.norm(projected_gravity_not_normalized) == 0:
			return projected_gravity_not_normalized
		else:
			return projected_gravity_not_normalized / np.linalg.norm(
				projected_gravity_not_normalized
			)

	# External force acting on the body data.cfrc_ext[]
	def feet_contact_forces(self, external_forces):
		feet_contact_forces = external_forces[self.cfrc_ext_feet_indices]
		return np.linalg.norm(feet_contact_forces, axis=1)

	######### Positive Reward functions #########

	def phase_sync_reward(self, data):
		"""Reward for synchronized movement in phase: both front and rear pairs move together."""
		feet_contact_forces = self.feet_contact_forces(data.cfrc_ext)
		curr_contact = feet_contact_forces > 1.0

		# Check if front and rear legs are synchronized
		front_sync = curr_contact[0] == curr_contact[1]  # FR and FL synchronized
		rear_sync = curr_contact[2] == curr_contact[3]   # RR and RL synchronized

		if front_sync and rear_sync:
			return self.reward_weights["phase_sync_reward"]
		else:
			return 0

	def front_then_back_contact_reward(self, data):
		"""Reward for contacting the ground with front legs before rear legs."""
		feet_contact_forces = self.feet_contact_forces(data.cfrc_ext)
		curr_contact = feet_contact_forces > 1.0

		# Check if front legs (FR, FL) make contact before rear legs (RR, RL)
		front_contact = curr_contact[0] or curr_contact[1]  # FR or FL
		rear_contact = curr_contact[2] or curr_contact[3]   # RR or RL

		if front_contact and not rear_contact:
			return self.reward_weights["front_then_back_contact_reward"]
		else:
			return 0

	def weight_distribution_reward(self, data):
		"""Reward for an even distribution of weight across all four legs."""
		feet_contact_forces = self.feet_contact_forces(data.cfrc_ext)
		total_force = np.sum(feet_contact_forces)
		avg_force_per_leg = total_force / 4

		# Calculate deviation from the average for each leg
		deviation = np.abs(feet_contact_forces - avg_force_per_leg)
		max_deviation = np.max(deviation)

		# Reward is higher if deviation from average force is minimal
		return max(0, self.reward_weights["weight_distribution_reward"] - max_deviation)

	def feet_contact_forces(self, external_forces):
		"""Calculate the contact forces at each foot."""
		feet_contact_forces = external_forces[self.cfrc_ext_feet_indices]
		return np.linalg.norm(feet_contact_forces, axis=1)

	def diagonal_gait_reward(self, dt, external_forces):
		feet_contact_force_mag = self.feet_contact_forces(external_forces)
		curr_contact = feet_contact_force_mag > 1.0
		self.last_contacts = curr_contact

		pair_1_air = (not curr_contact[0]) and (not curr_contact[3])  # FR y RL
		pair_2_air = (not curr_contact[1]) and (not curr_contact[2])  # FL y RR

		self.feet_air_time += dt

		reward = 0
		if pair_1_air or pair_2_air:
			reward += 1.0

		if curr_contact[0] and curr_contact[3]:  # FR y RL
			reward += 1.0
			self.feet_air_time[0] = 0
			self.feet_air_time[3] = 0

		if curr_contact[1] and curr_contact[2]:  # FL y RR
			reward += 1.0
			self.feet_air_time[1] = 0
			self.feet_air_time[2] = 0

		if np.linalg.norm(self.desired_velocity[:2]) > 0.1:
			return reward
		else:
			return 0

	# self.data.qvel[:2]
	def linear_velocity_tracking_reward(self, xyz_velocity):
		vel_sqr_error = np.sum(
			np.square(self.desired_velocity[:2] - xyz_velocity)
		)
		return np.exp(-vel_sqr_error / self.tracking_velocity_sigma)

	# self.data.qvel[5]
	def angular_velocity_tracking_reward(self, angular_velocity_y):
		vel_sqr_error = np.square(self.desired_velocity[2] - angular_velocity_y)
		return np.exp(-vel_sqr_error / self.tracking_velocity_sigma)

	def feet_air_time_reward(self, dt, external_forces):
		feet_contact_force_mag = self.feet_contact_forces(external_forces)
		curr_contact = feet_contact_force_mag > 1.0
		contact_filter = np.logical_or(curr_contact, self.last_contacts)
		self.last_contacts = curr_contact

		first_contact = (self.feet_air_time > 0.0) * contact_filter
		self.feet_air_time += dt
		air_time_reward = np.sum((self.feet_air_time - 1.0) * first_contact)
		air_time_reward *= np.linalg.norm(self.desired_velocity[:2]) > 0.1

		# zero-out the air time for the feet that have just made contact (i.e. contact_filter==1)
		self.feet_air_time *= ~contact_filter

		return air_time_reward

	def healthy_reward(self, veclocity_vector, position_vector):
		return self.is_healthy(veclocity_vector, position_vector)

	######### Negative Reward functions #########

	def diagonal_gait_cost(self, external_forces):
		feet_contact_force_mag = self.feet_contact_forces(external_forces)
		curr_contact = feet_contact_force_mag > 1.0
		cost = 0
		if curr_contact[0] != curr_contact[3]:  #FR and RL
			cost += 1.0

		if curr_contact[1] != curr_contact[2]:  # FL y RR desincronizados
			cost += 1.0

		return cost

	def feet_contact_forces_cost(self, external_forces):
		return np.sum(
			(self.feet_contact_forces(external_forces) - self.max_contact_force).clip(min=0.0)
		)
	
	def non_flat_base_cost(self, orientation_vector):
		# Penalize the robot for not being flat on the ground
		return np.sum(np.square(self.get_projected_gravity(orientation_vector)[:2]))

	# data.cfrc_ext
	def collision_cost(self, external_forces):
		return np.sum(1.0*(np.linalg.norm(external_forces[self.cfrc_ext_contact_indices]) > 0.1))

	# self.data.qpos[7:]
	def joint_limit_cost(self, joints_angle_vector):
		# Penalize the robot for joints exceeding the soft control range
		out_of_range = (self.soft_joint_range[:, 0] - joints_angle_vector).clip(
			min=0.0
		) + (joints_angle_vector - self.soft_joint_range[:, 1]).clip(min=0.0)
		return np.sum(out_of_range)

	# qfrc_actuator torque self.data.qfrc_actuator[-12:]
	def torque_cost(self, torque_vector):
		return np.sum(np.square(torque_vector))
	
	def limit_hips_mobility_reward(self, hip_joint_position_vector):
		hip_movility_sum = np.sum(np.abs(hip_joint_position_vector))
		return np.exp(-hip_movility_sum)

	# vertical_velocity self.data.qvel[2]
	def vertical_velocity_cost(self, vertical_velocity):
		return np.square(vertical_velocity)

	# angular cost self.data.qvel[3:5])
	def xy_angular_velocity_cost(self, xy_angular_vector):
		return np.sum(np.square(xy_angular_vector))

	def action_rate_cost(self, action):
		return np.sum(np.square(self.last_action - action))

	# joint _velocity self.data.qvel[6:]
	def joint_velocity_cost(self, join_velocity_vector):
		return np.sum(np.square(join_velocity_vector))

	# joint_aceleration_cost self.data.qacc[6:]
	def acceleration_cost(self, joint_aceleration_vector):
		return np.sum(np.square(joint_aceleration_vector))

	# joint position vector self.data.qpos[7:]
	def default_joint_position_cost(self, joint_angle_position_vector):
		return np.sum(np.square(joint_angle_position_vector - self.default_joint_position))

	def smoothness_cost(self, joint_angle_position_vector):
		return np.sum(np.square(joint_angle_position_vector - self.last_action))

	def curriculum_factor(self):
		return self.curriculum_base**0.997

	def sample_desired_vel(self):
		desired_vel = np.random.default_rng().uniform(
			low=self.desired_velocity_min, high=self.desired_velocity_max
		)
		return desired_vel

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

