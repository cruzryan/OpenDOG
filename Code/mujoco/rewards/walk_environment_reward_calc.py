import numpy as np
import mujoco
#self.model.opt.gravity, self.model.key_ctrl[0], self.model.actuator_ctrlrange

class WalkEnvironmentRewardCalc:
	"""
	Kinematic body tree according with aparition 
	kt = [0, 1 ,2 , 3, 4, 5, 6, 7, 8, 9]
	0: chasis
	1: FL_tigh
	2: FL_calf
	3: FL_paw
	
	4: FR_tigh
	5: FR_calf
	6: FR_paw

	7: BL_tigh
	8: BL_calf
	9: BL_paw

	10: BR_tigh
	11: BR_calf
	12: BR_paw
	"""
	def __init__(self, gravity, default_joint_position, actuator_range):

		self.reward_weights = {
			"linear_vel_tracking": 3.0,
			"angular_vel_tracking": 1.0,
			"distance": .5, 
			"healthy": .98,
			"feet_airtime": .2,
			"diagonal_gait_reward": .6,
			"contact_force": .005,
		}

		self.cost_weights = {
			"cost_distance": 4,
			"torque": 0.0001,
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

		# 4:FL, 7:FR, 10:BL, 13:BR
		self.diagonal_walk_patterns = [
			[True, True, True, True],
			[True, True, False, True],
			[True, False, False, True],
			[True, False, True, True],
			[True, True, True, True],
			[True, True, True, False],
			[False, True, True, False],
			[True, True, True, True],
		]
	
		self.default_z_force_max = 3.10
		self.default_x_force_max = 3.10
		self.current_pattern_index = 0
		self.consecutive_matches = 0
		self.max_distance_achieve = 0

		self.gravity_vector = np.array(gravity)
		self.default_joint_position = np.array(default_joint_position)
		self.desired_velocity_min = np.array([1.2, -0.0, -0.0])
		self.desired_velocity_max = np.array([2, 0.0, 0.0])
		self.desired_velocity = self.sample_desired_vel()
		self.obs_scale = {
			"linear_velocity": 2.0,
			"angular_velocity": 0.25,
			"walked_distance": 1.80,
			"dofs_position": 1.0,
			"dofs_velocity": 0.05,
		}
		self.tracking_velocity_sigma = 0.25
		
		#Control deviation of a rect line
		self.z_range = (1.30, 1.65)
		self.yaw_range = np.deg2rad(5)
		self.pitch_range = np.deg2rad(5)
		self.roll_range = np.deg2rad(5)

		self.feet_air_time = np.zeros(4)
		self.last_contacts = np.zeros(4)
		self.body_feet_indices = [4, 7, 10, 13]  # 4:FR, 7:FL, 10:RR, 13:RL
		self.body_indices = [2, 3, 5, 6, 8, 9, 11, 12]
		
		dof_position_limit_multiplier = 0.5
		ctrl_range_offset = ( 0.5 * (1 - dof_position_limit_multiplier)*( actuator_range[:, 1] - actuator_range[:, 0] ))
		self.soft_joint_range = np.copy(actuator_range)
		self.soft_joint_range[:, 0] += ctrl_range_offset
		self.soft_joint_range[:, 1] -= ctrl_range_offset
		self.reset_noise_scale = 0.1
		self.last_action = np.zeros(8)
		self.clip_obs_threshold = 100.0

		self.time_feet_in_ground = 1000
		self.ground_force = [3, 0, 5]
		self.time_in_pattern = 0
		self.z_curve = self.ground_force[2]* np.sin(np.linspace(0, np.pi, 50))
		self.x_curve = self.ground_force[0]* np.sin(np.linspace(0, 2*np.pi, 50))
		self.samples_z = np.zeros((4, self.time_feet_in_ground), dtype=float)
		self.samples_x = np.zeros((4, self.time_feet_in_ground), dtype=float)

	# Last 12 values are the motor torques
	def torque_cost(torques):
		return np.sum(np.square(torques))

	def is_healthy(self, position, velocity):
		state = self.get_state_vector(position, velocity)
		
		if not np.isfinite(state).all():
			return False

		if not ( self.z_range[0] < state[2] < self.z_range[1]):
			return False
		
		current_roll, current_pitch, current_yaw = self.euler_from_quaternion(state[3:7])

		if not ( -self.roll_range < current_roll < self.roll_range):
			return False

		if not ( -self.pitch_range < current_pitch < self.pitch_range):
			return False

		if not ( -self.yaw_range < current_yaw < self.yaw_range):
			return False
		
		return True

	def get_reward_safe_range(self, position, velocity):
		state = self.get_state_vector(position, velocity)
		
		if not np.isfinite(state).all():
			return False

		if not ( self.z_range[0] < state[2] < self.z_range[1]):
			return False

		distance_z = min(state[2] - self.z_range[0], self.z_range[1] - state[2])

		current_roll, current_pitch, current_yaw = self.euler_from_quaternion(state[3:7])

		distance_roll = (self.roll_range - abs(current_roll)) if not ( abs(current_roll) > self.roll_range)  else 0
		distance_pitch = (self.pitch_range - abs(current_pitch)) if not ( abs(current_pitch) > self.pitch_range)  else 0
		distance_yaw = (self.yaw_range - abs(current_yaw)) if not ( abs(current_yaw) > self.yaw_range)  else 0

		max_distance = self.z_range[1] + self.roll_range + self.pitch_range + self.yaw_range

		return (distance_roll + distance_pitch + distance_yaw + distance_z)/max_distance

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

	######### Positive Reward functions #########
	def get_linear_velocity_tracking_reward(self, velocity, position_x):
		if position_x > 0:
			vel_sqr_error = np.sum(
				np.square(self.desired_velocity[:2] - velocity)
			)
			return np.exp(-vel_sqr_error / self.tracking_velocity_sigma)
		else:
			return 0
	
	def get_angular_velocity_tracking_reward(self, angular_velocity_y):
		vel_sqr_error = np.square(self.desired_velocity[2] - angular_velocity_y)
		return np.exp(-vel_sqr_error / self.tracking_velocity_sigma)
		
	def get_reward_ground_reaction_force(self, data, model):
		forces = self.get_paw_contact_forces(data, model)
		reward = 0
		for i in self.body_feet_indices:
			force_x = forces[i][0]
			force_z = forces[i][2]
			if force_x < 0 and force_z < 0 and data.qpos[0] > 0:
				reward += np.linalg.norm([force_x, force_z])
		return abs(reward)

	def get_reward_distance(self, position):
		if position[0] > 0 and self.max_distance_achieve < position[0]:
			self.max_distance_achieve = position[0]
			return position[0]
		else: 
			return 0

	def get_cost_distance(self, position):
		if position <= self.max_distance_achieve:
			return abs(position)
		return 0

	def diagonal_gait_reward(self, data, model):
		_, paw_index_to_contact_index = self.paws_in_ground(data, model)
		paw_contact_force = self.get_paw_contact_forces(data, model)

		current_state = [
			4 in paw_index_to_contact_index,  # FL
			7 in paw_index_to_contact_index,  # FR
			10 in paw_index_to_contact_index,  # BL
			13 in paw_index_to_contact_index # BR
		]

		expected_pattern = self.diagonal_walk_patterns[self.current_pattern_index]
		next_expected_pattern = self.diagonal_walk_patterns[self.current_pattern_index + 1]

		reward = 0
		contact_force = 0
		matches = 0
		self.time_in_pattern += 1
		for current, expected, next_expected_pattern, feet in zip(current_state, expected_pattern, next_expected_pattern,self.body_feet_indices):
			contact_force = np.linalg.norm([paw_contact_force[feet][0], paw_contact_force[feet][2]])
			if current == expected and contact_force >= 4.8 and self.time_in_pattern > 500:
				matches += 1

		pattern_matches = matches == len(self.body_feet_indices) 

		if pattern_matches:
			self.time_in_pattern = 0
			self.consecutive_matches += 1
			reward += self.consecutive_matches
			self.current_pattern_index = (self.current_pattern_index + 1) % len(self.diagonal_walk_patterns)
		else:
			self.time_in_pattern = 0
			reward = 0
			self.consecutive_matches = 0
			self.current_pattern_index = 0

		return reward

	def feet_air_time_reward(self, dt, data, model):
		paw_contact_force = self.get_paw_contact_forces(data, model)
		# FL_paw, FR_paw, BL_paw, BR_paw
		paws_current_contact = np.array([
			np.linalg.norm(paw_contact_force[4][0:3]),
			np.linalg.norm(paw_contact_force[7][0:3]),
			np.linalg.norm(paw_contact_force[10][0:3]),
			np.linalg.norm(paw_contact_force[13][0:3])
		])
		paws_curr_contact = paws_current_contact > 1.0
		contact_filter = np.logical_or(paws_curr_contact, self.last_contacts)
		self.last_contacts = paws_curr_contact

		first_contact = (self.feet_air_time > 0.0) * contact_filter
		self.feet_air_time += dt
		air_time_reward = np.sum((self.feet_air_time - 1.0) * first_contact)
		air_time_reward *= np.linalg.norm(self.desired_velocity[:2]) > 0.1
		self.feet_air_time *= ~contact_filter

		return air_time_reward

	######### Negative Reward functions #########
	
	def non_flat_base_cost(self, orientation_vector):
		return np.sum(np.square(self.get_projected_gravity(orientation_vector)[:2]))

	"""
	!important doesn't work, use contact approach
	"""
	def collision_cost(self, external_forces):
		return np.sum(1.0*(np.linalg.norm(external_forces[self.cfrc_ext_contact_indices]) > 0.1))

	def joint_limit_cost(self, joints_angle_vector):
		out_of_range = (self.soft_joint_range[:, 0] - joints_angle_vector).clip(
			min=0.0
		) + (joints_angle_vector - self.soft_joint_range[:, 1]).clip(min=0.0)
		return np.sum(out_of_range)

	def torque_cost(self, torque_vector):
		return np.sum(np.square(torque_vector))

	def vertical_velocity_cost(self, vertical_velocity):
		return np.square(vertical_velocity)

	def xy_angular_velocity_cost(self, xy_angular_vector):
		return np.sum(np.square(xy_angular_vector))

	def action_rate_cost(self, action):
		return np.sum(np.square(self.last_action - action))

	def joint_velocity_cost(self, join_velocity_vector):
		return np.sum(np.square(join_velocity_vector))

	def acceleration_cost(self, joint_aceleration_vector):
		return np.sum(np.square(joint_aceleration_vector))

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

	###	UTILS

	def get_state_vector(self, positions_vector, velocities_vector): 
		return np.concatenate([positions_vector.flat, velocities_vector.flat])

	def set_last_action(self, action):
		self.last_action = action

	def get_last_action(self):
		return self.last_action
	
	def paws_in_ground(self, data, model):
		ground_body_index = 0
		paws_contact_id = [-1,-1,-1,-1]
		
		paw_index_to_contact_index = {} 

		for contact_index in range(data.ncon): 
			contact = data.contact[contact_index] 
			
			geom1 = contact.geom1
			geom2 = contact.geom2

			for paw_index_enum, paw_body_index in enumerate(self.body_feet_indices):
				if (geom1 == ground_body_index and model.geom_bodyid[geom2] == paw_body_index) or \
					(geom2 == ground_body_index and model.geom_bodyid[geom1] == paw_body_index):
					paws_contact_id[paw_index_enum] = contact_index
					paw_index_to_contact_index[paw_body_index] = contact_index
					break
		
		return paws_contact_id, paw_index_to_contact_index

	def transform_force_coordinates_to_body(self, contact_frame, force_contact, body_quat):
		rotation_matrix_contact = np.reshape(contact_frame, (3, 3))
		rotation_matrix_body = np.zeros(9)

		mujoco.mju_quat2Mat(rotation_matrix_body, body_quat)
		
		rotation_matrix_body = np.reshape(rotation_matrix_body, (3, 3))
		force_global = rotation_matrix_contact @ force_contact
		force_body = rotation_matrix_body.T @ force_global

		return force_body

	def get_paw_contact_forces(self, data, model):
		paw_contact_forces = {paw_body_index: np.zeros(6) for paw_body_index in self.body_feet_indices}
		paw_contact_indices, paw_index_to_contact_index = self.paws_in_ground(data, model)

		for paw_body_index in self.body_feet_indices:
			contact_index = paw_index_to_contact_index.get(paw_body_index, -1)

			if contact_index != -1:
				force_array = np.zeros(6)
				mujoco._functions.mj_contactForce(model, data, contact_index, force_array)
				paw_contact_forces[paw_body_index][0:3] = self.transform_force_coordinates_to_body(
															data.contact[contact_index].frame,
															force_array[0:3],
															data.xquat[paw_body_index-1]
														)
				paw_contact_forces[paw_body_index][3:6] = force_array[3:6]
			else:
				paw_contact_forces[paw_body_index] = np.zeros(6)

		return paw_contact_forces
	
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

