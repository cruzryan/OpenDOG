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
			"linear_vel_tracking": 5.0,
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

		# 4:FR, 7:FL, 10:RR, 13:RL
		self.diagonal_walk_patterns = [
			[True, True, True, True],
			[True, True, True, False],
			[False, True, True, False],
			[False, True, True, True],
			[True, True, True, True],
			[True, True, False, True],
			[True, False, False, True],
			[True, True, True, True],
		]
	
		self.current_pattern_index = 0
		self.successful_patterns = 0
		self.gravity_vector = np.array(gravity)
		self.default_joint_position = np.array(default_joint_position)
		self.desired_velocity_min = np.array([0.8, -0.0, -0.0])
		self.desired_velocity_max = np.array([3, 0.0, 0.0])
		self.desired_velocity = self.sample_desired_vel()
		self.obs_scale = {
			"linear_velocity": 2.0,
			"angular_velocity": 0.25,
			"dofs_position": 1.0,
			"dofs_velocity": 0.05,
		}
		self.tracking_velocity_sigma = 0.25
		
		#Control deviation of a rect line
		self.healthy_z_range = (1.30, 1.65)
		self.healthy_yaw_range = (-np.deg2rad(15), np.deg2rad(15))
		self.healthy_pitch_range = (-np.deg2rad(15), np.deg2rad(15))
		self.healthy_roll_range = (-np.deg2rad(15), np.deg2rad(15))

		self.feet_air_time = np.zeros(4)
		self.last_contacts = np.zeros(4)
		self.body_feet_indices = [4, 7, 10, 13]  # 4:FR, 7:FL, 10:RR, 13:RL
		self.body_indices = [2, 3, 5, 6, 8, 9, 11, 12]
		
		dof_position_limit_multiplier = 0.9
		ctrl_range_offset = ( 0.5 * (1 - dof_position_limit_multiplier)*( actuator_range[:, 1] - actuator_range[:, 0] ))
		self.soft_joint_range = np.copy(actuator_range)
		self.soft_joint_range[:, 0] += ctrl_range_offset
		self.soft_joint_range[:, 1] -= ctrl_range_offset
		self.reset_noise_scale = 0.1
		self.last_action = np.zeros(8)
		self.clip_obs_threshold = 100.0

	def get_state_vector(self, positions_vector, velocities_vector): 
		return np.concatenate([positions_vector.flat, velocities_vector.flat])

	def get_last_action(self):
		return self.last_action

	def torque_cost(torques):
		# Last 12 values are the motor torques
		return np.sum(np.square(torques))

	def is_healthy(self, positions_vector, velocities_vector):
		# Obtener el vector de estado
		state = self.get_state_vector(positions_vector, velocities_vector)
		
		# Verificar que todos los valores en el estado sean finitos
		if not np.isfinite(state).all():
			return False
		
		# Verificar la altura (eje z)
		min_z, max_z = self.healthy_z_range
		if not (min_z <= state[2] <= max_z):
			return False
		
		# Obtener los ángulos de roll, pitch y yaw
		roll_x, pitch_y, yaw_z = self.euler_from_quaternion(state[3:7])
		
		# Definir los rangos saludables para los ángulos
		angle_ranges = [
			(roll_x, self.healthy_roll_range),
			(pitch_y, self.healthy_pitch_range),
			(yaw_z, self.healthy_yaw_range)
		]
		
		# Verificar que todos los ángulos estén dentro de sus rangos saludables
		for angle, (min_angle, max_angle) in angle_ranges:
			if not (min_angle <= angle <= max_angle):
				return False
		
		# Si todas las condiciones se cumplen, el estado es saludable
		return True

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

	def paws_in_ground(self, data, model):
		d = data
		m = model
		paws_body_indexes = [4, 7, 10, 13] 
		ground_body_index = 0
		paws_contact_id = [-1,-1,-1,-1]
		
		paw_index_to_contact_index = {} 

		for contact_index in range(d.ncon): 
			contact = d.contact[contact_index] 
			
			geom1 = contact.geom1
			geom2 = contact.geom2

			for paw_index_enum, paw_body_index in enumerate(paws_body_indexes):
				if (geom1 == ground_body_index and m.geom_bodyid[geom2] == paw_body_index) or \
				(geom2 == ground_body_index and m.geom_bodyid[geom1] == paw_body_index):
					paws_contact_id[paw_index_enum] = contact_index
					paw_index_to_contact_index[paw_body_index] = contact_index
					break

		return paws_contact_id, paw_index_to_contact_index

	def get_paw_contact_forces(
			self,
			data,
			model
		):
		d = data
		m = model
		paws_body_indexes = [4, 7, 10, 13]
		paw_contact_forces = {paw_body_index: np.zeros(6) for paw_body_index in paws_body_indexes}
		paw_contact_indices, paw_index_to_contact_index = self.paws_in_ground(data, model)

		for paw_body_index in paws_body_indexes:
			contact_index = paw_index_to_contact_index.get(paw_body_index, -1)

			if contact_index != -1:
				force_array = np.zeros(6)
				mujoco._functions.mj_contactForce(m, d, contact_index, force_array)
				paw_contact_forces[paw_body_index] = force_array
			else:
				paw_contact_forces[paw_body_index] = np.zeros(6)

		return paw_contact_forces


	######### Positive Reward functions #########

	def get_reward_ground_reaction_force(
			self,
			data,
			model):
		forces = self.get_paw_contact_forces(data, model)
		reward = 0
		for i in self.body_feet_indices:
			reward = reward + forces[i][0]
		return reward/1000

	def get_reward_distance(self, position):
		return position[0]/10

	def diagonal_gait_reward(self, dt, external_forces):
		paw_contact_force = self.feet_contact_forces(external_forces)
		paw_curr_contact = paw_contact_force > 1.0  # Umbral para determinar contacto
		contact_filter = np.logical_or(paw_curr_contact, self.last_contacts)
		self.last_contacts = paw_curr_contact

		current_pattern = self.diagonal_walk_patterns[self.current_pattern_index]
		partial_match = np.sum(np.array(contact_filter) == np.array(current_pattern)) / 4.0

		# Verificar si el patrón es completamente correcto
		if np.all(np.array(contact_filter) == np.array(current_pattern)):
			self.successful_patterns += 1
			self.current_pattern_index = (self.current_pattern_index + 1) % len(self.diagonal_walk_patterns)
		else:
			self.successful_patterns = max(self.successful_patterns - 1, 0)  # Penalizar si falla el patrón

		# Recompensa total
		reward = partial_match + (self.successful_patterns * 8)
		return reward

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
		paw_contact_force = self.feet_contact_forces(external_forces)
		paw_curr_contact = paw_contact_force > 1.0
		contact_filter = np.logical_or(paw_curr_contact, self.last_contacts)
		self.last_contacts = paw_curr_contact

		first_contact = (self.feet_air_time > 0.0) * contact_filter
		self.feet_air_time += dt
		air_time_reward = np.sum((self.feet_air_time - 1.0) * first_contact)
		air_time_reward *= np.linalg.norm(self.desired_velocity[:2]) > 0.1
		self.feet_air_time *= ~contact_filter

		return air_time_reward

	def healthy_reward(self, veclocity_vector, position_vector):
		return self.is_healthy(veclocity_vector, position_vector)

	######### Negative Reward functions #########

	def diagonal_gait_cost(self, external_forces):
		paw_contact_force = self.feet_contact_forces(external_forces)
		paw_curr_contact = paw_contact_force > 1.0
		cost = 0
		if paw_curr_contact[0] != paw_curr_contact[3]:  #FR and RL
			cost += 1.0

		if paw_curr_contact[1] != paw_curr_contact[2]:  # FL y RR desincronizados
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

	def set_last_action(self, action):
		self.last_action = action

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

