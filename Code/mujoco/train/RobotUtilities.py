import mujoco
import numpy as np

class RobotUtils:
	def __init__(self, model, data):
		self.model = model
		self.data = data

		self.joint_names = [
			'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
			'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
			'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
			'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'
		]

		self.paws = ['FR', 'FL', 'RR', 'RL']

		self.trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'trunk')

		self.joint_indexes = self._get_joint_indexes()

	def get_joint_data(self,):
		joint_data = {}
		for joint_name, joint_index in self.joint_indices:
			joint_pos = self.data.qpos[joint_index]
			joint_vel = self.data.qvel[joint_index]
			joint_data[joint_name] = (joint_pos, joint_vel)
		return joint_data

	def get_robot_position(self):
		return self.data.xpos[self.trunk_id]

	def get_robot_orientation(self):
		quat = self.data.xquat[self.trunk_id]
		return self._quat_to_euler(quat)

	def _get_joint_indexes(self, model):
		joint_type = mujoco.mjtObj.mjOBJ_JOINT
		joint_indices = []

		for joint_name in self.joint_names:
			joint_index = mujoco.mj_name2id(model, joint_type, joint_name)
			joint_indices.append((joint_name, joint_index))
		
		return joint_indices

	def check_paw_contact(self, data):
		contact_info = {}
		ground_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
		paw_geom_ids = {paw: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, paw) for paw in self.paws}
		
		for paw in self.paws:
			contact_info[paw] = False

		for i in range(data.ncon):
			contact = data.contact[i]
			geom1, geom2 = contact.geom1, contact.geom2
			
			for paw, paw_geom_id in paw_geom_ids.items():
				if (geom1 == paw_geom_id and geom2 == ground_geom_id) or (geom1 == ground_geom_id and geom2 == paw_geom_id):
					contact_info[paw] = True
		
		return contact_info

	def _quat_to_euler(quat):
		w, x, y, z = quat
		# Yaw (Z-axis rotation)
		yaw = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
		# Pitch (Y-axis rotation)
		pitch = np.arcsin(2.0*(w*y - z*x))
		# Roll (X-axis rotation)
		roll = np.arctan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))
		
		# Convert from radians to degrees
		yaw_deg = np.degrees(yaw)
		pitch_deg = np.degrees(pitch)
		roll_deg = np.degrees(roll)
		
		return yaw_deg, pitch_deg, roll_deg