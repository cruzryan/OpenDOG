import mujoco
import mujoco.viewer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import math
import os
import time
import json
from collections import deque
import threading

try:
    import keyboard
    KEYBOARD_LIB_AVAILABLE = True
except ImportError:
    KEYBOARD_LIB_AVAILABLE = False
    print("WARNING: 'keyboard' library not found. Ctrl+L rendering toggle will not be available.")
    print("To enable, run: pip install keyboard")


# --- Configuration ---
ACTUATOR_NAMES_ORDERED = [
    "FR_tigh_actuator", "FR_knee_actuator",
    "FL_tigh_actuator", "FL_knee_actuator",
    "BR_tigh_actuator", "BR_knee_actuator",
    "BL_tigh_actuator", "BL_knee_actuator",
]
LEG_ACTUATORS = {
    "FR": ["FR_tigh_actuator", "FR_knee_actuator"],
    "FL": ["FL_tigh_actuator", "FL_knee_actuator"],
    "BR": ["BR_tigh_actuator", "BR_knee_actuator"],
    "BL": ["BL_tigh_actuator", "BL_knee_actuator"],
}
ACTUATOR_TO_JOINT_NAME_MAP = {}
JOINT_NAME_TO_QPOS_IDX_MAP = {}


JSON_OUTPUT_PATH_TEMPLATE = './walk_rl_sym_ep{}.json'
PTH_SAVE_INTERVAL = 100
JSON_MAX_STEPS_EPISODIC = 50
JSON_MAX_STEPS_FINAL = 100
XML_FILE_NAME = 'walking_scene.xml'

# --- ADAPTIVE HYPERPARAMETER CONFIG ---
ADAPTATION_CHECK_INTERVAL = 10
AVGR_HISTORY_LEN = 5
MIN_LR = 1e-6; MAX_LR = 3e-4
MIN_ENTROPY_COEF = 0.0001; MAX_ENTROPY_COEF = 0.01
MIN_ACTION_LOG_STD = math.log(0.15); MAX_ACTION_LOG_STD = math.log(0.7)
INITIAL_LEARNING_RATE = 1e-4
INITIAL_ENTROPY_COEF = 0.005
INITIAL_ACTION_STD_INIT = 0.6

# RL Hyperparameters (others)
GAMMA = 0.99; GAE_LAMBDA = 0.95
VALUE_LOSS_COEF = 0.5; MAX_GRAD_NORM = 0.5; NUM_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 200
POLICY_UPDATE_INTERVAL = 4096; NUM_EPOCHS_PER_UPDATE = 10

# Refined Leg Position Penalty Hyperparameters
LEG_AT_HOME_THRESHOLD_DEG = 10.0
MOVING_LEG_MAX_DEVIATION_DEG = 30.0
LEG_POSITIONING_PENALTY = 1.5

# Action scaling
ACTION_AMPLITUDE_DEG = 20.0
ACTION_AMPLITUDE_RAD = math.radians(ACTION_AMPLITUDE_DEG)

# Environment parameters
TARGET_HEIGHT = 0.15; MIN_HEIGHT_TERMINAL = 0.03; MAX_HEIGHT_TERMINAL = 0.50
ORIENTATION_TERMINATION_LIMIT_DEG = 35.0
ORIENTATION_TERMINATION_LIMIT_RAD = math.radians(ORIENTATION_TERMINATION_LIMIT_DEG)
ORIENTATION_PENALTY_THRESHOLD_DEG = 30.0
ORIENTATION_PENALTY_THRESHOLD_RAD = math.radians(ORIENTATION_PENALTY_THRESHOLD_DEG)

MIN_FWD_DISPLACEMENT_FOR_BACKWARD_CHECK = 0.05
POLICY_DECISION_DT = 0.20; NUM_SETTLE_STEPS = 100; PHASE_CYCLE_DURATION_POLICY_STEPS = 4

# GAIT PARAMETERS
KNEE_SWING_FLEXION_DEG = 30.0
KNEE_SWING_FLEXION_RAD = math.radians(KNEE_SWING_FLEXION_DEG)
KNEE_FLEXION_SIGN_FACTOR = 1.0

# --- SIM-TO-REAL MAPPING CONFIGURATION ---
real_robot_home_deg_map = {name: 0.0 for name in ACTUATOR_NAMES_ORDERED}
real_robot_home_deg_map.update({
    "FR_tigh_actuator": -45.0, "FR_knee_actuator": 45.0, "BR_tigh_actuator": 45.0, "BR_knee_actuator": -45.0,
    "FL_tigh_actuator": 45.0, "FL_knee_actuator": 45.0, "BL_tigh_actuator": 45.0, "BL_knee_actuator": -45.0,
})
joint_scale_factors = {name: 1.0 for name in ACTUATOR_NAMES_ORDERED}
sim_keyframe_home_qpos_map = {}
actuator_ctrl_props = {}

# --- Global Render Toggle ---
g_dynamic_render_active = False # This flag is set by the keyboard hook
g_render_lock = threading.Lock() # To protect access to g_dynamic_render_active

def quat_to_ypr(quat):
    q0, q1, q2, q3 = quat[0], quat[1], quat[2], quat[3]
    sinr_cosp = 2 * (q0 * q1 + q2 * q3); cosr_cosp = 1 - 2 * (q1 * q1 + q2 * q2)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (q0 * q2 - q3 * q1)
    pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)
    siny_cosp = 2 * (q0 * q3 + q1 * q2); cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw, pitch, roll

def convert_sim_rad_to_real_deg(act_name, sim_rad_commanded, _sim_keyframe_home_qpos_map, _real_robot_home_deg_map, _scale_factors):
    joint_name_for_actuator = ACTUATOR_TO_JOINT_NAME_MAP.get(act_name)
    if joint_name_for_actuator is None: return None
    sim_home_qpos_rad = _sim_keyframe_home_qpos_map.get(joint_name_for_actuator)
    real_home_deg_offset = _real_robot_home_deg_map.get(act_name)
    scale = _scale_factors.get(act_name, 1.0)
    if sim_home_qpos_rad is None or real_home_deg_offset is None or not isinstance(sim_rad_commanded, (int, float)): return None
    sim_delta_rad = sim_rad_commanded - sim_home_qpos_rad
    real_delta_deg = scale * math.degrees(sim_delta_rad)
    real_target_deg = real_home_deg_offset + real_delta_deg
    return real_target_deg

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, action_dim), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(state_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1))
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * math.log(action_std_init))
    def forward(self, state):
        action_mean = self.actor(state); action_std = torch.exp(self.action_log_std.expand_as(action_mean))
        return Normal(action_mean, action_std), self.critic(state)

class QuadrupedEnv:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.sim_steps_per_policy_step = max(1, int(POLICY_DECISION_DT / self.model.opt.timestep))
        global actuator_ctrl_props
        for name in ACTUATOR_NAMES_ORDERED:
            mujoco_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if mujoco_act_id == -1: raise ValueError(f"Actuator '{name}' not found for props.")
            ctrlrange = self.model.actuator_ctrlrange[mujoco_act_id]
            center = (ctrlrange[0] + ctrlrange[1]) / 2.0
            actuator_ctrl_props[name] = {'center': center, 'ctrlrange': ctrlrange.copy(), 'mujoco_id': mujoco_act_id}

        self.state_dim = 1 + 2 + 8 + 8 + 1 + 2
        self.action_dim = 4
        temp_model_for_init = mujoco.MjModel.from_xml_path(model_path)
        temp_data_for_init = mujoco.MjData(temp_model_for_init)
        key_id_for_init = mujoco.mj_name2id(temp_model_for_init, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if key_id_for_init == -1: raise ValueError("Keyframe 'home' not found.")
        mujoco.mj_resetDataKeyframe(temp_model_for_init, temp_data_for_init, key_id_for_init)
        self.initial_qpos = temp_data_for_init.qpos.copy(); self.initial_qvel = temp_data_for_init.qvel.copy()
        self.initial_ctrl = temp_data_for_init.ctrl.copy()
        self.episode_policy_step_counter = 0
        self.last_commanded_clipped_sim_rad = self.initial_ctrl.copy()
        self.previous_x_qpos_in_episode = 0.0
        self.cumulative_positive_x_displacement = 0.0; self.cumulative_negative_x_displacement = 0.0
        self.previous_net_forward_displacement = 0.0
        self.target_gait_angles_rad_sim = np.zeros(len(ACTUATOR_NAMES_ORDERED))

    def _get_observation(self):
        z_pos = self.data.qpos[2]; _, pitch, roll = quat_to_ypr(self.data.qpos[3:7])
        joint_pos_values = []
        joint_vel_values = []
        for act_name in ACTUATOR_NAMES_ORDERED:
            joint_name = ACTUATOR_TO_JOINT_NAME_MAP[act_name]
            qpos_idx = JOINT_NAME_TO_QPOS_IDX_MAP[joint_name]
            joint_id_for_vel = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qvel_idx = self.model.jnt_dofadr[joint_id_for_vel]
            joint_pos_values.append(self.data.qpos[qpos_idx] - sim_keyframe_home_qpos_map[joint_name])
            joint_vel_values.append(self.data.qvel[qvel_idx])
        joint_pos_dev = np.array(joint_pos_values); joint_vel = np.array(joint_vel_values)
        trunk_fwd_vel = self.data.qvel[0]
        phase_progress = (self.episode_policy_step_counter % PHASE_CYCLE_DURATION_POLICY_STEPS) / PHASE_CYCLE_DURATION_POLICY_STEPS
        sin_phase = np.sin(phase_progress * 2 * np.pi); cos_phase = np.cos(phase_progress * 2 * np.pi)
        state = np.concatenate([[z_pos - TARGET_HEIGHT],
                                [pitch, roll],
                                joint_pos_dev,
                                joint_vel,
                                [trunk_fwd_vel],
                                [sin_phase, cos_phase]])
        return state.astype(np.float32)

    def reset(self):
        self.data.qpos[:] = self.initial_qpos; self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = self.initial_ctrl; mujoco.mj_forward(self.model, self.data)
        self.episode_policy_step_counter = 0
        self.previous_x_qpos_in_episode = self.data.qpos[0]
        self.cumulative_positive_x_displacement = 0.0; self.cumulative_negative_x_displacement = 0.0
        self.previous_net_forward_displacement = 0.0
        for _ in range(NUM_SETTLE_STEPS):
            self.data.ctrl[:] = self.initial_ctrl
            try: mujoco.mj_step(self.model, self.data)
            except mujoco.FatalError: break
            self.sync_viewer_if_active()
        self.last_commanded_clipped_sim_rad = self.data.ctrl.copy()
        self.previous_x_qpos_in_episode = self.data.qpos[0]
        return self._get_observation()

    def _set_target_gait_angles(self, current_gait_phase_idx):
        global sim_keyframe_home_qpos_map, KNEE_FLEXION_SIGN_FACTOR
        for i, act_name in enumerate(ACTUATOR_NAMES_ORDERED):
            joint_name = ACTUATOR_TO_JOINT_NAME_MAP[act_name]
            self.target_gait_angles_rad_sim[i] = sim_keyframe_home_qpos_map[joint_name]
        knee_swing_delta_rad = KNEE_FLEXION_SIGN_FACTOR * KNEE_SWING_FLEXION_RAD
        if current_gait_phase_idx == 0:
            fr_knee_idx = ACTUATOR_NAMES_ORDERED.index("FR_knee_actuator")
            bl_knee_idx = ACTUATOR_NAMES_ORDERED.index("BL_knee_actuator")
            self.target_gait_angles_rad_sim[fr_knee_idx] += knee_swing_delta_rad
            self.target_gait_angles_rad_sim[bl_knee_idx] += knee_swing_delta_rad
        elif current_gait_phase_idx == 2:
            fl_knee_idx = ACTUATOR_NAMES_ORDERED.index("FL_knee_actuator")
            br_knee_idx = ACTUATOR_NAMES_ORDERED.index("BR_knee_actuator")
            self.target_gait_angles_rad_sim[fl_knee_idx] += knee_swing_delta_rad
            self.target_gait_angles_rad_sim[br_knee_idx] += knee_swing_delta_rad

    def step(self, policy_actions_scaled_neg1_to_1):
        self.episode_policy_step_counter += 1
        current_gait_phase_idx = self.episode_policy_step_counter % PHASE_CYCLE_DURATION_POLICY_STEPS
        self._set_target_gait_angles(current_gait_phase_idx)

        final_clipped_commands_all_actuators = np.zeros_like(self.data.ctrl)
        actuator_deltas_rad = {
            "FR_tigh_actuator": policy_actions_scaled_neg1_to_1[0] * ACTION_AMPLITUDE_RAD,
            "FR_knee_actuator": policy_actions_scaled_neg1_to_1[1] * ACTION_AMPLITUDE_RAD,
            "FL_tigh_actuator": policy_actions_scaled_neg1_to_1[2] * ACTION_AMPLITUDE_RAD,
            "FL_knee_actuator": policy_actions_scaled_neg1_to_1[3] * ACTION_AMPLITUDE_RAD,
        }
        actuator_deltas_rad["BR_tigh_actuator"] = actuator_deltas_rad["FL_tigh_actuator"]
        actuator_deltas_rad["BR_knee_actuator"] = actuator_deltas_rad["FL_knee_actuator"]
        actuator_deltas_rad["BL_tigh_actuator"] = actuator_deltas_rad["FR_tigh_actuator"]
        actuator_deltas_rad["BL_knee_actuator"] = actuator_deltas_rad["FR_knee_actuator"]

        for i, act_name in enumerate(ACTUATOR_NAMES_ORDERED):
            props = actuator_ctrl_props[act_name]
            mujoco_act_id = props['mujoco_id']
            joint_name = ACTUATOR_TO_JOINT_NAME_MAP[act_name]
            sim_home_rad = sim_keyframe_home_qpos_map[joint_name]
            policy_delta_rad = actuator_deltas_rad[act_name]
            sim_target_rad_unclipped = sim_home_rad + policy_delta_rad
            clipped_sim_target_rad = np.clip(sim_target_rad_unclipped, props['ctrlrange'][0], props['ctrlrange'][1])
            final_clipped_commands_all_actuators[mujoco_act_id] = clipped_sim_target_rad

        self.data.ctrl[:] = final_clipped_commands_all_actuators
        sum_mj_errors = 0
        for _ in range(self.sim_steps_per_policy_step):
            try: mujoco.mj_step(self.model, self.data)
            except mujoco.FatalError: sum_mj_errors +=1; break
            self.sync_viewer_if_active()

        current_x_pos_after_sim_loop = self.data.qpos[0]
        delta_x_this_policy_step = current_x_pos_after_sim_loop - self.previous_x_qpos_in_episode
        if delta_x_this_policy_step > 0: self.cumulative_positive_x_displacement += delta_x_this_policy_step
        elif delta_x_this_policy_step < 0: self.cumulative_negative_x_displacement += abs(delta_x_this_policy_step)
        self.previous_x_qpos_in_episode = current_x_pos_after_sim_loop
        obs = self._get_observation()

        forward_vel_x = self.data.qvel[0]
        reward_forward_velocity = 600.0 * forward_vel_x
        current_net_forward_displacement = self.cumulative_positive_x_displacement - self.cumulative_negative_x_displacement
        delta_net_forward_displacement_this_step = current_net_forward_displacement - self.previous_net_forward_displacement
        self.previous_net_forward_displacement = current_net_forward_displacement
        reward_new_cumulative_progress = 0.0
        if delta_net_forward_displacement_this_step > 0.001: reward_new_cumulative_progress = 150.0 * delta_net_forward_displacement_this_step
        reward_backward_velocity_penalty = 0.0
        if forward_vel_x < -0.01: reward_backward_velocity_penalty = -70.0 * abs(forward_vel_x)
        reward_alive = 0.25
        reward_sideways_vel = -0.9 * abs(self.data.qvel[1])
        current_yaw, current_pitch, current_roll = quat_to_ypr(self.data.qpos[3:7])
        reward_excessive_orientation = 0.0
        orientation_penalty_factor = -0.25
        if abs(current_roll) > ORIENTATION_PENALTY_THRESHOLD_RAD: reward_excessive_orientation += orientation_penalty_factor * (abs(current_roll) - ORIENTATION_PENALTY_THRESHOLD_RAD)**2
        if abs(current_pitch) > ORIENTATION_PENALTY_THRESHOLD_RAD: reward_excessive_orientation += orientation_penalty_factor * (abs(current_pitch) - ORIENTATION_PENALTY_THRESHOLD_RAD)**2
        action_delta_sq_sum = np.sum([(final_clipped_commands_all_actuators[actuator_ctrl_props[name]['mujoco_id']] -
                                     self.last_commanded_clipped_sim_rad[actuator_ctrl_props[name]['mujoco_id']])**2
                                     for name in ACTUATOR_NAMES_ORDERED])
        reward_action_smoothness = -0.5 * action_delta_sq_sum

        reward_leg_positioning = 0.0
        legs_at_home_status = {}
        num_legs_at_home = 0
        num_legs_moving_too_far = 0
        for leg_id, leg_act_names_in_leg in LEG_ACTUATORS.items():
            leg_joints_max_dev_from_home_deg = 0.0
            for act_name_iter in leg_act_names_in_leg:
                mujoco_act_id = actuator_ctrl_props[act_name_iter]['mujoco_id']
                current_sim_cmd_rad = final_clipped_commands_all_actuators[mujoco_act_id]
                joint_name = ACTUATOR_TO_JOINT_NAME_MAP[act_name_iter]
                sim_home_qpos_rad = sim_keyframe_home_qpos_map[joint_name]
                scale = joint_scale_factors.get(act_name_iter, 1.0)
                sim_delta_rad_from_home = current_sim_cmd_rad - sim_home_qpos_rad
                real_equivalent_delta_deg = scale * math.degrees(sim_delta_rad_from_home)
                leg_joints_max_dev_from_home_deg = max(leg_joints_max_dev_from_home_deg, abs(real_equivalent_delta_deg))
            if leg_joints_max_dev_from_home_deg <= LEG_AT_HOME_THRESHOLD_DEG:
                legs_at_home_status[leg_id] = "at_home"
                num_legs_at_home += 1
            elif leg_joints_max_dev_from_home_deg <= MOVING_LEG_MAX_DEVIATION_DEG:
                legs_at_home_status[leg_id] = "moving_ok"
            else:
                legs_at_home_status[leg_id] = "moving_too_far"
                num_legs_moving_too_far += 1
        if num_legs_at_home < 2 or num_legs_moving_too_far > 0:
            reward_leg_positioning = -LEG_POSITIONING_PENALTY

        total_reward = (reward_forward_velocity + reward_new_cumulative_progress +
                        reward_backward_velocity_penalty + reward_alive  +
                        reward_sideways_vel + reward_excessive_orientation + reward_action_smoothness +
                        reward_leg_positioning)

        done = False
        info = {"sim_target_rad": final_clipped_commands_all_actuators.copy(), "termination_reason": "max_steps"}

        if sum_mj_errors > 0:
            total_reward -= 10.0; done = True; info["mj_error"] = True
            info["termination_reason"] = "mj_error"
        current_height_term = self.data.qpos[2]
        if not (MIN_HEIGHT_TERMINAL < current_height_term < MAX_HEIGHT_TERMINAL):
            done = True; info["termination_reason"] = "height_limit"
        if abs(current_roll) > ORIENTATION_TERMINATION_LIMIT_RAD or \
           abs(current_pitch) > ORIENTATION_TERMINATION_LIMIT_RAD:
            done = True; info["termination_reason"] = "orientation_limit"
        if not done:
            if self.cumulative_positive_x_displacement > MIN_FWD_DISPLACEMENT_FOR_BACKWARD_CHECK and \
               self.cumulative_negative_x_displacement > 0.5 * self.cumulative_positive_x_displacement:
                total_reward -= 2.0
                done = True; info["termination_reason"] = "too_much_backward"

        print_debug = False
        if g_dynamic_render_active and (self.episode_policy_step_counter % 5 == 0 or done):
            print_debug = True
        elif self.episode_policy_step_counter % 20 == 0 or self.episode_policy_step_counter == MAX_STEPS_PER_EPISODE -1 or done :
             print_debug = True

        if print_debug:
            status_str = ", ".join([f"{leg}: {status}" for leg, status in legs_at_home_status.items()])
            print(f"  DBG S{self.episode_policy_step_counter}: TotR={total_reward:.2f} || FwdVel={reward_forward_velocity:.2f} ({forward_vel_x:.3f}) | NewCumProg={reward_new_cumulative_progress:.2f} | "
                  f"BwdVelP={reward_backward_velocity_penalty:.2f} | Alive={reward_alive:.2f} | "
                  f"SideVelP={reward_sideways_vel:.2f} | OrientP={reward_excessive_orientation:.2f} (R:{math.degrees(current_roll):.1f} P:{math.degrees(current_pitch):.1f}) | SmoothP={reward_action_smoothness:.2f} | "
                  f"LegPosP={reward_leg_positioning:.2f} (AtHome:{num_legs_at_home}, TooFar:{num_legs_moving_too_far}) [{status_str}] || Term: {info['termination_reason'] if done else 'No'}")

        self.last_commanded_clipped_sim_rad = final_clipped_commands_all_actuators.copy()
        return obs, total_reward, done, info

    def launch_viewer_internal(self):
        if self.viewer is None:
            try:
                print("Attempting to launch viewer (from main thread)...")
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                if self.viewer:
                    print("Viewer launched successfully.")
                else:
                    print("launch_passive returned None - viewer launch failed.")
            except Exception as e:
                print(f"Error launching viewer: {e}")
                self.viewer = None
        elif self.viewer and not self.viewer.is_running():
            print("Viewer was closed by user, attempting to re-launch.")
            self.viewer = None
            self.launch_viewer_internal()

    def close_viewer_internal(self):
        if self.viewer and self.viewer.is_running():
            try:
                print("Attempting to close viewer (from main thread)...")
                self.viewer.close()
                print("Viewer closed.")
            except Exception as e:
                print(f"Error closing viewer: {e}")
            finally:
                self.viewer = None
        elif self.viewer and not self.viewer.is_running(): # Already closed
            self.viewer = None


    def sync_viewer_if_active(self):
        global g_dynamic_render_active, g_render_lock
        with g_render_lock:
            if g_dynamic_render_active:
                if self.viewer is None or not self.viewer.is_running():
                    self.launch_viewer_internal()
                if self.viewer and self.viewer.is_running():
                    try:
                        self.viewer.sync()
                    except Exception as e:
                        print(f"Error syncing viewer: {e}")
                        self.close_viewer_internal()
            else: # g_dynamic_render_active is False
                if self.viewer and self.viewer.is_running():
                    self.close_viewer_internal()

def toggle_render():
    global g_dynamic_render_active, g_render_lock
    with g_render_lock:
        g_dynamic_render_active = not g_dynamic_render_active
        status = "ON" if g_dynamic_render_active else "OFF"
        print(f"\nCtrl+L pressed: Dynamic rendering toggled {status}. Viewer will update on next env sync.")
        # The actual launch/close will happen in sync_viewer_if_active on the main thread

def train():
    print("Starting RL Training with REFINED LEG POSITION PENALTY and DYNAMIC RENDER TOGGLE (Ctrl+L)...")
    global sim_keyframe_home_qpos_map, ACTUATOR_TO_JOINT_NAME_MAP, JOINT_NAME_TO_QPOS_IDX_MAP, KNEE_FLEXION_SIGN_FACTOR
    print("\n--- Verifying XML and Configuring Mappings ---")

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    base_dir_for_our_robot = os.path.dirname(script_dir)
    xml_path = os.path.join(base_dir_for_our_robot, 'our_robot', XML_FILE_NAME)

    if not os.path.exists(xml_path):
        our_robot_actual_location_checked = os.path.join(base_dir_for_our_robot, 'our_robot')
        dir_contents_str = "N/A (directory itself not found or not a dir)"
        if os.path.isdir(our_robot_actual_location_checked):
            try:
                dir_contents = os.listdir(our_robot_actual_location_checked)
                dir_contents_str = ", ".join(dir_contents) if dir_contents else "(empty)"
            except Exception as e_listdir:
                dir_contents_str = f"(Error listing contents: {e_listdir})"
        alt_xml_path_cwd_subdir = os.path.join(os.getcwd(), 'our_robot', XML_FILE_NAME)
        if os.path.exists(alt_xml_path_cwd_subdir):
            xml_path = alt_xml_path_cwd_subdir
            print(f"INFO: Found XML via CWD subdirectory: {xml_path}")
        else:
            error_message = (
                f"CRITICAL ERROR: Could not find '{XML_FILE_NAME}'.\n"
                f"  - Script is located in: '{script_dir}'\n"
                f"  - Current Working Directory (CWD) is: '{os.getcwd()}'\n"
                f"  - Primary attempt expected 'our_robot' to be a sibling of '{script_dir}'s parent, path: '{os.path.join(base_dir_for_our_robot, 'our_robot', XML_FILE_NAME)}'\n"
                f"    (Contents of '{our_robot_actual_location_checked}': [{dir_contents_str}])\n"
                f"  - Alternative attempt expected 'our_robot' to be a subdirectory of CWD, path: '{alt_xml_path_cwd_subdir}'\n"
                f"Please ensure '{XML_FILE_NAME}' exists in 'our_robot', and 'our_robot' is structured as a sibling to 'sim2real' (the script's directory), OR as a subdirectory of your current working directory."
            )
            raise FileNotFoundError(error_message)

    print(f"Loading model for verification from: {xml_path}")
    verify_model = mujoco.MjModel.from_xml_path(xml_path)
    verify_data = mujoco.MjData(verify_model)
    print("Mapping actuators to joints and qpos indices...")
    for act_name in ACTUATOR_NAMES_ORDERED:
        act_id = mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if act_id == -1: raise ValueError(f"Actuator '{act_name}' not found.")
        joint_id = verify_model.actuator_trnid[act_id, 0]
        joint_name_str = mujoco.mj_id2name(verify_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name_str is None: raise ValueError(f"No joint name for actuator '{act_name}'.")
        ACTUATOR_TO_JOINT_NAME_MAP[act_name] = joint_name_str
        qpos_idx = verify_model.jnt_qposadr[joint_id]
        JOINT_NAME_TO_QPOS_IDX_MAP[joint_name_str] = qpos_idx
        print(f"  Actuator '{act_name}' -> Joint '{joint_name_str}' -> qpos_idx {qpos_idx}")
    key_id = mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if key_id == -1: raise ValueError("Keyframe 'home' not found.")
    mujoco.mj_resetDataKeyframe(verify_model, verify_data, key_id)
    print("Extracting 'home' keyframe qpos values (sim_keyframe_home_qpos_map):")
    for act_name in ACTUATOR_NAMES_ORDERED:
        joint_name = ACTUATOR_TO_JOINT_NAME_MAP[act_name]
        qpos_idx = JOINT_NAME_TO_QPOS_IDX_MAP[joint_name]
        sim_keyframe_home_qpos_map[joint_name] = verify_data.qpos[qpos_idx]
        print(f"  Joint '{joint_name}': {sim_keyframe_home_qpos_map[joint_name]:.4f} rad")
    print("Determining knee flexion direction...")
    fr_knee_act_name = "FR_knee_actuator"
    if fr_knee_act_name not in ACTUATOR_TO_JOINT_NAME_MAP: raise ValueError(f"{fr_knee_act_name} not mapped.")
    fr_knee_joint_name = ACTUATOR_TO_JOINT_NAME_MAP[fr_knee_act_name]
    fr_knee_qpos_idx = JOINT_NAME_TO_QPOS_IDX_MAP[fr_knee_joint_name]
    fr_knee_act_id_mj = mujoco.mj_name2id(verify_model, mujoco.mjtObj.mjOBJ_ACTUATOR, fr_knee_act_name)
    mujoco.mj_resetDataKeyframe(verify_model, verify_data, key_id)
    qpos_before_perturb = verify_data.qpos[fr_knee_qpos_idx]
    current_ctrl_at_home_fr_knee = verify_data.ctrl[fr_knee_act_id_mj]
    perturb_ctrl_delta = 0.05
    ctrl_range = verify_model.actuator_ctrlrange[fr_knee_act_id_mj]
    perturbed_ctrl = np.clip(current_ctrl_at_home_fr_knee + perturb_ctrl_delta, ctrl_range[0], ctrl_range[1])
    temp_data_perturb = mujoco.MjData(verify_model)
    mujoco.mj_resetDataKeyframe(verify_model, temp_data_perturb, key_id)
    temp_data_perturb.ctrl[fr_knee_act_id_mj] = perturbed_ctrl
    mujoco.mj_step(verify_model, temp_data_perturb)
    qpos_after_perturb = temp_data_perturb.qpos[fr_knee_qpos_idx]
    delta_qpos = qpos_after_perturb - qpos_before_perturb
    if abs(delta_qpos) < 1e-5: KNEE_FLEXION_SIGN_FACTOR = 1.0; print(f"Warning: Knee qpos barely changed. Default KNEE_FLEXION_SIGN_FACTOR=1.0")
    elif np.sign(delta_qpos) == np.sign(KNEE_SWING_FLEXION_RAD if KNEE_SWING_FLEXION_RAD !=0 else 1.0): KNEE_FLEXION_SIGN_FACTOR = 1.0
    else: KNEE_FLEXION_SIGN_FACTOR = -1.0
    print(f"  Knee '{fr_knee_joint_name}' qpos change: {delta_qpos:.4f}. Target KNEE_SWING_FLEXION_RAD sign: {np.sign(KNEE_SWING_FLEXION_RAD if KNEE_SWING_FLEXION_RAD !=0 else 1.0):.0f}. KNEE_FLEXION_SIGN_FACTOR = {KNEE_FLEXION_SIGN_FACTOR}")
    print(f"--- Verification Complete ---")

    env = QuadrupedEnv(xml_path)

    if KEYBOARD_LIB_AVAILABLE:
        try:
            keyboard.add_hotkey('ctrl+l', toggle_render)
            print("INFO: Ctrl+L hotkey for render toggle is active.")
        except Exception as e: # Catch broader exceptions if keyboard lib has issues
            print(f"WARNING: Failed to set Ctrl+L hotkey: {e}. Dynamic rendering may not work.")
            if "try running with sudo" in str(e).lower():
                 print("       On Linux, you might need to run the script with 'sudo python ...'")

    state_dim = env.state_dim; action_dim = env.action_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}. State_dim: {state_dim}, Action_dim: {action_dim}")
    current_lr = INITIAL_LEARNING_RATE; current_entropy_coef = INITIAL_ENTROPY_COEF
    agent = ActorCritic(state_dim, action_dim, action_std_init=INITIAL_ACTION_STD_INIT).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=current_lr)
    states_mem, actions_mem, log_probs_mem, rewards_mem, values_mem, masks_mem = [], [], [], [], [], []
    episode_rewards = deque(maxlen=100); avg_reward_history = deque(maxlen=AVGR_HISTORY_LEN); total_env_steps = 0

    try:
        for episode in range(1, NUM_EPISODES + 1):
            state = env.reset(); current_episode_reward = 0; termination_reason_ep = "max_steps"
            for step_num in range(MAX_STEPS_PER_EPISODE):
                total_env_steps += 1
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action_dist, value = agent(state_tensor)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action).sum(axis=-1)
                action_np = action.squeeze(0).cpu().numpy()
                next_state, reward, done, info = env.step(action_np)
                states_mem.append(state); actions_mem.append(action_np);
                log_probs_mem.append(log_prob.item())
                rewards_mem.append(reward); values_mem.append(value.item()); masks_mem.append(1-done)
                state = next_state; current_episode_reward += reward
                if done:
                    termination_reason_ep = info.get("termination_reason", "unknown")
                    break
            episode_rewards.append(current_episode_reward)
            current_avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            avg_reward_history.append(current_avg_reward)
            print(f"Ep: {episode}, Steps: {step_num+1}, R: {current_episode_reward:.2f}, AvgR: {current_avg_reward:.2f} (Term: {termination_reason_ep}), LR: {current_lr:.1e}, EntC: {current_entropy_coef:.4f}, ActStd: {torch.exp(agent.action_log_std.mean()).item():.3f}")
            if len(states_mem) >= POLICY_UPDATE_INTERVAL:
                returns, advantages = [], []
                last_value = 0.0
                if not done:
                    with torch.no_grad():
                        _, last_value_tensor = agent(torch.FloatTensor(next_state).unsqueeze(0).to(device))
                        last_value = last_value_tensor.item()
                gae = 0.0
                for i in reversed(range(len(rewards_mem))):
                    next_mask_idx = i + 1 if i + 1 < len(masks_mem) else i; next_mask = masks_mem[next_mask_idx]
                    next_val_if_not_terminal = values_mem[i+1] if i + 1 < len(values_mem) else last_value
                    bootstrap_val = next_val_if_not_terminal * next_mask
                    delta = rewards_mem[i] + GAMMA * bootstrap_val - values_mem[i]
                    gae = delta + GAMMA * GAE_LAMBDA * masks_mem[i] * gae
                    returns.insert(0, gae + values_mem[i]); advantages.insert(0, gae)
                states_tensor = torch.FloatTensor(np.array(states_mem)).to(device)
                actions_tensor = torch.FloatTensor(np.array(actions_mem)).to(device)
                returns_tensor = torch.FloatTensor(returns).to(device).unsqueeze(1)
                advantages_tensor = torch.FloatTensor(advantages).to(device)
                advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
                for _ in range(NUM_EPOCHS_PER_UPDATE):
                    action_dist_new, values_new = agent(states_tensor)
                    log_probs_new = action_dist_new.log_prob(actions_tensor).sum(axis=-1)
                    entropy = action_dist_new.entropy().mean()
                    actor_loss = -(log_probs_new * advantages_tensor).mean()
                    critic_loss = nn.MSELoss()(values_new, returns_tensor)
                    loss = actor_loss + VALUE_LOSS_COEF * critic_loss - current_entropy_coef * entropy
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                states_mem, actions_mem, log_probs_mem, rewards_mem, values_mem, masks_mem = [], [], [], [], [], []
            if episode % ADAPTATION_CHECK_INTERVAL == 0 and len(avg_reward_history) >= AVGR_HISTORY_LEN:
                first_half_avg = np.mean(list(avg_reward_history)[:AVGR_HISTORY_LEN//2]); second_half_avg = np.mean(list(avg_reward_history)[AVGR_HISTORY_LEN//2:])
                trend_diff = second_half_avg - first_half_avg
                significant_change_threshold = 0.10 * abs(current_avg_reward) if abs(current_avg_reward) > 20 else 2.0
                print(f"  Adapting Hypers: TrendDiff={trend_diff:.2f}, CurrentAvgR={current_avg_reward:.2f}, Thresh={significant_change_threshold:.2f}")
                if trend_diff < -significant_change_threshold :
                    print("    Trend: WORSENING..."); current_lr = max(MIN_LR, current_lr * 0.75); current_entropy_coef = max(MIN_ENTROPY_COEF, current_entropy_coef * 0.9)
                    with torch.no_grad(): agent.action_log_std.data = torch.clamp(agent.action_log_std.data - math.log(1.05), MIN_ACTION_LOG_STD, MAX_ACTION_LOG_STD)
                elif abs(trend_diff) < significant_change_threshold * 0.3 :
                    print("    Trend: STAGNANT..."); current_entropy_coef = min(MAX_ENTROPY_COEF, current_entropy_coef * 1.05)
                    with torch.no_grad(): agent.action_log_std.data = torch.clamp(agent.action_log_std.data + math.log(1.03), MIN_ACTION_LOG_STD, MAX_ACTION_LOG_STD)
                    if current_lr < MAX_LR * 0.1 : current_lr = min(MAX_LR, current_lr * 1.05)
                elif trend_diff > significant_change_threshold:
                    print("    Trend: IMPROVING...")
                    if current_lr > MIN_LR * 5: current_lr = max(MIN_LR, current_lr * 0.95)
                for param_group in optimizer.param_groups: param_group['lr'] = current_lr
                print(f"    New LR: {current_lr:.1e}, New EntC: {current_entropy_coef:.4f}, New ActStd: {torch.exp(agent.action_log_std.mean()).item():.3f}")
            if episode % PTH_SAVE_INTERVAL == 0:
                 pth_path = f'quadruped_ac_sym_ep{episode}.pth'
                 torch.save(agent.state_dict(), pth_path); print(f"Model saved to {pth_path}")
                 json_path = JSON_OUTPUT_PATH_TEMPLATE.format(episode)
                 generate_walk_json(agent, env, device, json_output_path=json_path, num_steps=JSON_MAX_STEPS_EPISODIC)
    except KeyboardInterrupt:
        print("Training interrupted by user (Ctrl+C).")
    finally:
        print("Training finished or interrupted.")
        if KEYBOARD_LIB_AVAILABLE:
            try: keyboard.remove_hotkey('ctrl+l'); print("INFO: Ctrl+L hotkey removed.")
            except Exception as e: print(f"WARNING: Could not remove hotkey: {e}")
        final_pth_path = 'quadruped_ac_sym_final.pth'
        torch.save(agent.state_dict(), final_pth_path); print(f"Final model saved as '{final_pth_path}'")
        final_json_path = JSON_OUTPUT_PATH_TEMPLATE.format('final')
        generate_walk_json(agent, env, device, json_output_path=final_json_path, num_steps=JSON_MAX_STEPS_FINAL)
        if env.viewer:
            env.close_viewer_internal() # Use internal close

def generate_walk_json(agent, env, device, json_output_path, num_steps):
    print(f"\nGenerating walk sequence ({num_steps} steps) to {json_output_path}...")
    global g_dynamic_render_active
    if g_dynamic_render_active:
        # env.sync_viewer_if_active() will handle launching if needed during env.reset/step
        print("INFO: Dynamic rendering is ON. Viewer should appear during JSON generation steps if not already open.")
    else:
        print("INFO: Dynamic rendering is OFF. JSON generation will be headless.")

    real_robot_sequence_for_json = []
    state = env.reset() # This calls sync_viewer_if_active
    for step_count in range(num_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_dist, _ = agent(state_tensor)
            action_mean = action_dist.mean
            action_np_symmetric = action_mean.squeeze(0).cpu().numpy()
        next_state, _, done, info = env.step(action_np_symmetric) # This calls sync_viewer_if_active
        sim_targets_rad_all_clipped = info.get("sim_target_rad")
        if sim_targets_rad_all_clipped is None:
            print(f"Warning: sim_target_rad not in info on step {step_count}. Skipping."); continue
        targets_real_deg = {}
        for i, name_ordered in enumerate(ACTUATOR_NAMES_ORDERED):
            mujoco_act_id = actuator_ctrl_props[name_ordered]['mujoco_id']
            real_deg = convert_sim_rad_to_real_deg(name_ordered,
                                                   sim_targets_rad_all_clipped[mujoco_act_id],
                                                   sim_keyframe_home_qpos_map,
                                                   real_robot_home_deg_map,
                                                   joint_scale_factors)
            targets_real_deg[name_ordered] = round(real_deg, 2) if real_deg is not None else 0.0
        real_robot_sequence_for_json.append({"duration": round(POLICY_DECISION_DT, 3), "targets_deg": targets_real_deg})
        state = next_state
        if done:
            print(f"Episode ended prematurely during JSON gen at step {step_count+1} (Term: {info.get('termination_reason', 'unknown')}).")
            break
        # No need for explicit check of viewer.is_running() here, sync_viewer_if_active handles it
    if not real_robot_sequence_for_json: print(f"Warning: No steps for JSON {json_output_path}."); return
    try:
        with open(json_output_path, 'w') as f: json.dump(real_robot_sequence_for_json, f, indent=2)
        print(f"Successfully saved {len(real_robot_sequence_for_json)} steps to {json_output_path}")
    except IOError as e: print(f"ERROR writing JSON to {json_output_path}: {e}")

if __name__ == "__main__":
    train()