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


# --- Output Paths ---
OUTPUT_BASE_DIR = './output'
JSON_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'json')
PTH_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'pth')
JSON_OUTPUT_PATH_TEMPLATE = os.path.join(JSON_OUTPUT_DIR, 'walk_rl_sym_ep{}.json')
PTH_SAVE_TEMPLATE = os.path.join(PTH_OUTPUT_DIR, 'quadruped_ac_sym_ep{}.pth')
FINAL_PTH_SAVE_NAME = os.path.join(PTH_OUTPUT_DIR, 'quadruped_ac_sym_final.pth')


PTH_SAVE_INTERVAL = 100
JSON_MAX_STEPS_EPISODIC = 50
JSON_MAX_STEPS_FINAL = 100
XML_FILE_NAME = 'walking_scene.xml'

# --- ADAPTIVE HYPERPARAMETER CONFIG ---
ADAPTATION_CHECK_INTERVAL = 10
AVGR_HISTORY_LEN = 5
MIN_LR = 1e-6; MAX_LR = 3e-4
MIN_ENTROPY_COEF = 0.0001; MAX_ENTROPY_COEF = 0.01
MIN_ACTION_LOG_STD = math.log(0.10); MAX_ACTION_LOG_STD = math.log(0.5)
INITIAL_LEARNING_RATE = 1e-4
INITIAL_ENTROPY_COEF = 0.002
INITIAL_ACTION_STD_INIT = 0.4

# RL Hyperparameters (others)
GAMMA = 0.99; GAE_LAMBDA = 0.95
VALUE_LOSS_COEF = 0.5; MAX_GRAD_NORM = 0.5; NUM_EPISODES = 10000 # Or less for faster testing
MAX_STEPS_PER_EPISODE = 250
POLICY_UPDATE_INTERVAL = 2048
NUM_EPOCHS_PER_UPDATE = 10

# Refined Leg Position Penalty Hyperparameters
LEG_AT_HOME_THRESHOLD_DEG = 5.0
MOVING_LEG_MAX_DEVIATION_DEG = 40.0
LEG_POSITIONING_PENALTY = 0.5 # Per leg not meeting its phase-specific criteria

# Action scaling
ACTION_AMPLITUDE_DEG = 40.0
ACTION_AMPLITUDE_RAD = math.radians(ACTION_AMPLITUDE_DEG)

# Environment parameters
ORIENTATION_TERMINATION_LIMIT_DEG = 25.0
ORIENTATION_TERMINATION_LIMIT_RAD = math.radians(ORIENTATION_TERMINATION_LIMIT_DEG)
ORIENTATION_PENALTY_THRESHOLD_DEG = 5.0
ORIENTATION_PENALTY_THRESHOLD_RAD = math.radians(ORIENTATION_PENALTY_THRESHOLD_DEG)
YAW_PENALTY_THRESHOLD_DEG = 10.0
YAW_PENALTY_THRESHOLD_RAD = math.radians(YAW_PENALTY_THRESHOLD_DEG)

MIN_FWD_DISPLACEMENT_FOR_BACKWARD_CHECK = 0.05
POLICY_DECISION_DT = 0.10;
NUM_SETTLE_STEPS = 100;
PHASE_CYCLE_DURATION_POLICY_STEPS = 2

# --- SIM-TO-REAL MAPPING CONFIGURATION ---
real_robot_home_deg_map = {name: 0.0 for name in ACTUATOR_NAMES_ORDERED}
real_robot_home_deg_map.update({
    "FR_tigh_actuator": -45.0, "FR_knee_actuator": 45.0,
    "FL_tigh_actuator": 45.0,  "FL_knee_actuator": 45.0,
    "BR_tigh_actuator": 45.0,  "BR_knee_actuator": -45.0,
    "BL_tigh_actuator": 45.0,  "BL_knee_actuator": -45.0,
})
joint_scale_factors = {name: 1.0 for name in ACTUATOR_NAMES_ORDERED}
sim_keyframe_home_qpos_map = {}
actuator_ctrl_props = {}

# --- Global Render Toggle ---
g_dynamic_render_active = False
g_render_lock = threading.Lock()

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
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * math.log(action_std_init))
    def forward(self, state):
        action_mean = self.actor(state)
        action_std = torch.exp(self.action_log_std.expand_as(action_mean))
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
            actuator_ctrl_props[name] = {'ctrlrange': ctrlrange.copy(), 'mujoco_id': mujoco_act_id}

        self.state_dim = 3 + 8 + 8 + 1 + 2
        self.action_dim = 4
        
        temp_model_for_init = mujoco.MjModel.from_xml_path(model_path)
        temp_data_for_init = mujoco.MjData(temp_model_for_init)
        key_id_for_init = mujoco.mj_name2id(temp_model_for_init, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if key_id_for_init == -1: raise ValueError("Keyframe 'home' not found.")
        mujoco.mj_resetDataKeyframe(temp_model_for_init, temp_data_for_init, key_id_for_init)
        
        self.initial_qpos = temp_data_for_init.qpos.copy()
        self.initial_qvel = temp_data_for_init.qvel.copy()
        self.initial_ctrl = temp_data_for_init.ctrl.copy()
        self.initial_body_y_pos = self.initial_qpos[1]

        self.episode_policy_step_counter = 0 # This counter tracks steps *within* an episode
        self.last_commanded_clipped_sim_rad = self.initial_ctrl.copy()
        self.previous_x_qpos_in_episode = 0.0
        self.cumulative_positive_x_displacement = 0.0; self.cumulative_negative_x_displacement = 0.0
        self.previous_net_forward_displacement = 0.0

    def _get_observation(self):
        yaw, pitch, roll = quat_to_ypr(self.data.qpos[3:7])
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
        trunk_fwd_vel_global_x = self.data.qvel[0]
        
        # Use the current step counter to determine the phase for the *current* observation
        # This phase info is for the *next* action selection
        current_phase_for_obs = self.episode_policy_step_counter % PHASE_CYCLE_DURATION_POLICY_STEPS
        phase_progress_norm = current_phase_for_obs / (PHASE_CYCLE_DURATION_POLICY_STEPS -1 ) if PHASE_CYCLE_DURATION_POLICY_STEPS >1 else 0.0
        sin_phase = np.sin(phase_progress_norm * np.pi) 
        cos_phase = np.cos(phase_progress_norm * np.pi)

        state = np.concatenate([[yaw, pitch, roll], joint_pos_dev, joint_vel,
                                [trunk_fwd_vel_global_x], [sin_phase, cos_phase]])
        return state.astype(np.float32)

    def reset_to_home_keyframe(self): # Used by perturbation test primarily
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            self.data.ctrl[:] = self.initial_ctrl
            mujoco.mj_forward(self.model, self.data)
        else:
            self.data.qpos[:] = self.initial_qpos; self.data.qvel[:] = self.initial_qvel
            self.data.ctrl[:] = self.initial_ctrl; mujoco.mj_forward(self.model, self.data)
        for _ in range(NUM_SETTLE_STEPS):
            self.data.ctrl[:] = self.initial_ctrl
            try: mujoco.mj_step(self.model, self.data)
            except mujoco.FatalError: break
            self.sync_viewer_if_active()
        self.last_commanded_clipped_sim_rad = self.data.ctrl.copy()
        # For perturbation test, the episode_policy_step_counter might be set externally before calling this
        return self._get_observation()

    def reset(self): # For RL training
        self.episode_policy_step_counter = 0 # Reset for the new episode *before* getting first obs
        self.reset_to_home_keyframe() # This will use counter = 0 for its _get_observation()
        self.previous_x_qpos_in_episode = self.data.qpos[0]
        self.cumulative_positive_x_displacement = 0.0; self.cumulative_negative_x_displacement = 0.0
        self.previous_net_forward_displacement = 0.0
        return self._get_observation() # Return obs for counter = 0

    def _apply_actions_and_step(self, 
                                fr_tigh_policy_delta_rad,
                                knee_pair1_swing_delta_rad, 
                                fl_tigh_policy_delta_rad,
                                knee_pair2_swing_delta_rad, 
                                current_phase_idx_for_action): # Phase that *this action* corresponds to
        final_clipped_commands_all_actuators = np.zeros_like(self.data.ctrl)

        act_fr_tigh_delta = fr_tigh_policy_delta_rad
        act_fl_tigh_delta = fl_tigh_policy_delta_rad
        act_bl_tigh_delta = fr_tigh_policy_delta_rad
        act_br_tigh_delta = fl_tigh_policy_delta_rad

        act_fr_knee_delta, act_fl_knee_delta, act_bl_knee_delta, act_br_knee_delta = 0.0, 0.0, 0.0, 0.0

        if current_phase_idx_for_action == 0:  # FR/BL swing, FL/BR stance
            act_fr_knee_delta = knee_pair1_swing_delta_rad
            act_bl_knee_delta = -knee_pair1_swing_delta_rad
            act_fl_knee_delta = 0.0 
            act_br_knee_delta = 0.0 
        elif current_phase_idx_for_action == 1:  # FL/BR swing, FR/BL stance
            act_fl_knee_delta = knee_pair2_swing_delta_rad
            act_br_knee_delta = -knee_pair2_swing_delta_rad
            act_fr_knee_delta = 0.0 
            act_bl_knee_delta = 0.0 
            
        actuator_deltas_rad = {
            "FR_tigh_actuator": act_fr_tigh_delta, "FR_knee_actuator": act_fr_knee_delta,
            "FL_tigh_actuator": act_fl_tigh_delta, "FL_knee_actuator": act_fl_knee_delta,
            "BR_tigh_actuator": act_br_tigh_delta, "BL_tigh_actuator": act_bl_tigh_delta,
            "BR_knee_actuator": act_br_knee_delta, "BL_knee_actuator": act_bl_knee_delta,
        }
        # Debug print for _apply_actions_and_step
        # print(f"DEBUG _apply: Phase {current_phase_idx_for_action}, PolicyKnee1 {knee_pair1_swing_delta_rad:.2f}, PolicyKnee2 {knee_pair2_swing_delta_rad:.2f}")
        # print(f"DEBUG _apply: Applied Deltas Rad: FRK:{act_fr_knee_delta:.2f} FLK:{act_fl_knee_delta:.2f} BLK:{act_bl_knee_delta:.2f} BRK:{act_br_knee_delta:.2f}")

        for act_name in ACTUATOR_NAMES_ORDERED:
            props=actuator_ctrl_props[act_name]; mujoco_act_id=props['mujoco_id']
            joint_name=ACTUATOR_TO_JOINT_NAME_MAP[act_name]; sim_home_rad=sim_keyframe_home_qpos_map[joint_name]
            applied_delta_rad = actuator_deltas_rad[act_name]
            sim_target_rad_unclipped = sim_home_rad + applied_delta_rad
            clipped_sim_target_rad = np.clip(sim_target_rad_unclipped, props['ctrlrange'][0], props['ctrlrange'][1])
            final_clipped_commands_all_actuators[mujoco_act_id] = clipped_sim_target_rad
            
        self.data.ctrl[:] = final_clipped_commands_all_actuators
        sum_mj_errors = 0
        for _ in range(self.sim_steps_per_policy_step):
            try: mujoco.mj_step(self.model, self.data)
            except mujoco.FatalError: sum_mj_errors +=1; break
            self.sync_viewer_if_active()
        return final_clipped_commands_all_actuators, sum_mj_errors

    def step(self, policy_actions_scaled_neg1_to_1):
        # Determine the phase for the action being taken *now*
        # This phase was observed at the *end* of the previous step (or at reset)
        current_phase_idx_for_action = self.episode_policy_step_counter % PHASE_CYCLE_DURATION_POLICY_STEPS

        fr_tigh_delta_rad = policy_actions_scaled_neg1_to_1[0] * ACTION_AMPLITUDE_RAD
        knee_pair1_swing_delta_rad = policy_actions_scaled_neg1_to_1[1] * ACTION_AMPLITUDE_RAD
        fl_tigh_delta_rad = policy_actions_scaled_neg1_to_1[2] * ACTION_AMPLITUDE_RAD
        knee_pair2_swing_delta_rad = policy_actions_scaled_neg1_to_1[3] * ACTION_AMPLITUDE_RAD

        final_clipped_commands_all_actuators, sum_mj_errors = self._apply_actions_and_step(
            fr_tigh_delta_rad, knee_pair1_swing_delta_rad, 
            fl_tigh_delta_rad, knee_pair2_swing_delta_rad,
            current_phase_idx_for_action # Use phase corresponding to current step counter
        )
        
        self.episode_policy_step_counter += 1 # Increment counter *after* using it for current action's phase

        current_x_pos_after_sim_loop = self.data.qpos[0]
        delta_x_this_policy_step = current_x_pos_after_sim_loop - self.previous_x_qpos_in_episode
        if delta_x_this_policy_step > 0: self.cumulative_positive_x_displacement += delta_x_this_policy_step
        elif delta_x_this_policy_step < 0: self.cumulative_negative_x_displacement += abs(delta_x_this_policy_step)
        self.previous_x_qpos_in_episode = current_x_pos_after_sim_loop
        
        obs = self._get_observation() # Observation will reflect state *after* action, and phase for *next* step

        # --- Rewards ---
        forward_vel_global_x = self.data.qvel[0]
        reward_forward_velocity = 150.0 * forward_vel_global_x 
        current_net_forward_displacement = self.cumulative_positive_x_displacement - self.cumulative_negative_x_displacement
        delta_net_forward_displacement_this_step = current_net_forward_displacement - self.previous_net_forward_displacement
        self.previous_net_forward_displacement = current_net_forward_displacement
        reward_new_cumulative_progress = 0.0
        if delta_net_forward_displacement_this_step > 0.0005:
             reward_new_cumulative_progress = 15.0 * delta_net_forward_displacement_this_step
        reward_backward_velocity_penalty = 0.0
        if forward_vel_global_x < -0.005:
            reward_backward_velocity_penalty = -5.0 * abs(forward_vel_global_x) 
        reward_alive = 0.05
        reward_sideways_vel = -0.2 * abs(self.data.qvel[1]) 
        reward_y_pos_stability = -0.1 * abs(self.data.qpos[1] - self.initial_body_y_pos) 
        current_yaw, current_pitch, current_roll = quat_to_ypr(self.data.qpos[3:7])
        reward_excessive_orientation = 0.0
        orientation_penalty_factor = -0.05 
        if abs(current_roll) > ORIENTATION_PENALTY_THRESHOLD_RAD:
            reward_excessive_orientation += orientation_penalty_factor * (abs(current_roll) - ORIENTATION_PENALTY_THRESHOLD_RAD)**2
        if abs(current_pitch) > ORIENTATION_PENALTY_THRESHOLD_RAD:
            reward_excessive_orientation += orientation_penalty_factor * (abs(current_pitch) - ORIENTATION_PENALTY_THRESHOLD_RAD)**2
        if abs(current_yaw) > YAW_PENALTY_THRESHOLD_RAD:
            reward_excessive_orientation += orientation_penalty_factor * (abs(current_yaw) - YAW_PENALTY_THRESHOLD_RAD)**2
        action_delta_sq_sum = np.sum([(final_clipped_commands_all_actuators[actuator_ctrl_props[name]['mujoco_id']] -
                                     self.last_commanded_clipped_sim_rad[actuator_ctrl_props[name]['mujoco_id']])**2
                                     for name in ACTUATOR_NAMES_ORDERED])
        reward_action_smoothness = -0.01 * action_delta_sq_sum 
        
        reward_leg_positioning = 0.0
        legs_at_home_status = {} # "at_home", "moving_ok", "moving_too_far" (based on real delta from real home)
        
        is_fr_bl_swing_phase = current_phase_idx_for_action == 0
        num_swing_legs_too_far = 0
        num_stance_legs_not_home = 0

        for leg_id, leg_act_names_in_leg in LEG_ACTUATORS.items():
            is_this_leg_swinging = (is_fr_bl_swing_phase and leg_id in ["FR", "BL"]) or \
                                   (not is_fr_bl_swing_phase and leg_id in ["FL", "BR"])
            
            leg_joints_max_dev_from_real_home_deg = 0.0
            all_joints_in_leg_at_real_home = True

            for act_name_iter in leg_act_names_in_leg:
                mujoco_act_id = actuator_ctrl_props[act_name_iter]['mujoco_id']
                current_sim_cmd_rad = final_clipped_commands_all_actuators[mujoco_act_id]
                
                # Calculate real delta from real home
                real_target_deg_val = convert_sim_rad_to_real_deg(
                    act_name_iter, current_sim_cmd_rad, sim_keyframe_home_qpos_map,
                    real_robot_home_deg_map, joint_scale_factors
                )
                if real_target_deg_val is None: real_target_deg_val = real_robot_home_deg_map[act_name_iter] # Fallback
                
                real_delta_from_real_home_deg = abs(real_target_deg_val - real_robot_home_deg_map[act_name_iter])
                leg_joints_max_dev_from_real_home_deg = max(leg_joints_max_dev_from_real_home_deg, real_delta_from_real_home_deg)
                if real_delta_from_real_home_deg > LEG_AT_HOME_THRESHOLD_DEG:
                    all_joints_in_leg_at_real_home = False
            
            if is_this_leg_swinging:
                if leg_joints_max_dev_from_real_home_deg > MOVING_LEG_MAX_DEVIATION_DEG:
                    legs_at_home_status[leg_id] = "swing_too_far"
                    num_swing_legs_too_far +=1
                else:
                    legs_at_home_status[leg_id] = "swing_ok"
            else: # Stance leg
                if not all_joints_in_leg_at_real_home: # Check if any joint in stance leg deviated
                    legs_at_home_status[leg_id] = "stance_off_home"
                    num_stance_legs_not_home +=1
                else:
                    legs_at_home_status[leg_id] = "stance_at_home"
        
        if num_swing_legs_too_far > 0 or num_stance_legs_not_home > 0 :
            reward_leg_positioning = -(num_swing_legs_too_far + num_stance_legs_not_home) * LEG_POSITIONING_PENALTY


        total_reward = (reward_forward_velocity + reward_new_cumulative_progress +
                        reward_backward_velocity_penalty + reward_alive  +
                        reward_sideways_vel + reward_y_pos_stability +
                        reward_excessive_orientation + reward_action_smoothness + reward_leg_positioning)
        done = False; info = {"sim_target_rad": final_clipped_commands_all_actuators.copy(), "termination_reason": "max_steps"}
        if sum_mj_errors > 0:
            total_reward -= 20.0; done = True; info["mj_error"] = True; info["termination_reason"] = "mj_error"
        if abs(current_roll) > ORIENTATION_TERMINATION_LIMIT_RAD or \
           abs(current_pitch) > ORIENTATION_TERMINATION_LIMIT_RAD or \
           abs(current_yaw) > ORIENTATION_TERMINATION_LIMIT_RAD:
            total_reward -= 5.0; done = True; info["termination_reason"] = "orientation_limit"
        if not done and self.cumulative_positive_x_displacement > MIN_FWD_DISPLACEMENT_FOR_BACKWARD_CHECK and \
           self.cumulative_negative_x_displacement > 0.75 * self.cumulative_positive_x_displacement:
            total_reward -= 5.0; done = True; info["termination_reason"] = "too_much_backward"
        print_debug = False
        if g_dynamic_render_active and (self.episode_policy_step_counter%5==0 or done): print_debug=True
        elif self.episode_policy_step_counter%20==0 or self.episode_policy_step_counter==MAX_STEPS_PER_EPISODE or done : print_debug=True # MAX_STEPS_PER_EPISODE (not -1)
        if print_debug:
            status_str = ", ".join([f"{k}: {v}" for k,v in legs_at_home_status.items()])
            phase_str = "FR/BL_sw" if current_phase_idx_for_action == 0 else "FL/BR_sw"
            print(f"  DBG S{self.episode_policy_step_counter-1} Ph:{phase_str}: TotR={total_reward:.2f} || FwdVel={reward_forward_velocity:.2f} ({forward_vel_global_x:.3f}) | NewCumProg={reward_new_cumulative_progress:.2f} | BwdVelP={reward_backward_velocity_penalty:.2f} | Alive={reward_alive:.2f} | SideVelP={reward_sideways_vel:.2f} | YPosStabP={reward_y_pos_stability:.2f} (Y:{self.data.qpos[1]:.2f}) | OrientP={reward_excessive_orientation:.2f} (Y:{math.degrees(current_yaw):.1f} R:{math.degrees(current_roll):.1f} P:{math.degrees(current_pitch):.1f}) | SmoothP={reward_action_smoothness:.2f} | LegPosP={reward_leg_positioning:.2f} [{status_str}] || Term: {info['termination_reason'] if done else 'No'}")
        self.last_commanded_clipped_sim_rad = final_clipped_commands_all_actuators.copy()
        return obs, total_reward, done, info

    def launch_viewer_internal(self):
        if self.viewer is None:
            try: print("Attempting to launch viewer..."); self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as e: print(f"Error launching viewer: {e}"); self.viewer = None
        elif self.viewer and not self.viewer.is_running():
            print("Viewer closed, re-launching."); self.viewer = None; self.launch_viewer_internal()
    def close_viewer_internal(self):
        if self.viewer and self.viewer.is_running():
            try: print("Attempting to close viewer..."); self.viewer.close()
            except Exception as e: print(f"Error closing viewer: {e}")
            finally: self.viewer = None
    def sync_viewer_if_active(self):
        global g_dynamic_render_active, g_render_lock
        with g_render_lock:
            if g_dynamic_render_active:
                if self.viewer is None or not self.viewer.is_running(): self.launch_viewer_internal()
                if self.viewer and self.viewer.is_running():
                    try: self.viewer.sync()
                    except Exception as e: print(f"Error syncing: {e}"); self.close_viewer_internal()
            elif self.viewer and self.viewer.is_running(): self.close_viewer_internal()

def toggle_render():
    global g_dynamic_render_active, g_render_lock
    with g_render_lock: g_dynamic_render_active = not g_dynamic_render_active
    print(f"\nDynamic rendering toggled {'ON' if g_dynamic_render_active else 'OFF'}.")

def run_actuator_perturbation_test(env, test_delta_deg=15.0):
    print("\n--- Running Actuator Perturbation Test (Automatic) ---")
    test_delta_rad = math.radians(test_delta_deg)
    global g_dynamic_render_active 
    original_render_state = g_dynamic_render_active
    g_dynamic_render_active = True 
    env.sync_viewer_if_active()
    action_interpretation_map = {
        0: "FR_tigh_delta", 1: "Knee_P1(FR/BL)_sw_delta",
        2: "FL_tigh_delta", 3: "Knee_P2(FL/BR)_sw_delta"
    }
    for policy_action_idx_to_perturb in range(env.action_dim):
        for sign in [1, -1]:
            for phase_to_test in range(PHASE_CYCLE_DURATION_POLICY_STEPS):
                phase_str = "FR/BL_swing" if phase_to_test == 0 else "FL/BR_swing"
                print(f"\nPerturbing: {action_interpretation_map[policy_action_idx_to_perturb]} by {sign*test_delta_deg:.1f} deg | For Action Phase: {phase_str} ({phase_to_test})")
                env.reset_to_home_keyframe()
                # env.episode_policy_step_counter = phase_to_test # Not needed if _apply takes phase directly
                time.sleep(0.05) 
                base_policy_outputs_rad = [0.0] * env.action_dim 
                perturbed_policy_outputs_rad = list(base_policy_outputs_rad)
                perturbed_policy_outputs_rad[policy_action_idx_to_perturb] = sign * test_delta_rad
                sim_commands_rad, _ = env._apply_actions_and_step(
                    fr_tigh_policy_delta_rad    = perturbed_policy_outputs_rad[0],
                    knee_pair1_swing_delta_rad  = perturbed_policy_outputs_rad[1],
                    fl_tigh_policy_delta_rad    = perturbed_policy_outputs_rad[2],
                    knee_pair2_swing_delta_rad  = perturbed_policy_outputs_rad[3],
                    current_phase_idx_for_action= phase_to_test # Explicitly pass the phase being tested
                )
                time.sleep(0.1) 
                print("  Resulting Real-World Target Degrees:")
                print("  Actuator          |SimKeyHome|RealHome|AppliedSimDelta|SimTarget|RealTarget|RealDelta")
                print("                    |(rad)     |(deg)   |(rad)          |(rad)    |(deg)     |(deg)")
                eff_fr_t_delta=perturbed_policy_outputs_rad[0]; eff_fl_t_delta=perturbed_policy_outputs_rad[2]
                eff_bl_t_delta=perturbed_policy_outputs_rad[0]; eff_br_t_delta=perturbed_policy_outputs_rad[2]
                eff_fr_k_delta,eff_bl_k_delta,eff_fl_k_delta,eff_br_k_delta = 0,0,0,0
                if phase_to_test == 0: # FR/BL swing
                    eff_fr_k_delta=perturbed_policy_outputs_rad[1]; eff_bl_k_delta=-perturbed_policy_outputs_rad[1]
                elif phase_to_test == 1: # FL/BR swing
                    eff_fl_k_delta=perturbed_policy_outputs_rad[3]; eff_br_k_delta=-perturbed_policy_outputs_rad[3]
                effective_deltas_map = {
                    "FR_tigh_actuator":eff_fr_t_delta, "FR_knee_actuator":eff_fr_k_delta,
                    "FL_tigh_actuator":eff_fl_t_delta, "FL_knee_actuator":eff_fl_k_delta,
                    "BL_tigh_actuator":eff_bl_t_delta, "BL_knee_actuator":eff_bl_k_delta,
                    "BR_tigh_actuator":eff_br_t_delta, "BR_knee_actuator":eff_br_k_delta,
                }
                for act_name in ACTUATOR_NAMES_ORDERED:
                    mj_id=actuator_ctrl_props[act_name]['mujoco_id']; sim_target_val=sim_commands_rad[mj_id]
                    real_target_val=convert_sim_rad_to_real_deg(act_name,sim_target_val,sim_keyframe_home_qpos_map,real_robot_home_deg_map,joint_scale_factors)
                    j_name=ACTUATOR_TO_JOINT_NAME_MAP[act_name]; sim_h_val=sim_keyframe_home_qpos_map[j_name]; real_h_val=real_robot_home_deg_map[act_name]
                    applied_sim_delta_for_print=effective_deltas_map[act_name]
                    real_d_val=real_target_val-real_h_val if real_target_val is not None else "N/A"
                    real_target_str=f"{real_target_val:6.1f}" if real_target_val is not None else " N/A  "
                    real_d_str=f"{real_d_val:6.1f}" if isinstance(real_d_val,(float,int)) else " N/A  "
                    print(f"    {act_name:<18}: {sim_h_val:6.2f} | {real_h_val:6.1f} | {applied_sim_delta_for_print:13.2f} | {sim_target_val:6.2f} | {real_target_str} | {real_d_str}")
                time.sleep(0.05) 
    print("--- Perturbation Test Complete ---")
    g_dynamic_render_active = original_render_state; env.sync_viewer_if_active()

def train():
    print("Starting RL Training...")
    global sim_keyframe_home_qpos_map, ACTUATOR_TO_JOINT_NAME_MAP, JOINT_NAME_TO_QPOS_IDX_MAP
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True); os.makedirs(PTH_OUTPUT_DIR, exist_ok=True)
    print(f"JSON outputs: {JSON_OUTPUT_DIR}, PTH outputs: {PTH_OUTPUT_DIR}")
    print("\n--- Verifying XML & Mappings ---")
    try: script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: script_dir = os.getcwd()
    xml_path = os.path.join(os.path.dirname(script_dir), 'our_robot', XML_FILE_NAME)
    if not os.path.exists(xml_path):
        xml_path = os.path.join(os.getcwd(), 'our_robot', XML_FILE_NAME)
        if not os.path.exists(xml_path): 
            raise FileNotFoundError(f"CRITICAL ERROR: XML file '{XML_FILE_NAME}' not found.")
    print(f"Loading model from: {xml_path}")
    verify_model=mujoco.MjModel.from_xml_path(xml_path); verify_data=mujoco.MjData(verify_model)
    for act_name in ACTUATOR_NAMES_ORDERED:
        act_id=mujoco.mj_name2id(verify_model,mujoco.mjtObj.mjOBJ_ACTUATOR,act_name); assert act_id!=-1
        j_id=verify_model.actuator_trnid[act_id,0]; j_name=mujoco.mj_id2name(verify_model,mujoco.mjtObj.mjOBJ_JOINT,j_id); assert j_name
        ACTUATOR_TO_JOINT_NAME_MAP[act_name]=j_name; JOINT_NAME_TO_QPOS_IDX_MAP[j_name]=verify_model.jnt_qposadr[j_id]
    key_id=mujoco.mj_name2id(verify_model,mujoco.mjtObj.mjOBJ_KEY,'home'); assert key_id!=-1
    mujoco.mj_resetDataKeyframe(verify_model,verify_data,key_id)
    print("Home keyframe qpos (sim_keyframe_home_qpos_map):")
    for act_name in ACTUATOR_NAMES_ORDERED:
        j_name=ACTUATOR_TO_JOINT_NAME_MAP[act_name]; q_idx=JOINT_NAME_TO_QPOS_IDX_MAP[j_name]
        sim_keyframe_home_qpos_map[j_name]=verify_data.qpos[q_idx]
        print(f"  {j_name:<15}: {sim_keyframe_home_qpos_map[j_name]:.4f} rad (Real Home: {real_robot_home_deg_map[act_name]:.1f} deg)")
    print("--- Verification Complete ---")
    env=QuadrupedEnv(xml_path)
    run_actuator_perturbation_test(env)
    if KEYBOARD_LIB_AVAILABLE:
        try: keyboard.add_hotkey('ctrl+l',toggle_render); print("INFO: Ctrl+L hotkey active.")
        except Exception as e: print(f"WARNING: Failed to set Ctrl+L hotkey: {e}.")
    state_dim=env.state_dim; action_dim=env.action_dim; device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:{device}. State_dim:{state_dim}, Action_dim:{action_dim}")
    current_lr=INITIAL_LEARNING_RATE; current_entropy_coef=INITIAL_ENTROPY_COEF
    agent=ActorCritic(state_dim,action_dim,INITIAL_ACTION_STD_INIT).to(device); optimizer=optim.Adam(agent.parameters(),lr=current_lr)
    states_mem,actions_mem,log_probs_mem,rewards_mem,values_mem,masks_mem = [],[],[],[],[],[]
    episode_rewards=deque(maxlen=100); avg_reward_history=deque(maxlen=AVGR_HISTORY_LEN); total_env_steps=0
    try:
        for episode in range(1,NUM_EPISODES+1):
            state=env.reset(); current_episode_reward=0; termination_reason_ep="max_steps"
            for step_num in range(MAX_STEPS_PER_EPISODE):
                total_env_steps+=1; state_tensor=torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action_dist,value_tensor=agent(state_tensor); action_tensor=action_dist.sample()
                    log_prob_tensor=action_dist.log_prob(action_tensor).sum(axis=-1)
                action_np=action_tensor.squeeze(0).cpu().numpy()
                next_state,reward,done,info=env.step(action_np)
                states_mem.append(state);actions_mem.append(action_np);log_probs_mem.append(log_prob_tensor.item())
                rewards_mem.append(reward);values_mem.append(value_tensor.item());masks_mem.append(1-done)
                state=next_state; current_episode_reward+=reward
                if done: termination_reason_ep=info.get("termination_reason","unknown"); break
            episode_rewards.append(current_episode_reward)
            current_avg_reward=np.mean(episode_rewards) if episode_rewards else 0.0; avg_reward_history.append(current_avg_reward)
            print(f"Ep:{episode}, Steps:{step_num+1}, R:{current_episode_reward:.2f}, AvgR:{current_avg_reward:.2f} (Term:{termination_reason_ep}), LR:{current_lr:.1e}, EntC:{current_entropy_coef:.4f}, ActStd:{torch.exp(agent.action_log_std.mean()).item():.3f}")
            if len(states_mem)>=POLICY_UPDATE_INTERVAL:
                returns,advantages=[],[]; last_value=0.0
                if not done:
                    with torch.no_grad(): _,last_value_tensor=agent(torch.FloatTensor(next_state).unsqueeze(0).to(device)); last_value=last_value_tensor.item()
                gae=0.0
                for i in reversed(range(len(rewards_mem))):
                    next_val=values_mem[i+1] if i+1<len(values_mem) else last_value; bootstrap_val=next_val*masks_mem[i]
                    delta=rewards_mem[i]+GAMMA*bootstrap_val-values_mem[i]; gae=delta+GAMMA*GAE_LAMBDA*masks_mem[i]*gae
                    returns.insert(0,gae+values_mem[i]); advantages.insert(0,gae)
                states_tensor=torch.FloatTensor(np.array(states_mem)).to(device); actions_tensor=torch.FloatTensor(np.array(actions_mem)).to(device)
                returns_tensor=torch.FloatTensor(returns).to(device).unsqueeze(1); advantages_tensor=torch.FloatTensor(advantages).to(device)
                advantages_tensor=(advantages_tensor-advantages_tensor.mean())/(advantages_tensor.std()+1e-8)
                for _ in range(NUM_EPOCHS_PER_UPDATE):
                    action_dist_new,values_new=agent(states_tensor); log_probs_new=action_dist_new.log_prob(actions_tensor).sum(axis=-1)
                    entropy=action_dist_new.entropy().mean(); actor_loss=-(log_probs_new*advantages_tensor).mean()
                    critic_loss=nn.MSELoss()(values_new,returns_tensor); loss=actor_loss+VALUE_LOSS_COEF*critic_loss-current_entropy_coef*entropy
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(agent.parameters(),MAX_GRAD_NORM);optimizer.step()
                states_mem,actions_mem,log_probs_mem,rewards_mem,values_mem,masks_mem = [],[],[],[],[],[]
            if episode%ADAPTATION_CHECK_INTERVAL==0 and len(avg_reward_history)>=AVGR_HISTORY_LEN:
                first_half_avg=np.mean(list(avg_reward_history)[:AVGR_HISTORY_LEN//2]); second_half_avg=np.mean(list(avg_reward_history)[AVGR_HISTORY_LEN//2:])
                trend_diff=second_half_avg-first_half_avg; significant_change_threshold=0.10*abs(current_avg_reward) if abs(current_avg_reward)>10 else 1.0
                print(f"  Adapting Hypers: TrendDiff={trend_diff:.2f}, CurrentAvgR={current_avg_reward:.2f}, Thresh={significant_change_threshold:.2f}")
                if trend_diff<-significant_change_threshold:
                    print("    Trend: WORSENING..."); current_lr=max(MIN_LR,current_lr*0.75); current_entropy_coef=max(MIN_ENTROPY_COEF,current_entropy_coef*0.9)
                    with torch.no_grad(): agent.action_log_std.data=torch.clamp(agent.action_log_std.data-math.log(1.05),MIN_ACTION_LOG_STD,MAX_ACTION_LOG_STD)
                elif abs(trend_diff)<significant_change_threshold*0.3:
                    print("    Trend: STAGNANT..."); current_entropy_coef=min(MAX_ENTROPY_COEF,current_entropy_coef*1.05)
                    with torch.no_grad(): agent.action_log_std.data=torch.clamp(agent.action_log_std.data+math.log(1.03),MIN_ACTION_LOG_STD,MAX_ACTION_LOG_STD)
                    if current_lr<MAX_LR*0.1: current_lr=min(MAX_LR,current_lr*1.05)
                elif trend_diff>significant_change_threshold:
                    print("    Trend: IMPROVING...");
                    if current_lr>MIN_LR*5: current_lr=max(MIN_LR,current_lr*0.95)
                for param_group in optimizer.param_groups: param_group['lr']=current_lr
                print(f"    New LR:{current_lr:.1e}, New EntC:{current_entropy_coef:.4f}, New ActStd:{torch.exp(agent.action_log_std.mean()).item():.3f}")
            if episode%PTH_SAVE_INTERVAL==0:
                 pth_path=PTH_SAVE_TEMPLATE.format(episode); torch.save(agent.state_dict(),pth_path); print(f"Model saved to {pth_path}")
                 json_path=JSON_OUTPUT_PATH_TEMPLATE.format(episode); generate_walk_json(agent,env,device,json_path,JSON_MAX_STEPS_EPISODIC)
    except KeyboardInterrupt: print("Training interrupted by user (Ctrl+C).")
    finally:
        print("Training finished or interrupted.")
        if KEYBOARD_LIB_AVAILABLE:
            try: keyboard.remove_hotkey('ctrl+l'); print("INFO: Ctrl+L hotkey removed.")
            except Exception as e: print(f"WARNING: Could not remove hotkey: {e}")
        torch.save(agent.state_dict(),FINAL_PTH_SAVE_NAME); print(f"Final model saved as '{FINAL_PTH_SAVE_NAME}'")
        final_json_path=JSON_OUTPUT_PATH_TEMPLATE.format('final'); generate_walk_json(agent,env,device,final_json_path,JSON_MAX_STEPS_FINAL)
        if env.viewer: env.close_viewer_internal()

def generate_walk_json(agent,env,dev,json_path,num_steps):
    print(f"\nGenerating walk ({num_steps} steps) to {json_path}...")
    global g_dynamic_render_active
    if g_dynamic_render_active: print("INFO: Dynamic rendering ON for JSON generation.")
    else: print("INFO: Dynamic rendering OFF for JSON generation.")
    real_robot_sequence=[]; state=env.reset() # Resets episode_policy_step_counter to 0
    for step_count in range(num_steps): # step_count will go 0, 1, ...
        # For JSON generation, the phase for the action should be based on step_count
        # env.episode_policy_step_counter is used internally by env.step()
        # but for consistency in JSON generation, let's ensure it's aligned if we need to log phase
        
        state_tensor=torch.FloatTensor(state).unsqueeze(0).to(dev)
        with torch.no_grad():
            action_dist,_=agent(state_tensor); action_mean=action_dist.mean; action_np=action_mean.squeeze(0).cpu().numpy()
        
        # Log phase *before* env.step() increments its internal counter for the next state
        # phase_for_this_action_json = env.episode_policy_step_counter % PHASE_CYCLE_DURATION_POLICY_STEPS
        # print(f"JSON Gen Step {step_count}, Env Counter Before Step: {env.episode_policy_step_counter}, Phase for action: {phase_for_this_action_json}")

        next_state,_,done,info=env.step(action_np) # env.step uses and then increments its internal counter

        sim_target_rad_cmds=info.get("sim_target_rad")
        if sim_target_rad_cmds is None: print(f"WARNING: sim_target_rad not in info (step {step_count}). Skipping JSON step."); continue
        targets_deg_dict={}
        for act_name_ordered in ACTUATOR_NAMES_ORDERED:
            mujoco_act_id=actuator_ctrl_props[act_name_ordered]['mujoco_id']
            real_deg_val=convert_sim_rad_to_real_deg(act_name_ordered,sim_target_rad_cmds[mujoco_act_id],sim_keyframe_home_qpos_map,real_robot_home_deg_map,joint_scale_factors)
            targets_deg_dict[act_name_ordered]=round(real_deg_val,2) if real_deg_val is not None else 0.0
        real_robot_sequence.append({"duration":round(POLICY_DECISION_DT,3),"targets_deg":targets_deg_dict})
        state=next_state
        if done: print(f"Episode ended prematurely during JSON generation (step {step_count+1}, Term: {info.get('termination_reason','unknown')})."); break
    if not real_robot_sequence: print(f"WARNING: No steps generated for JSON file {json_path}."); return
    try:
        os.makedirs(os.path.dirname(json_path),exist_ok=True)
        with open(json_path,'w') as f: json.dump(real_robot_sequence,f,indent=2)
        print(f"Successfully saved {len(real_robot_sequence)} steps to {json_path}")
    except IOError as e: print(f"ERROR writing JSON to {json_path}: {e}")

if __name__ == "__main__":
    train()