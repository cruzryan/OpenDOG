import mujoco
import mujoco.viewer
import torch
import torch.nn as nn
import numpy as np
import math
import os
import time
import threading

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import matplotlib.ticker as mticker # For formatting ticks

# --- Configuration (Assume all from previous script are here) ---
ACTUATOR_NAMES_ORDERED = [
    "FR_tigh_actuator", "FR_knee_actuator",
    "FL_tigh_actuator", "FL_knee_actuator",
    "BR_tigh_actuator", "BR_knee_actuator",
    "BL_tigh_actuator", "BL_knee_actuator",
]
ACTUATOR_TO_JOINT_NAME_MAP = {}
JOINT_NAME_TO_QPOS_IDX_MAP = {}
sim_keyframe_home_qpos_map = {}
actuator_ctrl_props = {}
XML_FILE_NAME = 'walking_scene.xml'
INITIAL_ACTION_STD_INIT = 0.6
POLICY_DECISION_DT = 0.20
NUM_SETTLE_STEPS = 100
PHASE_CYCLE_DURATION_POLICY_STEPS = 4
ACTION_AMPLITUDE_DEG = 20.0
ACTION_AMPLITUDE_RAD = math.radians(ACTION_AMPLITUDE_DEG)
TARGET_HEIGHT = 0.15
MIN_HEIGHT_TERMINAL = 0.03; MAX_HEIGHT_TERMINAL = 0.50
ORIENTATION_TERMINATION_LIMIT_DEG = 35.0
ORIENTATION_TERMINATION_LIMIT_RAD = math.radians(ORIENTATION_TERMINATION_LIMIT_DEG)
MIN_FWD_DISPLACEMENT_FOR_BACKWARD_CHECK = 0.05
KNEE_SWING_FLEXION_DEG = 30.0
KNEE_SWING_FLEXION_RAD = math.radians(KNEE_SWING_FLEXION_DEG)
KNEE_FLEXION_SIGN_FACTOR = 1.0
# --- Global Render Toggle ---
g_dynamic_render_active = False
g_render_lock = threading.Lock()

# --- Helper Functions ---
def quat_to_ypr(quat):
    q0, q1, q2, q3 = quat[0], quat[1], quat[2], quat[3]
    sinr_cosp = 2 * (q0 * q1 + q2 * q3); cosr_cosp = 1 - 2 * (q1 * q1 + q2 * q2)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (q0 * q2 - q3 * q1)
    pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)
    siny_cosp = 2 * (q0 * q3 + q1 * q2); cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw, pitch, roll

# --- ActorCritic Model Definition (Modified to store activations) ---
class ActorCritic(nn.Module): # (Same as previous, ensure it's the one with self.activations)
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        from torch.distributions import Normal

        self.actor_fc1 = nn.Linear(state_dim, 256)
        self.actor_tanh1 = nn.Tanh()
        self.actor_fc2 = nn.Linear(256, 256)
        self.actor_tanh2 = nn.Tanh()
        self.actor_fc_out = nn.Linear(256, action_dim)
        self.actor_tanh_out = nn.Tanh()

        self.critic_fc1 = nn.Linear(state_dim, 256)
        self.critic_tanh1 = nn.Tanh()
        self.critic_fc2 = nn.Linear(256, 256)
        self.critic_tanh2 = nn.Tanh()
        self.critic_fc_out = nn.Linear(256, 1)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * math.log(action_std_init))
        self.activations = {}

    def forward(self, state):
        from torch.distributions import Normal
        self.activations.clear()
        self.activations['input_state'] = state.detach().cpu()

        a_x = self.actor_tanh1(self.actor_fc1(state))
        self.activations['actor_l1_post_tanh'] = a_x.detach().cpu()
        a_x = self.actor_tanh2(self.actor_fc2(a_x))
        self.activations['actor_l2_post_tanh'] = a_x.detach().cpu()
        action_mean_raw = self.actor_fc_out(a_x)
        action_mean = self.actor_tanh_out(action_mean_raw)
        self.activations['action_mean_scaled'] = action_mean.detach().cpu() # Key for scaled actions

        c_x = self.critic_tanh1(self.critic_fc1(state))
        self.activations['critic_l1_post_tanh'] = c_x.detach().cpu()
        c_x = self.critic_tanh2(self.critic_fc2(c_x))
        self.activations['critic_l2_post_tanh'] = c_x.detach().cpu()
        value = self.critic_fc_out(c_x)
        self.activations['value_output'] = value.detach().cpu()

        action_std = torch.exp(self.action_log_std.expand_as(action_mean))
        return Normal(action_mean, action_std), value

# --- QuadrupedEnv Definition ---
class QuadrupedEnv: # (Same as previous)
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.sim_steps_per_policy_step = max(1, int(POLICY_DECISION_DT / self.model.opt.timestep))
        global actuator_ctrl_props; actuator_ctrl_props.clear()
        for name in ACTUATOR_NAMES_ORDERED:
            mujoco_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if mujoco_act_id == -1: raise ValueError(f"Actuator '{name}' not found for props.")
            ctrlrange = self.model.actuator_ctrlrange[mujoco_act_id]
            actuator_ctrl_props[name] = {'ctrlrange': ctrlrange.copy(), 'mujoco_id': mujoco_act_id}
        self.state_dim = 1 + 2 + 8 + 8 + 1 + 2; self.action_dim = 4
        key_id_for_init = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if key_id_for_init == -1: raise ValueError("Keyframe 'home' not found in XML for env init.")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id_for_init)
        self.initial_qpos = self.data.qpos.copy(); self.initial_qvel = self.data.qvel.copy()
        self.initial_ctrl = self.data.ctrl.copy(); self.episode_policy_step_counter = 0
        self.last_commanded_clipped_sim_rad = self.initial_ctrl.copy()
        self.previous_x_qpos_in_episode = 0.0; self.cumulative_positive_x_displacement = 0.0
        self.cumulative_negative_x_displacement = 0.0
        self.target_gait_angles_rad_sim = np.zeros(len(ACTUATOR_NAMES_ORDERED))
    def _get_observation(self):
        z_pos = self.data.qpos[2]; _, pitch, roll = quat_to_ypr(self.data.qpos[3:7])
        joint_pos_values = []; joint_vel_values = []
        for act_name in ACTUATOR_NAMES_ORDERED:
            joint_name = ACTUATOR_TO_JOINT_NAME_MAP[act_name]; qpos_idx = JOINT_NAME_TO_QPOS_IDX_MAP[joint_name]
            joint_id_for_vel = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qvel_idx = self.model.jnt_dofadr[joint_id_for_vel]
            joint_pos_values.append(self.data.qpos[qpos_idx] - sim_keyframe_home_qpos_map[joint_name])
            joint_vel_values.append(self.data.qvel[qvel_idx])
        joint_pos_dev = np.array(joint_pos_values); joint_vel = np.array(joint_vel_values)
        trunk_fwd_vel = self.data.qvel[0]
        phase_progress = (self.episode_policy_step_counter % PHASE_CYCLE_DURATION_POLICY_STEPS) / PHASE_CYCLE_DURATION_POLICY_STEPS
        sin_phase = np.sin(phase_progress * 2 * np.pi); cos_phase = np.cos(phase_progress * 2 * np.pi)
        state = np.concatenate([[z_pos - TARGET_HEIGHT], [pitch, roll], joint_pos_dev, joint_vel, [trunk_fwd_vel], [sin_phase, cos_phase]])
        return state.astype(np.float32)
    def reset(self):
        self.data.qpos[:] = self.initial_qpos; self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = self.initial_ctrl; mujoco.mj_forward(self.model, self.data)
        self.episode_policy_step_counter = 0; self.previous_x_qpos_in_episode = self.data.qpos[0]
        self.cumulative_positive_x_displacement = 0.0; self.cumulative_negative_x_displacement = 0.0
        for _ in range(NUM_SETTLE_STEPS):
            self.data.ctrl[:] = self.initial_ctrl
            try: mujoco.mj_step(self.model, self.data)
            except mujoco.FatalError: break; self.sync_viewer_if_active()
        self.last_commanded_clipped_sim_rad = self.data.ctrl.copy(); self.previous_x_qpos_in_episode = self.data.qpos[0]
        return self._get_observation()
    def _set_target_gait_angles(self, current_gait_phase_idx):
        global sim_keyframe_home_qpos_map, KNEE_FLEXION_SIGN_FACTOR
        for i, act_name in enumerate(ACTUATOR_NAMES_ORDERED):
            joint_name = ACTUATOR_TO_JOINT_NAME_MAP[act_name]
            self.target_gait_angles_rad_sim[i] = sim_keyframe_home_qpos_map[joint_name]
        knee_swing_delta_rad = KNEE_FLEXION_SIGN_FACTOR * KNEE_SWING_FLEXION_RAD
        if current_gait_phase_idx == 0:
            fr_idx=ACTUATOR_NAMES_ORDERED.index("FR_knee_actuator"); bl_idx=ACTUATOR_NAMES_ORDERED.index("BL_knee_actuator")
            self.target_gait_angles_rad_sim[fr_idx]+=knee_swing_delta_rad; self.target_gait_angles_rad_sim[bl_idx]+=knee_swing_delta_rad
        elif current_gait_phase_idx == 2:
            fl_idx=ACTUATOR_NAMES_ORDERED.index("FL_knee_actuator"); br_idx=ACTUATOR_NAMES_ORDERED.index("BR_knee_actuator")
            self.target_gait_angles_rad_sim[fl_idx]+=knee_swing_delta_rad; self.target_gait_angles_rad_sim[br_idx]+=knee_swing_delta_rad
    def step(self, policy_actions_scaled_neg1_to_1):
        self.episode_policy_step_counter += 1
        current_gait_phase_idx = self.episode_policy_step_counter % PHASE_CYCLE_DURATION_POLICY_STEPS
        self._set_target_gait_angles(current_gait_phase_idx)
        final_clipped_commands_all_actuators = np.zeros_like(self.data.ctrl) # For ALL 8 actuators
        actuator_deltas_rad = { # For 4 policy outputs
            "FR_tigh_actuator": policy_actions_scaled_neg1_to_1[0] * ACTION_AMPLITUDE_RAD,
            "FR_knee_actuator": policy_actions_scaled_neg1_to_1[1] * ACTION_AMPLITUDE_RAD,
            "FL_tigh_actuator": policy_actions_scaled_neg1_to_1[2] * ACTION_AMPLITUDE_RAD,
            "FL_knee_actuator": policy_actions_scaled_neg1_to_1[3] * ACTION_AMPLITUDE_RAD,
        } # Apply symmetry for other 4
        actuator_deltas_rad["BR_tigh_actuator"] = actuator_deltas_rad["FL_tigh_actuator"]
        actuator_deltas_rad["BR_knee_actuator"] = actuator_deltas_rad["FL_knee_actuator"]
        actuator_deltas_rad["BL_tigh_actuator"] = actuator_deltas_rad["FR_tigh_actuator"]
        actuator_deltas_rad["BL_knee_actuator"] = actuator_deltas_rad["FR_knee_actuator"]
        for i, act_name in enumerate(ACTUATOR_NAMES_ORDERED): # Iterate all 8
            props = actuator_ctrl_props[act_name]; mujoco_act_id = props['mujoco_id']
            base_angle_for_policy_delta_rad = self.target_gait_angles_rad_sim[i]
            policy_delta_rad = actuator_deltas_rad[act_name] # Get the symmetric delta
            sim_target_rad_unclipped = base_angle_for_policy_delta_rad + policy_delta_rad
            clipped_sim_target_rad = np.clip(sim_target_rad_unclipped, props['ctrlrange'][0], props['ctrlrange'][1])
            final_clipped_commands_all_actuators[mujoco_act_id] = clipped_sim_target_rad
        self.data.ctrl[:] = final_clipped_commands_all_actuators
        sum_mj_errors = 0
        for _ in range(self.sim_steps_per_policy_step):
            try: mujoco.mj_step(self.model, self.data)
            except mujoco.FatalError: sum_mj_errors +=1; break
            self.sync_viewer_if_active()
        # ... (rest of step logic for obs, reward, done, info remains the same)
        current_x_pos_after_sim_loop = self.data.qpos[0]
        delta_x_this_policy_step = current_x_pos_after_sim_loop - self.previous_x_qpos_in_episode
        if delta_x_this_policy_step > 0: self.cumulative_positive_x_displacement += delta_x_this_policy_step
        elif delta_x_this_policy_step < 0: self.cumulative_negative_x_displacement += abs(delta_x_this_policy_step)
        self.previous_x_qpos_in_episode = current_x_pos_after_sim_loop
        obs = self._get_observation(); reward = 1.0; done = False
        info = {"sim_target_rad": final_clipped_commands_all_actuators.copy(), "termination_reason": "running"} # CRITICAL: pass final commands
        if sum_mj_errors > 0: done = True; info["termination_reason"] = "mj_error"
        current_height_term = self.data.qpos[2]
        if not (MIN_HEIGHT_TERMINAL < current_height_term < MAX_HEIGHT_TERMINAL): done = True; info["termination_reason"] = "height_limit"
        _, current_pitch, current_roll = quat_to_ypr(self.data.qpos[3:7])
        if abs(current_roll) > ORIENTATION_TERMINATION_LIMIT_RAD or abs(current_pitch) > ORIENTATION_TERMINATION_LIMIT_RAD: done = True; info["termination_reason"] = "orientation_limit"
        if not done and self.episode_policy_step_counter > 10:
             if self.cumulative_positive_x_displacement > MIN_FWD_DISPLACEMENT_FOR_BACKWARD_CHECK and self.cumulative_negative_x_displacement > 0.5 * self.cumulative_positive_x_displacement:
                 done = True; info["termination_reason"] = "too_much_backward"
        self.last_commanded_clipped_sim_rad = final_clipped_commands_all_actuators.copy()
        return obs, reward, done, info
    def launch_viewer_internal(self): # (Same as previous)
        if self.viewer is None:
            try: self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception: self.viewer = None
        elif self.viewer and not self.viewer.is_running(): self.viewer = None; self.launch_viewer_internal()
    def close_viewer_internal(self): # (Same as previous)
        if self.viewer and self.viewer.is_running():
            try: self.viewer.close()
            except Exception: pass
            finally: self.viewer = None
        elif self.viewer and not self.viewer.is_running(): self.viewer = None
    def sync_viewer_if_active(self): # (Same as previous)
        global g_dynamic_render_active, g_render_lock
        with g_render_lock:
            if g_dynamic_render_active:
                if self.viewer is None or not self.viewer.is_running(): self.launch_viewer_internal()
                if self.viewer and self.viewer.is_running():
                    try: self.viewer.sync()
                    except Exception: self.close_viewer_internal()
            elif self.viewer and self.viewer.is_running(): self.close_viewer_internal()

# --- Function to setup environment dependencies (globals) ---
def setup_env_dependencies(xml_file_path): # (Same as previous)
    print("\n--- Setting up Environment Dependencies ---")
    global ACTUATOR_TO_JOINT_NAME_MAP, JOINT_NAME_TO_QPOS_IDX_MAP, sim_keyframe_home_qpos_map, KNEE_FLEXION_SIGN_FACTOR
    ACTUATOR_TO_JOINT_NAME_MAP.clear(); JOINT_NAME_TO_QPOS_IDX_MAP.clear(); sim_keyframe_home_qpos_map.clear()
    model = mujoco.MjModel.from_xml_path(xml_file_path); data = mujoco.MjData(model)
    for act_name in ACTUATOR_NAMES_ORDERED:
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        joint_id = model.actuator_trnid[act_id, 0]
        joint_name_str = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        ACTUATOR_TO_JOINT_NAME_MAP[act_name] = joint_name_str
        qpos_idx = model.jnt_qposadr[joint_id]
        JOINT_NAME_TO_QPOS_IDX_MAP[joint_name_str] = qpos_idx
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    for act_name in ACTUATOR_NAMES_ORDERED:
        joint_name = ACTUATOR_TO_JOINT_NAME_MAP[act_name]; qpos_idx = JOINT_NAME_TO_QPOS_IDX_MAP[joint_name]
        sim_keyframe_home_qpos_map[joint_name] = data.qpos[qpos_idx]
    fr_knee_act_name = "FR_knee_actuator"
    fr_knee_joint_name = ACTUATOR_TO_JOINT_NAME_MAP[fr_knee_act_name]
    fr_knee_qpos_idx = JOINT_NAME_TO_QPOS_IDX_MAP[fr_knee_joint_name]
    fr_knee_act_id_mj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, fr_knee_act_name)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    qpos_before_perturb = data.qpos[fr_knee_qpos_idx]
    current_ctrl_at_home_fr_knee = data.ctrl[fr_knee_act_id_mj]
    perturb_ctrl_delta = 0.05; ctrl_range = model.actuator_ctrlrange[fr_knee_act_id_mj]
    perturbed_ctrl = np.clip(current_ctrl_at_home_fr_knee + perturb_ctrl_delta, ctrl_range[0], ctrl_range[1])
    temp_data_perturb = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, temp_data_perturb, key_id)
    temp_data_perturb.ctrl[fr_knee_act_id_mj] = perturbed_ctrl
    mujoco.mj_step(model, temp_data_perturb)
    qpos_after_perturb = temp_data_perturb.qpos[fr_knee_qpos_idx]
    delta_qpos = qpos_after_perturb - qpos_before_perturb
    if abs(delta_qpos) < 1e-5: KNEE_FLEXION_SIGN_FACTOR = 1.0
    elif np.sign(delta_qpos) == np.sign(KNEE_SWING_FLEXION_RAD if KNEE_SWING_FLEXION_RAD != 0 else 1.0): KNEE_FLEXION_SIGN_FACTOR = 1.0
    else: KNEE_FLEXION_SIGN_FACTOR = -1.0
    print(f"  KNEE_FLEXION_SIGN_FACTOR = {KNEE_FLEXION_SIGN_FACTOR}")
    print("--- Environment Dependencies Setup Complete ---")
    return xml_file_path

# --- Visualization Data Storage and Plot Artists ---
viz_artists = {} # Holds all matplotlib artist objects for updating
# Shortened labels for state vector components for better display
STATE_LABELS_SHORT = [
    "z_H", "Pitch", "Roll", # Body pose
    "FRt_p", "FRk_p", "FLt_p", "FLk_p", "BRt_p", "BRk_p", "BLt_p", "BLk_p", # Joint positions (relative to home)
    "FRt_v", "FRk_v", "FLt_v", "FLk_v", "BRt_v", "BRk_v", "BLt_v", "BLk_v", # Joint velocities
    "FwdVel", "Sin(Ph)", "Cos(Ph)" # Body velocity & Gait Phase
]
# Labels for the 4 scaled action outputs from the policy network
POLICY_ACTION_LABELS_SHORT = ["FRt_Δ", "FRk_Δ", "FLt_Δ", "FLk_Δ"] # Delta indicates they are additive adjustments
# Labels for ALL 8 final commanded actuator angles (after full calculation)
FINAL_MOTOR_COMMAND_LABELS_SHORT = [name.replace('_actuator','').replace('_tigh','T').replace('_knee','K') for name in ACTUATOR_NAMES_ORDERED]


# Normalizer for Tanh outputs (-1 to 1)
tanh_norm = Normalize(vmin=-1.0, vmax=1.0)
# Normalizer for joint angles in radians (e.g., -pi/2 to pi/2, adjust if needed based on your robot's limits)
# We'll use the actual ctrlrange from MuJoCo for the final motor commands plot for accuracy
# angle_norm_rad = Normalize(vmin=-math.pi/2, vmax=math.pi/2)
angle_norm_deg = Normalize(vmin=-90, vmax=90) # Example for degrees

# Store latest motor commands for plotting
latest_final_motor_commands_rad = np.zeros(len(ACTUATOR_NAMES_ORDERED))

def initialize_intuitive_plots(agent, env, fig):
    global viz_artists, latest_final_motor_commands_rad
    viz_artists.clear()
    latest_final_motor_commands_rad = np.zeros(len(ACTUATOR_NAMES_ORDERED)) # Reset

    # --- Define Plot Structure ---
    # GridSpec: rows, columns, figure
    # We'll have sections: Input, Actor, Critic, (Optional: Static Weights)
    # Let's try a 5-row structure initially.
    # Row 0: Input State
    # Row 1: Actor Hidden Layer 1
    # Row 2: Actor Hidden Layer 2 + Scaled Policy Actions
    # Row 3: Final Motor Commands (Derived)
    # Row 4: Critic Hidden Layers + Value Output

    gs = GridSpec(5, 2, figure=fig, hspace=0.8, wspace=0.3,
                  height_ratios=[1, 1, 1.2, 1.5, 1]) # Give more height to motor commands

    # --- Row 0: Input State ---
    ax_input = fig.add_subplot(gs[0, :]) # Span both columns
    data_placeholder = np.zeros((1, env.state_dim))
    img_input = ax_input.imshow(data_placeholder, aspect='auto', cmap='viridis', interpolation='nearest')
    ax_input.set_title("1. Neural Network Input (Current Robot State)", fontsize=10, loc='left', pad=10)
    ax_input.set_yticks([])
    ax_input.set_xticks(np.arange(len(STATE_LABELS_SHORT)))
    ax_input.set_xticklabels(STATE_LABELS_SHORT, rotation=90, ha="right", fontsize=7)
    fig.colorbar(img_input, ax=ax_input, orientation='horizontal', fraction=0.05, pad=0.2, label="State Value")
    viz_artists['input_state_img'] = img_input
    ax_input.text(1.01, 0.5, "What the robot currently sees/feels.\n(e.g., height, tilt, joint positions, phase of walk)",
                  transform=ax_input.transAxes, fontsize=8, va='center', ha='left', wrap=True,
                  bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", lw=0.5))


    # --- Actor Network Flow ---
    # Row 1, Col 0: Actor Hidden Layer 1
    ax_actor_h1 = fig.add_subplot(gs[1, 0])
    data_placeholder_h1 = np.zeros((1, 256)) # Assuming 256 neurons
    img_actor_h1 = ax_actor_h1.imshow(data_placeholder_h1, aspect='auto', cmap='coolwarm', norm=tanh_norm, interpolation='nearest')
    ax_actor_h1.set_title("2a. Actor Network - Hidden Layer 1 Activations", fontsize=9)
    ax_actor_h1.set_yticks([])
    ax_actor_h1.set_xticks([])
    fig.colorbar(img_actor_h1, ax=ax_actor_h1, orientation='horizontal', fraction=0.05, pad=0.2, label="Activation (-1 to 1)")
    viz_artists['actor_l1_post_tanh_img'] = img_actor_h1

    # Row 1, Col 1: Actor Hidden Layer 2
    ax_actor_h2 = fig.add_subplot(gs[1, 1])
    img_actor_h2 = ax_actor_h2.imshow(data_placeholder_h1, aspect='auto', cmap='coolwarm', norm=tanh_norm, interpolation='nearest')
    ax_actor_h2.set_title("2b. Actor Network - Hidden Layer 2 Activations", fontsize=9)
    ax_actor_h2.set_yticks([])
    ax_actor_h2.set_xticks([])
    fig.colorbar(img_actor_h2, ax=ax_actor_h2, orientation='horizontal', fraction=0.05, pad=0.2, label="Activation (-1 to 1)")
    viz_artists['actor_l2_post_tanh_img'] = img_actor_h2

    # Row 2, Col 0: Scaled Policy Actions (NN Output for Actor)
    ax_policy_actions = fig.add_subplot(gs[2, 0])
    data_placeholder_actions = np.zeros((1, env.action_dim))
    img_policy_actions = ax_policy_actions.imshow(data_placeholder_actions, aspect='auto', cmap='coolwarm', norm=tanh_norm, interpolation='nearest')
    ax_policy_actions.set_title("3. Actor Output: Scaled Adjustments [-1,1]", fontsize=9)
    ax_policy_actions.set_yticks([])
    ax_policy_actions.set_xticks(np.arange(len(POLICY_ACTION_LABELS_SHORT)))
    ax_policy_actions.set_xticklabels(POLICY_ACTION_LABELS_SHORT, rotation=0, ha="center", fontsize=8)
    fig.colorbar(img_policy_actions, ax=ax_policy_actions, orientation='horizontal', fraction=0.05, pad=0.25, label="Scaled Action")
    viz_artists['action_mean_scaled_img'] = img_policy_actions
    ax_policy_actions.text(0.5, -0.5, "How much to adjust each of the 4 primary motor groups.\nPositive means one way, negative the other.",
                  transform=ax_policy_actions.transAxes, fontsize=8, va='top', ha='center', wrap=True,
                  bbox=dict(boxstyle="round,pad=0.3", fc="lightcyan", ec="gray", lw=0.5))


    # Row 3: Final Motor Commands (This is new and crucial for "what angles")
    ax_final_motors = fig.add_subplot(gs[3, :]) # Span both columns
    # We need to know the ctrlrange for each motor to set the norm
    # For simplicity, let's assume a common range or find min/max of all actual ctrlranges
    # Or, even better, plot as bar chart with actual values
    num_motors = len(ACTUATOR_NAMES_ORDERED)
    motor_positions_placeholder = np.zeros(num_motors)
    bars_final_motors = ax_final_motors.bar(np.arange(num_motors), motor_positions_placeholder, color='skyblue')
    ax_final_motors.set_title("4. FINAL MOTOR COMMANDS (Sent to Robot's Joints in Degrees)", fontsize=10, loc='left', pad=10, color='green', weight='bold')
    ax_final_motors.set_xticks(np.arange(num_motors))
    ax_final_motors.set_xticklabels(FINAL_MOTOR_COMMAND_LABELS_SHORT, rotation=45, ha="right", fontsize=8)
    ax_final_motors.set_ylabel("Angle (Degrees)", fontsize=9)
    ax_final_motors.axhline(0, color='grey', lw=0.8) # Zero line
    # Determine y-limits from typical motor ranges (e.g., -90 to +90 degrees)
    # We will dynamically set this in update based on ctrlrange later for more accuracy
    ax_final_motors.set_ylim(-100, 100) # Placeholder, will be updated
    viz_artists['final_motor_commands_bars'] = bars_final_motors
    viz_artists['final_motor_commands_ax'] = ax_final_motors
    # Add text annotations for each bar
    bar_texts = []
    for i in range(num_motors):
        bar_texts.append(ax_final_motors.text(i, 0, "0°", ha='center', va='bottom', fontsize=7, color='black'))
    viz_artists['final_motor_commands_texts'] = bar_texts
    ax_final_motors.text(1.01, 0.5, "The actual target angles for each of the 8 leg joints.\nCalculated from Actor Output + Gait Pattern + Limits.",
                  transform=ax_final_motors.transAxes, fontsize=8, va='center', ha='left', wrap=True,
                  bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="gray", lw=0.5))


    # --- Critic Network Flow ---
    # Row 4, Col 0: Critic Hidden Layers (can combine for simplicity for noob)
    ax_critic_h = fig.add_subplot(gs[4, 0])
    img_critic_h1 = ax_critic_h.imshow(data_placeholder_h1, aspect='auto', cmap='coolwarm', norm=tanh_norm, interpolation='nearest')
    ax_critic_h.set_title("5a. Critic Network - Hidden Activations", fontsize=9)
    ax_critic_h.set_yticks([]); ax_critic_h.set_xticks([])
    fig.colorbar(img_critic_h1, ax=ax_critic_h, orientation='horizontal', fraction=0.05, pad=0.2, label="Activation (-1 to 1)")
    viz_artists['critic_l1_post_tanh_img'] = img_critic_h1 # Just show L1 for simplicity, or combine L1 and L2

    # Row 4, Col 1: Critic Value Output
    ax_value = fig.add_subplot(gs[4, 1])
    value_placeholder = np.array([[0.0]])
    img_value = ax_value.imshow(value_placeholder, aspect='auto', cmap='RdYlGn', vmin=-5, vmax=5, interpolation='nearest') # Example range
    ax_value.set_title("5b. Critic Output: Estimated Value", fontsize=9, color='purple', weight='bold')
    ax_value.set_yticks([]); ax_value.set_xticks([])
    cbar_value = fig.colorbar(img_value, ax=ax_value, orientation='horizontal', fraction=0.05, pad=0.25)
    viz_artists['value_output_img'] = img_value
    viz_artists['value_output_cbar'] = cbar_value # Store cbar to update its label
    value_text_annotation = ax_value.text(0.5, 0.5, "Value: 0.0", ha='center', va='center', fontsize=10, color='black', weight='bold')
    viz_artists['value_output_text'] = value_text_annotation
    ax_value.text(0.5, -0.5, "Network's guess of 'how good' the current situation is.\nHigher is generally better.",
                  transform=ax_value.transAxes, fontsize=8, va='top', ha='center', wrap=True,
                  bbox=dict(boxstyle="round,pad=0.3", fc="thistle", ec="gray", lw=0.5))

    # --- Optional: Static Weights (Can be a separate smaller figure or section) ---
    # For now, let's skip static weights to focus on dynamic parts for "noob" clarity.
    # If adding them, use a new GridSpec section.

    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.07, right=0.85) # Fine tune
    plt.show(block=False)

def update_intuitive_plots(agent, fig):
    global viz_artists, latest_final_motor_commands_rad
    if not viz_artists or not hasattr(agent, 'activations') or not agent.activations:
        return

    # 1. Input State
    if 'input_state_img' in viz_artists and 'input_state' in agent.activations:
        data_np = agent.activations['input_state'].numpy().squeeze().reshape(1, -1)
        img = viz_artists['input_state_img']
        img.set_data(data_np)
        min_val, max_val = data_np.min(), data_np.max()
        if min_val == max_val: min_val -=0.1; max_val+=0.1
        img.set_clim(vmin=min_val, vmax=max_val)

    # 2a. Actor L1
    if 'actor_l1_post_tanh_img' in viz_artists and 'actor_l1_post_tanh' in agent.activations:
        data_np = agent.activations['actor_l1_post_tanh'].numpy().squeeze().reshape(1, -1)
        viz_artists['actor_l1_post_tanh_img'].set_data(data_np)
    # 2b. Actor L2
    if 'actor_l2_post_tanh_img' in viz_artists and 'actor_l2_post_tanh' in agent.activations: # Assuming you added this key
        data_np = agent.activations['actor_l2_post_tanh'].numpy().squeeze().reshape(1, -1)
        viz_artists['actor_l2_post_tanh_img'].set_data(data_np)


    # 3. Scaled Policy Actions
    if 'action_mean_scaled_img' in viz_artists and 'action_mean_scaled' in agent.activations:
        data_np = agent.activations['action_mean_scaled'].numpy().squeeze().reshape(1, -1)
        viz_artists['action_mean_scaled_img'].set_data(data_np)

    # 4. Final Motor Commands (Degrees) - This uses `latest_final_motor_commands_rad`
    if 'final_motor_commands_bars' in viz_artists:
        commands_deg = np.degrees(latest_final_motor_commands_rad)
        bars = viz_artists['final_motor_commands_bars']
        bar_texts = viz_artists['final_motor_commands_texts']
        ax = viz_artists['final_motor_commands_ax']
        min_angle_deg, max_angle_deg = float('inf'), float('-inf')

        for i, bar in enumerate(bars):
            bar.set_height(commands_deg[i])
            bar_texts[i].set_text(f"{commands_deg[i]:.1f}°")
            bar_texts[i].set_y(commands_deg[i] + (5 if commands_deg[i] >= 0 else -15) ) # Adjust text pos

            # Determine dynamic y-limits based on actual command ranges and MuJoCo ctrlrange
            # This part makes the y-axis adaptive and accurate
            act_name = ACTUATOR_NAMES_ORDERED[i]
            mujoco_act_id = actuator_ctrl_props[act_name]['mujoco_id'] # Need mujoco_id in props
            ctrl_min_rad, ctrl_max_rad = actuator_ctrl_props[act_name]['ctrlrange']
            min_angle_deg = min(min_angle_deg, math.degrees(ctrl_min_rad))
            max_angle_deg = max(max_angle_deg, math.degrees(ctrl_max_rad))


        # Add some padding to y-limits
        y_padding = (max_angle_deg - min_angle_deg) * 0.1
        ax.set_ylim(min_angle_deg - y_padding, max_angle_deg + y_padding)


    # 5a. Critic L1
    if 'critic_l1_post_tanh_img' in viz_artists and 'critic_l1_post_tanh' in agent.activations: # Assuming key
        data_np = agent.activations['critic_l1_post_tanh'].numpy().squeeze().reshape(1, -1)
        viz_artists['critic_l1_post_tanh_img'].set_data(data_np)

    # 5b. Value Output
    if 'value_output_img' in viz_artists and 'value_output' in agent.activations:
        value = agent.activations['value_output'].item()
        viz_artists['value_output_img'].set_data(np.array([[value]]))
        # Dynamically adjust colorbar for value, or keep fixed if you know the typical range
        # viz_artists['value_output_img'].set_clim(vmin=min(value-1, -5), vmax=max(value+1, 5))
        viz_artists['value_output_text'].set_text(f"Value: {value:.2f}")
        # Update cbar label to show current range if dynamic, or keep fixed
        # cbar = viz_artists['value_output_cbar']
        # cbar.set_label(f"Est. Value ({cbar.vmin:.1f} to {cbar.vmax:.1f})", size=8)


    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# --- Main Visualization Loop ---
def run_simulation_and_visualize_intuitive(agent, env, device, num_steps=1000, plot_update_freq=2):
    global g_dynamic_render_active, latest_final_motor_commands_rad
    g_dynamic_render_active = True
    state = env.reset()

    plt.ion()
    fig_main = plt.figure(figsize=(14, 16)) # Wider and taller for clarity
    initialize_intuitive_plots(agent, env, fig_main)

    print("\nStarting simulation and INTUITIVE visualization loop...")
    time_last_plot_update = time.time()

    try:
        for step_count in range(num_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_dist, _ = agent(state_tensor) # Populates agent.activations
                if 'action_mean_scaled' in agent.activations:
                    action_to_take = agent.activations['action_mean_scaled'].numpy().squeeze()
                else: # Fallback
                    action_to_take = action_dist.mean.cpu().numpy().squeeze()

            next_state, _, done, info = env.step(action_to_take)
            
            # Store the final motor commands (in radians, as used by sim) from `info`
            # Ensure that `info["sim_target_rad"]` contains the commands for ALL 8 actuators
            # in the order of ACTUATOR_NAMES_ORDERED.
            # The current env.step returns final_clipped_commands_all_actuators which are indexed by mujoco_id.
            # We need to map them back to the ACTUATOR_NAMES_ORDERED sequence.
            sim_targets_from_info = info.get("sim_target_rad")
            if sim_targets_from_info is not None:
                for i, act_name in enumerate(ACTUATOR_NAMES_ORDERED):
                    mujoco_id = actuator_ctrl_props[act_name]['mujoco_id']
                    latest_final_motor_commands_rad[i] = sim_targets_from_info[mujoco_id]
            else:
                print("Warning: 'sim_target_rad' not in info from env.step(). Motor command plot might be inaccurate.")


            current_time = time.time()
            if (step_count % plot_update_freq == 0 and current_time - time_last_plot_update > 0.05) or done: # Plot more often
                if plt.fignum_exists(fig_main.number):
                    update_intuitive_plots(agent, fig_main)
                    time_last_plot_update = current_time
                else: break

            state = next_state
            if done:
                print(f"Episode done. Reason: {info.get('termination_reason', 'unknown')}. Resetting.")
                state = env.reset()
                if not plt.fignum_exists(fig_main.number): break
    except KeyboardInterrupt: print("\nLoop interrupted.")
    except Exception as e: print(f"Error in loop: {e}"); import traceback; traceback.print_exc()
    finally:
        print("Closing viz."); plt.ioff()
        if 'fig_main' in locals() and fig_main and plt.fignum_exists(fig_main.number): plt.close(fig_main)
        if env.viewer: env.close_viewer_internal()
        g_dynamic_render_active = False

if __name__ == "__main__":
    MODEL_LOAD_PATH = './best/quadruped_ac_terrain_ep6800.pth'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    xml_full_path = None; script_dir = os.getcwd()
    try: script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: pass
    path_attempt1 = os.path.join(os.path.dirname(script_dir), 'our_robot', XML_FILE_NAME)
    path_attempt2 = os.path.join(os.getcwd(), 'our_robot', XML_FILE_NAME)
    if os.path.exists(path_attempt1): xml_full_path = path_attempt1
    elif os.path.exists(path_attempt2): xml_full_path = path_attempt2
    else: print(f"ERROR: XML '{XML_FILE_NAME}' not found. Exiting."); exit()
    print(f"Found XML at: {xml_full_path}")

    setup_env_dependencies(xml_full_path)
    env = QuadrupedEnv(model_path=xml_full_path)
    state_dim = env.state_dim; action_dim = env.action_dim
    agent = ActorCritic(state_dim, action_dim, action_std_init=INITIAL_ACTION_STD_INIT).to(DEVICE)

    if not os.path.exists(MODEL_LOAD_PATH): print(f"ERROR: Model '{MODEL_LOAD_PATH}' not found."); exit()
    try:
        print(f"Loading state_dict from '{MODEL_LOAD_PATH}' (weights_only=True)...")
        original_state_dict = torch.load(MODEL_LOAD_PATH, map_location=DEVICE, weights_only=True)
        new_state_dict = {}
        key_map_actor = {"actor.0.weight": "actor_fc1.weight", "actor.0.bias": "actor_fc1.bias",
                         "actor.2.weight": "actor_fc2.weight", "actor.2.bias": "actor_fc2.bias",
                         "actor.4.weight": "actor_fc_out.weight", "actor.4.bias": "actor_fc_out.bias"}
        for old, new in key_map_actor.items():
            if old in original_state_dict: new_state_dict[new] = original_state_dict[old]
        key_map_critic = {"critic.0.weight": "critic_fc1.weight", "critic.0.bias": "critic_fc1.bias",
                          "critic.2.weight": "critic_fc2.weight", "critic.2.bias": "critic_fc2.bias",
                          "critic.4.weight": "critic_fc_out.weight", "critic.4.bias": "critic_fc_out.bias"}
        for old, new in key_map_critic.items():
            if old in original_state_dict: new_state_dict[new] = original_state_dict[old]
        if "action_log_std" in original_state_dict: new_state_dict["action_log_std"] = original_state_dict["action_log_std"]
        
        missing, unexpected = agent.load_state_dict(new_state_dict, strict=False)
        if missing: print(f"Missing keys after remapping: {missing}")
        if not missing: print("State_dict loaded successfully (non-strict).")
        else: print("ERROR: Model loading failed due to missing keys."); exit()
        agent.eval()
    except Exception as e: print(f"Error loading model: {e}"); import traceback; traceback.print_exc(); exit()

    run_simulation_and_visualize_intuitive(agent, env, DEVICE, num_steps=5000, plot_update_freq=1) # Very frequent updates

    print("\nProgram finished.")