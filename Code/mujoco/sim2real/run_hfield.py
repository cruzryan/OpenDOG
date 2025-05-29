import locale
import os
# import argparse # No longer needed
import time
import threading
import random
import math

import mujoco
import mujoco.viewer
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

try:
    import keyboard
    KEYBOARD_LIB_AVAILABLE = True
except ImportError:
    KEYBOARD_LIB_AVAILABLE = False
    print("WARNING: 'keyboard' library not found. Ctrl+L rendering toggle will not be available.")
    print("To enable, run: pip install keyboard")

# --- Configuration (Subset from run2.py, relevant for running/viewing) ---
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
sim_keyframe_home_qpos_map = {}
actuator_ctrl_props = {}

# --- HARDCODED PARAMETERS ---
MODEL_PTH_TO_LOAD = './best/quadruped_ac_terrain_ep6800.pth' # <-- YOUR MODEL PATH
XML_FILE_NAME = '..\our_robot\walking_scene.xml' # Assumed to be in the same directory as run.py
MAX_SIMULATION_STEPS = 20000 # How long each "episode" runs before reset

# RL Hyperparameters (relevant for environment behavior and agent architecture)
INITIAL_ACTION_STD_INIT = 0.3

# Action scaling
ACTION_AMPLITUDE_DEG = 50.0
ACTION_AMPLITUDE_RAD = math.radians(ACTION_AMPLITUDE_DEG)

# Environment parameters
ORIENTATION_TERMINATION_LIMIT_DEG = 35.0
ORIENTATION_TERMINATION_LIMIT_RAD = math.radians(ORIENTATION_TERMINATION_LIMIT_DEG)
ORIENTATION_PENALTY_THRESHOLD_DEG = 15.0
ORIENTATION_PENALTY_THRESHOLD_RAD = math.radians(ORIENTATION_PENALTY_THRESHOLD_DEG)
YAW_PENALTY_THRESHOLD_DEG = 35.0
YAW_PENALTY_THRESHOLD_RAD = math.radians(YAW_PENALTY_THRESHOLD_DEG)
Z_POS_STABILITY_PENALTY_COEF = 0.25

MIN_FWD_DISPLACEMENT_FOR_BACKWARD_CHECK = 0.05
POLICY_DECISION_DT = 0.08
NUM_SETTLE_STEPS = 100

# --- Terrain Configuration ---
TERRAIN_GENERATION_EPISODIC = True
TERRAIN_HFIELD_NAME = "terrain_hfield"
TERRAIN_ROWS = 100
TERRAIN_COLS = 100
TERRAIN_MAX_ABS_HEIGHT = 1.5
TERRAIN_SMOOTHNESS_FACTOR = 0.3
TERRAIN_NUM_SMOOTH_PASSES = 4
TERRAIN_BUMP_PROBABILITY = 0.8
TERRAIN_FLAT_AREA_AT_START = 10.0


# --- Global variables for rendering ---
g_dynamic_render_active = True
g_render_lock = threading.Lock()

# --- Helper Functions ---
def quat_to_ypr(quat):
    q0, q1, q2, q3 = quat[0], quat[1], quat[2], quat[3]
    sinr_cosp = 2 * (q0*q1 + q2*q3); cosr_cosp = 1 - 2 * (q1*q1 + q2*q2); roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (q0*q2 - q3*q1); pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi/2, sinp)
    siny_cosp = 2 * (q0*q3 + q1*q2); cosy_cosp = 1 - 2 * (q2*q2 + q3*q3); yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw, pitch, roll

# --- Model and Environment Classes ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, 1024), nn.Tanh(), nn.Linear(1024, 512), nn.Tanh(), nn.Linear(512, action_dim), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(state_dim, 1024), nn.Tanh(), nn.Linear(1024, 512), nn.Tanh(), nn.Linear(512, 1))
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * math.log(action_std_init))

    def forward(self, state):
        action_mean = self.actor(state)
        action_std = torch.exp(self.action_log_std.expand_as(action_mean))
        return Normal(action_mean, action_std), self.critic(state)

class QuadrupedEnv:
    def __init__(self, model_path_xml): # Renamed argument to avoid conflict
        self.model = mujoco.MjModel.from_xml_path(model_path_xml)
        self.hfield_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, TERRAIN_HFIELD_NAME)
        if self.hfield_id == -1: print(f"WARNING: Height field asset '{TERRAIN_HFIELD_NAME}' not found.")
        else:
            if self.model.hfield_nrow[self.hfield_id] != TERRAIN_ROWS or self.model.hfield_ncol[self.hfield_id] != TERRAIN_COLS:
                raise ValueError(f"Mismatch XML hfield '{TERRAIN_HFIELD_NAME}' asset dimensions and Python TERRAIN_ROWS/COLS.")
            expected_hfield_data_size = TERRAIN_ROWS * TERRAIN_COLS
            if self.model.hfield_data is None or self.model.hfield_data.size != expected_hfield_data_size:
                 self.model.hfield_data = np.zeros(expected_hfield_data_size, dtype=np.float32)
            hfield_geom_id_for_pos = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, TERRAIN_HFIELD_NAME)
            if hfield_geom_id_for_pos != -1: self.hfield_geom_pos = self.model.geom_pos[hfield_geom_id_for_pos].copy()
            else: self.hfield_geom_pos = np.array([0.0, 0.0, 0.0])
            self.xml_hfield_asset_size = self.model.hfield_size[self.hfield_id].copy()

        self.data = mujoco.MjData(self.model); self.viewer = None
        self.sim_steps_per_policy_step = max(1, int(POLICY_DECISION_DT / self.model.opt.timestep))
        global actuator_ctrl_props
        actuator_ctrl_props.clear()
        for name in ACTUATOR_NAMES_ORDERED:
            mujoco_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if mujoco_act_id == -1: raise ValueError(f"Actuator '{name}' not found.")
            ctrlrange = self.model.actuator_ctrlrange[mujoco_act_id]
            actuator_ctrl_props[name] = {'ctrlrange': ctrlrange.copy(), 'mujoco_id': mujoco_act_id}
        
        self.state_dim = 3 + 8 + 1; self.action_dim = 8
        
        _tm = mujoco.MjModel.from_xml_path(model_path_xml); _td = mujoco.MjData(_tm)
        k_id = mujoco.mj_name2id(_tm, mujoco.mjtObj.mjOBJ_KEY, 'home'); assert k_id!=-1,"'home' keyframe not found in XML"
        mujoco.mj_resetDataKeyframe(_tm, _td, k_id)
        self.initial_qpos_home = _td.qpos.copy(); self.initial_qvel_home = _td.qvel.copy()
        self.initial_ctrl_home = _td.ctrl.copy(); self.initial_body_y_pos_home = self.initial_qpos_home[1]
        self.initial_body_z_pos_on_flat = self.initial_qpos_home[2]; del _tm, _td
        
        self.current_initial_body_z_pos = self.initial_body_z_pos_on_flat
        self.current_episode_steps = 0; self.last_commanded_clipped_sim_rad = self.initial_ctrl_home.copy()
        self.previous_x_qpos_in_episode = 0.0; self.cumulative_positive_x_displacement = 0.0
        self.cumulative_negative_x_displacement = 0.0; self.previous_net_forward_displacement = 0.0
        self.obstacle_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle")

    def _get_observation(self):
        yaw,pitch,roll = quat_to_ypr(self.data.qpos[3:7])
        joint_pos_deltas = []
        for act_name in ACTUATOR_NAMES_ORDERED:
            joint_name = ACTUATOR_TO_JOINT_NAME_MAP[act_name]
            qpos_idx = JOINT_NAME_TO_QPOS_IDX_MAP[joint_name]
            home_qpos = sim_keyframe_home_qpos_map[joint_name]
            joint_pos_deltas.append(self.data.qpos[qpos_idx] - home_qpos)
        
        body_x_vel = self.data.qvel[0]
        state = np.concatenate([[yaw, pitch, roll], np.array(joint_pos_deltas), [body_x_vel]])
        return state.astype(np.float32)

    def _generate_random_terrain(self):
        if self.hfield_id == -1: return
        height_data_raw = np.zeros((TERRAIN_ROWS, TERRAIN_COLS), dtype=np.float32)
        hfield_world_pos_x = self.hfield_geom_pos[0]
        hfield_world_pos_y = self.hfield_geom_pos[1]
        xml_hfield_x_extent = self.xml_hfield_asset_size[0]
        xml_hfield_y_extent = self.xml_hfield_asset_size[1]
        
        cell_size_x = xml_hfield_x_extent / (TERRAIN_COLS - 1)
        cell_size_y = xml_hfield_y_extent / (TERRAIN_ROWS - 1)
        
        robot_start_x = self.initial_qpos_home[0]
        robot_start_y = self.initial_qpos_home[1]
        flat_circle_radius = random.uniform(0.1, 0.4)
        
        for r in range(TERRAIN_ROWS):
            for c in range(TERRAIN_COLS):
                cell_wx = hfield_world_pos_x - (xml_hfield_x_extent/2.0) + c * cell_size_x
                cell_wy = hfield_world_pos_y - (xml_hfield_y_extent/2.0) + r * cell_size_y
                distance = np.sqrt((cell_wx - robot_start_x)**2 + (cell_wy - robot_start_y)**2)
                
                if distance >= flat_circle_radius:
                    base_height = random.uniform(-TERRAIN_MAX_ABS_HEIGHT, TERRAIN_MAX_ABS_HEIGHT)
                    freq_x = random.uniform(0.2, 0.6); freq_y = random.uniform(0.2, 0.6)
                    position_noise = (np.sin(cell_wx * freq_x) * np.cos(cell_wy * freq_y) + 
                                    np.sin(cell_wx * freq_x * 2) * np.cos(cell_wy * freq_y * 2)) * TERRAIN_MAX_ABS_HEIGHT * 0.7
                    spike = 0
                    if random.random() < 0.2: spike = random.uniform(-TERRAIN_MAX_ABS_HEIGHT * 0.8, TERRAIN_MAX_ABS_HEIGHT * 0.8)
                    height_data_raw[r, c] = base_height + position_noise + spike
                    if abs(distance - flat_circle_radius) < 1.0: height_data_raw[r, c] *= 1.5
        
        smoothed = height_data_raw.copy()
        if TERRAIN_SMOOTHNESS_FACTOR > 0 and TERRAIN_NUM_SMOOTH_PASSES > 0:
            for _ in range(TERRAIN_NUM_SMOOTH_PASSES):
                temp = smoothed.copy()
                for r_idx in range(1,TERRAIN_ROWS-1):
                    for c_idx in range(1,TERRAIN_COLS-1):
                        cell_wx = hfield_world_pos_x-(xml_hfield_x_extent/2.0)+c_idx*cell_size_x
                        cell_wy = hfield_world_pos_y-(xml_hfield_y_extent/2.0)+r_idx*cell_size_y
                        distance = np.sqrt((cell_wx-robot_start_x)**2 + (cell_wy-robot_start_y)**2)
                        if distance >= flat_circle_radius:
                            avg = np.mean(temp[r_idx-1:r_idx+2, c_idx-1:c_idx+2])
                            smoothed[r_idx,c_idx] = temp[r_idx,c_idx]*(1-TERRAIN_SMOOTHNESS_FACTOR) + avg*TERRAIN_SMOOTHNESS_FACTOR
        
        h_to_norm = smoothed; min_h, max_h = np.min(h_to_norm), np.max(h_to_norm)
        if max_h <= min_h + 1e-4: norm_h_data = np.full_like(h_to_norm, 0.5)
        else: norm_h_data = (h_to_norm - min_h) / (max_h - min_h)
        
        self.model.hfield_data[:] = norm_h_data.T.flatten()
        if self.viewer and self.viewer.is_running():
            try:
                mujoco.mjr_uploadHField(self.model, self.viewer.user_scn.scn.context, self.hfield_id)
                self.viewer.sync()
            except Exception as e: print(f"Error uploading heightfield: {e}")
        mujoco.mj_resetData(self.model, self.data)
        self.sync_viewer_if_active()

    def _randomize_discrete_obstacle(self):
        if self.obstacle_geom_id != -1: 
            self.model.geom_rgba[self.obstacle_geom_id,3]=0.0
            self.model.geom_pos[self.obstacle_geom_id,2]=-100.0

    def reset_to_home_keyframe(self):
        if TERRAIN_GENERATION_EPISODIC and self.hfield_id!=-1: self._generate_random_terrain()
        
        self.data.qpos[:]=self.initial_qpos_home; self.data.qvel[:]=self.initial_qvel_home
        self.data.ctrl[:]=self.initial_ctrl_home; mujoco.mj_forward(self.model,self.data)
        
        for _ in range(NUM_SETTLE_STEPS): 
            self.data.ctrl[:]=self.initial_ctrl_home
            try: mujoco.mj_step(self.model,self.data)
            except mujoco.FatalError: break
            self.sync_viewer_if_active()
        
        mujoco.mj_forward(self.model,self.data)
        self.previous_x_qpos_in_episode=self.data.qpos[0]
        self.current_initial_body_z_pos=self.data.qpos[2]
        self.last_commanded_clipped_sim_rad=self.data.ctrl.copy()
        self._randomize_discrete_obstacle()
        mujoco.mj_forward(self.model,self.data)
        return self._get_observation()

    def reset(self):
        self.current_episode_steps=0
        seed = int(time.time() * 1000) % (2**32 - 1)
        random.seed(seed); np.random.seed(seed)
        self.reset_to_home_keyframe()
        self.cumulative_positive_x_displacement=0.0; self.cumulative_negative_x_displacement=0.0
        self.previous_net_forward_displacement=0.0
        return self._get_observation()

    def _apply_actions_and_step(self, policy_actions_rad):
        final_commands_sim_rad = np.zeros_like(self.data.ctrl)
        for i, act_name in enumerate(ACTUATOR_NAMES_ORDERED):
            props = actuator_ctrl_props[act_name]
            mujoco_act_id = props['mujoco_id']
            joint_name_for_actuator = ACTUATOR_TO_JOINT_NAME_MAP[act_name]
            sim_home_qpos_rad = sim_keyframe_home_qpos_map[joint_name_for_actuator]
            
            action_delta_rad = policy_actions_rad[i]
            sim_target_rad_unclipped = sim_home_qpos_rad + action_delta_rad
            
            sim_target_rad_clipped = np.clip(sim_target_rad_unclipped, 
                                             props['ctrlrange'][0], 
                                             props['ctrlrange'][1])
            final_commands_sim_rad[mujoco_act_id] = sim_target_rad_clipped
        
        self.data.ctrl[:] = final_commands_sim_rad
        sim_exception_count = 0
        for _ in range(self.sim_steps_per_policy_step):
            try:
                mujoco.mj_step(self.model, self.data)
            except mujoco.FatalError:
                sim_exception_count += 1
                break 
            self.sync_viewer_if_active()
        return final_commands_sim_rad, sim_exception_count

    def step(self, policy_actions_scaled_neg1_to_1):
        self.current_episode_steps += 1
        policy_actions_rad = policy_actions_scaled_neg1_to_1 * ACTION_AMPLITUDE_RAD
        
        final_commands_sim_rad, sim_exception_count = self._apply_actions_and_step(policy_actions_rad)
        obs = self._get_observation()
        
        current_x_qpos = self.data.qpos[0]
        delta_x = current_x_qpos - self.previous_x_qpos_in_episode
        if delta_x > 0: self.cumulative_positive_x_displacement += delta_x
        elif delta_x < 0: self.cumulative_negative_x_displacement += abs(delta_x)
        self.previous_x_qpos_in_episode = current_x_qpos
        
        done = False
        info = {"sim_target_rad": final_commands_sim_rad.copy(), "termination_reason": "max_steps_playback"}

        if sim_exception_count > 0:
            done = True; info["mj_error"] = True; info["termination_reason"] = "mj_error_playback"
        
        current_yaw, current_pitch, current_roll = quat_to_ypr(self.data.qpos[3:7])
        if not done and (abs(current_roll) > ORIENTATION_TERMINATION_LIMIT_RAD or \
                         abs(current_pitch) > ORIENTATION_TERMINATION_LIMIT_RAD or \
                         abs(current_yaw) > YAW_PENALTY_THRESHOLD_RAD * 1.5):
            done = True; info["termination_reason"] = "orientation_limit_playback"

        if not done and self.cumulative_positive_x_displacement > MIN_FWD_DISPLACEMENT_FOR_BACKWARD_CHECK and \
           self.cumulative_negative_x_displacement > 0.85 * self.cumulative_positive_x_displacement:
            done = True; info["termination_reason"] = "too_much_backward_playback"
        
        reward = 0.0 
            
        self.last_commanded_clipped_sim_rad = final_commands_sim_rad.copy()
        return obs, reward, done, info

    def launch_viewer_internal(self):
        if self.viewer is None:
            try: self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as e: self.viewer = None; print(f"Failed to launch viewer: {e}")
        elif self.viewer and not self.viewer.is_running():
            self.viewer = None; self.launch_viewer_internal()

    def close_viewer_internal(self):
        if self.viewer and self.viewer.is_running():
            try: self.viewer.close()
            except Exception: pass
            finally: self.viewer = None
            
    def sync_viewer_if_active(self):
        global g_dynamic_render_active, g_render_lock
        with g_render_lock:
            if g_dynamic_render_active:
                if self.viewer is None or not self.viewer.is_running():
                    self.launch_viewer_internal()
                if self.viewer and self.viewer.is_running():
                    try: self.viewer.sync()
                    except Exception: self.close_viewer_internal()
            elif self.viewer and self.viewer.is_running():
                self.close_viewer_internal()

# --- Rendering Toggle ---
def toggle_render():
    global g_dynamic_render_active, g_render_lock
    with g_render_lock:
        g_dynamic_render_active = not g_dynamic_render_active
    print(f"\nDynamic rendering toggled {'ON' if g_dynamic_render_active else 'OFF'}.")

# --- Initialization of Global Maps ---
def initialize_global_maps_and_config(xml_file_path_init): # Renamed argument
    global sim_keyframe_home_qpos_map, ACTUATOR_TO_JOINT_NAME_MAP, JOINT_NAME_TO_QPOS_IDX_MAP
    
    print(f"--- Verifying XML & Mappings for {xml_file_path_init} ---")
    if not os.path.exists(xml_file_path_init):
        raise FileNotFoundError(f"CRITICAL: XML '{xml_file_path_init}' not found.")
    
    _model_temp = mujoco.MjModel.from_xml_path(xml_file_path_init)
    _data_temp = mujoco.MjData(_model_temp)

    ACTUATOR_TO_JOINT_NAME_MAP.clear()
    JOINT_NAME_TO_QPOS_IDX_MAP.clear()
    for act_name in ACTUATOR_NAMES_ORDERED:
        actuator_id = mujoco.mj_name2id(_model_temp, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if actuator_id == -1: raise ValueError(f"Actuator '{act_name}' not found in XML.")
        
        joint_id = _model_temp.actuator_trnid[actuator_id, 0]
        joint_name = mujoco.mj_id2name(_model_temp, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not joint_name: raise ValueError(f"Joint for actuator '{act_name}' not found.")
        
        ACTUATOR_TO_JOINT_NAME_MAP[act_name] = joint_name
        JOINT_NAME_TO_QPOS_IDX_MAP[joint_name] = _model_temp.jnt_qposadr[joint_id]

    keyframe_id = mujoco.mj_name2id(_model_temp, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if keyframe_id == -1: raise ValueError("'home' keyframe not found in XML.")
    mujoco.mj_resetDataKeyframe(_model_temp, _data_temp, keyframe_id)
    
    sim_keyframe_home_qpos_map.clear()
    print("Home keyframe qpos (sim_keyframe_home_qpos_map - from flat ground):")
    for act_name in ACTUATOR_NAMES_ORDERED:
        joint_name = ACTUATOR_TO_JOINT_NAME_MAP[act_name]
        qpos_idx = JOINT_NAME_TO_QPOS_IDX_MAP[joint_name]
        sim_keyframe_home_qpos_map[joint_name] = _data_temp.qpos[qpos_idx]
        print(f"  {joint_name:<15} (Act: {act_name:<18}): {sim_keyframe_home_qpos_map[joint_name]:.4f} rad")
    
    del _model_temp, _data_temp
    print("--- Initialization of Global Maps Complete ---")


# --- Main Execution Logic ---
def run_simulation(model_pth_path_run, xml_file_path_run, max_steps_run): # Renamed arguments
    global g_dynamic_render_active
    
    try:
        locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_NUMERIC, 'C')
        except locale.Error:
            print(f"Warning: Could not set a suitable LC_NUMERIC.")

    initialize_global_maps_and_config(xml_file_path_run)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = QuadrupedEnv(xml_file_path_run)
    state_dim = env.state_dim
    action_dim = env.action_dim

    agent = ActorCritic(state_dim, action_dim, INITIAL_ACTION_STD_INIT).to(device)
    
    print(f"Loading model from: {model_pth_path_run}")
    if not os.path.exists(model_pth_path_run):
        print(f"ERROR: Model file not found at {model_pth_path_run}")
        print(f"Please ensure '{model_pth_path_run}' exists or update the MODEL_PTH_TO_LOAD variable in the script.")
        return
        
    try:
        agent.load_state_dict(torch.load(model_pth_path_run, map_location=device))
        agent.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if KEYBOARD_LIB_AVAILABLE:
        try:
            keyboard.add_hotkey('ctrl+l', toggle_render)
            print("INFO: Ctrl+L hotkey active to toggle rendering.")
        except Exception as e:
            print(f"WARNING: Failed to set Ctrl+L hotkey: {e}.")

    state = env.reset()
    g_dynamic_render_active = True
    env.launch_viewer_internal()

    try:
        for step_num in range(max_steps_run):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_dist, _ = agent(state_tensor)
                action = action_dist.mean
            
            action_np = action.squeeze(0).cpu().numpy()
            next_state, reward, done, info = env.step(action_np)
            
            state = next_state

            if done:
                print(f"Episode finished after {step_num + 1} steps. Reason: {info.get('termination_reason', 'unknown')}")
                print("Resetting environment...")
                time.sleep(1)
                state = env.reset()
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        print("Closing viewer and cleaning up.")
        if KEYBOARD_LIB_AVAILABLE:
            try: keyboard.remove_hotkey('ctrl+l')
            except: pass
        env.close_viewer_internal()

if __name__ == "__main__":
    # No argparse, directly call run_simulation with hardcoded values
    print(f"Starting simulation with hardcoded model: {MODEL_PTH_TO_LOAD}")
    print(f"Using XML: {XML_FILE_NAME}")
    print(f"Max steps per run: {MAX_SIMULATION_STEPS}")
    
    # Ensure the XML file path is correct (e.g., if it's in the same dir as script)
    # If XML_FILE_NAME is just the name, os.path.join is not strictly needed but good practice
    xml_full_path = XML_FILE_NAME
    # If XML_FILE_NAME could be an absolute path already, this might need adjustment,
    # but for "walking_scene.xml" it's likely relative.

    # The MODEL_PTH_TO_LOAD is also assumed to be relative to where the script is run
    # or an absolute path. If it's relative to the script directory, you might want:
    # model_full_path = os.path.join(script_dir, MODEL_PTH_TO_LOAD)
    # For now, using MODEL_PTH_TO_LOAD as is.

    run_simulation(MODEL_PTH_TO_LOAD, xml_full_path, MAX_SIMULATION_STEPS)