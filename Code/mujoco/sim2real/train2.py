import locale
import os

# print(f"Attempting to set LC_NUMERIC. Current locale: {locale.getlocale(locale.LC_NUMERIC)}")
try:
    locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8') 
    # print(f"Successfully set LC_NUMERIC to: {locale.getlocale(locale.LC_NUMERIC)}")
except locale.Error as e1:
    # print(f"Failed to set locale to en_US.UTF-8: {e1}")
    try:
        locale.setlocale(locale.LC_NUMERIC, 'C')
        # print(f"Successfully set LC_NUMERIC to: {locale.getlocale(locale.LC_NUMERIC)} (C locale)")
    except locale.Error as e2:
        print(f"Failed to set locale to C: {e2}")
        # print(f"ERROR: Could not set a suitable LC_NUMERIC. Current setting: {locale.getlocale(locale.LC_NUMERIC)}. ")

import mujoco 
import mujoco.viewer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import math
import time
import json
from collections import deque
import threading
import random
import time

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
OUTPUT_BASE_DIR = './output_terrain_v4' 
JSON_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'json')
PTH_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'pth')
JSON_OUTPUT_PATH_TEMPLATE = os.path.join(JSON_OUTPUT_DIR, 'walk_rl_terrain_ep{}.json')
PTH_SAVE_TEMPLATE = os.path.join(PTH_OUTPUT_DIR, 'quadruped_ac_terrain_ep{}.pth')
FINAL_PTH_SAVE_NAME = os.path.join(PTH_OUTPUT_DIR, 'quadruped_ac_terrain_final.pth')

XML_FILE_NAME = 'walking_scene.xml' 

PTH_SAVE_INTERVAL = 100
JSON_MAX_STEPS_EPISODIC = 150
JSON_MAX_STEPS_FINAL = 300

# --- ADAPTIVE HYPERPARAMETER CONFIG ---
ADAPTATION_CHECK_INTERVAL = 10
AVGR_HISTORY_LEN = 5
MIN_LR = 1e-6; MAX_LR = 3e-4
MIN_ENTROPY_COEF = 0.0001; MAX_ENTROPY_COEF = 0.01
MIN_ACTION_LOG_STD = math.log(0.05); MAX_ACTION_LOG_STD = math.log(0.4)
INITIAL_LEARNING_RATE = 1e-4
INITIAL_ENTROPY_COEF = 0.001
INITIAL_ACTION_STD_INIT = 0.3

# RL Hyperparameters (others)
GAMMA = 0.99; GAE_LAMBDA = 0.95
VALUE_LOSS_COEF = 0.5; MAX_GRAD_NORM = 0.5; NUM_EPISODES = 25000
MAX_STEPS_PER_EPISODE = 1000
POLICY_UPDATE_INTERVAL = 2048
NUM_EPOCHS_PER_UPDATE = 10

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
Z_POS_STABILITY_PENALTY_COEF = 0.25 # Increased penalty coefficient

MIN_FWD_DISPLACEMENT_FOR_BACKWARD_CHECK = 0.05
POLICY_DECISION_DT = 0.08
NUM_SETTLE_STEPS = 100

# --- Terrain Configuration ---
TERRAIN_GENERATION_EPISODIC = True
TERRAIN_HFIELD_NAME = "terrain_hfield" 
TERRAIN_ROWS = 100                      
TERRAIN_COLS = 100                       
TERRAIN_MAX_ABS_HEIGHT = 1.5 # Increased max height for more dramatic terrain
TERRAIN_SMOOTHNESS_FACTOR = 0.3 # Reduced smoothing for rougher terrain
TERRAIN_NUM_SMOOTH_PASSES = 4 # Fewer smooth passes for more dramatic features
TERRAIN_BUMP_PROBABILITY = 0.8 # Much higher probability for terrain features
TERRAIN_FLAT_AREA_AT_START = 10.0 # Increased to match the circular flat area


# --- SIM-TO-REAL MAPPING CONFIGURATION ---
real_robot_home_deg_map = {name: 0.0 for name in ACTUATOR_NAMES_ORDERED}
real_robot_home_deg_map.update({
    "FR_tigh_actuator": -45.0, "FR_knee_actuator": 45.0, "FL_tigh_actuator": 45.0,  "FL_knee_actuator": 45.0,
    "BR_tigh_actuator": 45.0,  "BR_knee_actuator": -45.0, "BL_tigh_actuator": 45.0,  "BL_knee_actuator": -45.0,
})
joint_scale_factors = {name: 1.0 for name in ACTUATOR_NAMES_ORDERED}
sim_keyframe_home_qpos_map = {} 
actuator_ctrl_props = {}

g_dynamic_render_active = False
g_render_lock = threading.Lock()

def quat_to_ypr(quat): # Standard conversion
    q0, q1, q2, q3 = quat[0], quat[1], quat[2], quat[3]
    sinr_cosp = 2 * (q0*q1 + q2*q3); cosr_cosp = 1 - 2 * (q1*q1 + q2*q2); roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (q0*q2 - q3*q1); pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi/2, sinp)
    siny_cosp = 2 * (q0*q3 + q1*q2); cosy_cosp = 1 - 2 * (q2*q2 + q3*q3); yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw, pitch, roll

def convert_sim_rad_to_real_deg(act_name, sim_rad_commanded, _sim_keyframe_home_qpos_map, _real_robot_home_deg_map, _scale_factors):
    joint_name_for_actuator = ACTUATOR_TO_JOINT_NAME_MAP.get(act_name)
    if joint_name_for_actuator is None: return None
    sim_home_qpos_rad = _sim_keyframe_home_qpos_map.get(joint_name_for_actuator)
    real_home_deg_offset = _real_robot_home_deg_map.get(act_name)
    scale = _scale_factors.get(act_name, 1.0)
    if sim_home_qpos_rad is None or real_home_deg_offset is None or not isinstance(sim_rad_commanded, (int, float)): return None
    sim_delta_rad = sim_rad_commanded - sim_home_qpos_rad; real_delta_deg = scale * math.degrees(sim_delta_rad)
    real_target_deg = real_home_deg_offset + real_delta_deg
    return real_target_deg

class ActorCritic(nn.Module): # Standard AC network
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, 1024), nn.Tanh(), nn.Linear(1024, 512), nn.Tanh(), nn.Linear(512, action_dim), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(state_dim, 1024), nn.Tanh(), nn.Linear(1024, 512), nn.Tanh(), nn.Linear(512, 1))
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * math.log(action_std_init))
    def forward(self, state):
        action_mean = self.actor(state); action_std = torch.exp(self.action_log_std.expand_as(action_mean))
        return Normal(action_mean, action_std), self.critic(state)

class QuadrupedEnv:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
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
        for name in ACTUATOR_NAMES_ORDERED:
            mujoco_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if mujoco_act_id == -1: raise ValueError(f"Actuator '{name}' not found.")
            ctrlrange = self.model.actuator_ctrlrange[mujoco_act_id]
            actuator_ctrl_props[name] = {'ctrlrange': ctrlrange.copy(), 'mujoco_id': mujoco_act_id}
        self.state_dim = 3 + 8 + 1; self.action_dim = 8
        _tm = mujoco.MjModel.from_xml_path(model_path); _td = mujoco.MjData(_tm)
        k_id = mujoco.mj_name2id(_tm, mujoco.mjtObj.mjOBJ_KEY, 'home'); assert k_id!=-1
        mujoco.mj_resetDataKeyframe(_tm, _td, k_id)
        self.initial_qpos_home = _td.qpos.copy(); self.initial_qvel_home = _td.qvel.copy()
        self.initial_ctrl_home = _td.ctrl.copy(); self.initial_body_y_pos_home = self.initial_qpos_home[1]
        self.initial_body_z_pos_on_flat = self.initial_qpos_home[2]; del _tm, _td
        self.current_initial_body_z_pos = self.initial_body_z_pos_on_flat
        self.current_episode_steps = 0; self.last_commanded_clipped_sim_rad = self.initial_ctrl_home.copy()
        self.previous_x_qpos_in_episode = 0.0; self.cumulative_positive_x_displacement = 0.0
        self.cumulative_negative_x_displacement = 0.0; self.previous_net_forward_displacement = 0.0
        self.obstacle_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle") 
   

    def _get_observation(self): # Standard observation
        yaw,pitch,roll=quat_to_ypr(self.data.qpos[3:7]); jp=[];
        for act_n in ACTUATOR_NAMES_ORDERED: jn=ACTUATOR_TO_JOINT_NAME_MAP[act_n]; qi=JOINT_NAME_TO_QPOS_IDX_MAP[jn]; jp.append(self.data.qpos[qi]-sim_keyframe_home_qpos_map[jn])
        jpd=np.array(jp); tvx=self.data.qvel[0]; state=np.concatenate([[yaw,pitch,roll],jpd,[tvx]])
        return state.astype(np.float32)

    def _generate_random_terrain(self): # Modified for proper circular flat area with chaos beyond
        if self.hfield_id == -1: return
        height_data_raw = np.zeros((TERRAIN_ROWS, TERRAIN_COLS), dtype=np.float32)
        hfield_world_pos_x = self.hfield_geom_pos[0]
        hfield_world_pos_y = self.hfield_geom_pos[1]
        xml_hfield_x_extent = self.xml_hfield_asset_size[0]
        xml_hfield_y_extent = self.xml_hfield_asset_size[1]
        
        # Calculate grid cell size in world units
        cell_size_x = xml_hfield_x_extent / (TERRAIN_COLS - 1)
        cell_size_y = xml_hfield_y_extent / (TERRAIN_ROWS - 1)
        
        # Robot start position in world coordinates
        robot_start_x = self.initial_qpos_home[0]
        robot_start_y = self.initial_qpos_home[1]
        flat_circle_radius = random.uniform(0.1, 0.4) # Randomize radius for larger flat area
        
        for r in range(TERRAIN_ROWS):
            for c in range(TERRAIN_COLS):
                # Calculate world coordinates for this cell
                cell_wx = hfield_world_pos_x - (xml_hfield_x_extent/2.0) + c * cell_size_x
                cell_wy = hfield_world_pos_y - (xml_hfield_y_extent/2.0) + r * cell_size_y
                
                # Calculate distance from robot start position
                distance = np.sqrt((cell_wx - robot_start_x)**2 + (cell_wy - robot_start_y)**2)
                
                # Generate height based on distance
                if distance >= flat_circle_radius:
                    # Base height from random noise
                    base_height = random.uniform(-TERRAIN_MAX_ABS_HEIGHT, TERRAIN_MAX_ABS_HEIGHT)
                    
                    # Add more dramatic position-based variation
                    freq_x = random.uniform(0.2, 0.6)
                    freq_y = random.uniform(0.2, 0.6)
                    position_noise = (np.sin(cell_wx * freq_x) * np.cos(cell_wy * freq_y) + 
                                    np.sin(cell_wx * freq_x * 2) * np.cos(cell_wy * freq_y * 2)) * TERRAIN_MAX_ABS_HEIGHT * 0.7
                    
                    # Add random spikes with low probability
                    spike = 0
                    if random.random() < 0.2:  # 20% chance of spike
                        spike = random.uniform(-TERRAIN_MAX_ABS_HEIGHT * 0.8, TERRAIN_MAX_ABS_HEIGHT * 0.8)
                    
                    height_data_raw[r, c] = base_height + position_noise + spike
                    
                    # Add extra chaos at the boundary
                    if abs(distance - flat_circle_radius) < 1.0:
                        height_data_raw[r, c] *= 1.5
        
        smoothed = height_data_raw.copy()
        if TERRAIN_SMOOTHNESS_FACTOR > 0 and TERRAIN_NUM_SMOOTH_PASSES > 0:
            # Only smooth the non-flat areas
            for _ in range(TERRAIN_NUM_SMOOTH_PASSES):
                temp = smoothed.copy()
                for r_idx in range(1,TERRAIN_ROWS-1):
                    for c_idx in range(1,TERRAIN_COLS-1):
                        # Calculate world coordinates for this cell
                        cell_wx = hfield_world_pos_x - (xml_hfield_x_extent/2.0) + c_idx * cell_size_x
                        cell_wy = hfield_world_pos_y - (xml_hfield_y_extent/2.0) + r_idx * cell_size_y
                        distance = np.sqrt((cell_wx - robot_start_x)**2 + (cell_wy - robot_start_y)**2)
                        
                        # Only smooth if we're outside the flat circle
                        if distance >= flat_circle_radius:
                            avg = np.mean(temp[r_idx-1:r_idx+2, c_idx-1:c_idx+2])
                            smoothed[r_idx,c_idx] = temp[r_idx,c_idx]*(1-TERRAIN_SMOOTHNESS_FACTOR) + avg*TERRAIN_SMOOTHNESS_FACTOR
        h_to_norm = smoothed; min_h, max_h = np.min(h_to_norm), np.max(h_to_norm)
        if max_h <= min_h + 1e-4: norm_h_data = np.full_like(h_to_norm, 0.5)
        else: norm_h_data = (h_to_norm - min_h) / (max_h - min_h)
        # Update heightfield and reset physics engine to ensure changes take effect
        self.model.hfield_data[:] = norm_h_data.T.flatten()

        # Upload the new heightfield data to the GPU for the viewer
        if self.viewer and self.viewer.is_running():
            try:
                mujoco.mjr_uploadHField(self.model, self.viewer.user_scn.scn.context, self.hfield_id)
                # Force a redraw of the viewer to reflect the updated heightfield
                self.viewer.sync()
            except Exception as e:
                print(f"Error uploading heightfield: {e}")

        mujoco.mj_resetData(self.model, self.data)
        self.sync_viewer_if_active()
        # world_z_min = self.hfield_geom_pos[2] + self.xml_hfield_asset_size[3] + np.min(norm_h_data) * self.xml_hfield_asset_size[2]
        # world_z_max = self.hfield_geom_pos[2] + self.xml_hfield_asset_size[3] + np.max(norm_h_data) * self.xml_hfield_asset_size[2]
        # print(f"DEBUG Terrain: Raw min/max {min_h:.2f}/{max_h:.2f}. World Z {world_z_min:.2f}/{world_z_max:.2f}")


    def get_terrain_height(self, world_x, world_y): # Standard hfield height lookup
        if self.hfield_id == -1: return self.hfield_geom_pos[2] if hasattr(self, 'hfield_geom_pos') else 0.0
        h_w_pos=self.hfield_geom_pos; xml_s=self.xml_hfield_asset_size
        hf_lx=world_x-h_w_pos[0]; hf_ly=world_y-h_w_pos[1]
        hf_c=self.model.hfield_ncol[self.hfield_id]; hf_r=self.model.hfield_nrow[self.hfield_id]
        c_if=(hf_lx+xml_s[0]/2.0)/xml_s[0]*(hf_c-1); r_if=(hf_ly+xml_s[1]/2.0)/xml_s[1]*(hf_r-1)
        c=int(np.clip(c_if,0,hf_c-1)); r=int(np.clip(r_if,0,hf_r-1))
        norm_val=self.model.hfield_data[c*hf_r+r]
        h_off=xml_s[3]+norm_val*xml_s[2]; fin_h=h_w_pos[2]+h_off
        return fin_h

    def _randomize_discrete_obstacle(self): # Effectively disabled
        if self.obstacle_geom_id != -1: self.model.geom_rgba[self.obstacle_geom_id,3]=0.0; self.model.geom_pos[self.obstacle_geom_id,2]=-100.0

    def reset_to_home_keyframe(self): # Standard reset with settling
        if TERRAIN_GENERATION_EPISODIC and self.hfield_id!=-1: self._generate_random_terrain()
        self.data.qpos[:]=self.initial_qpos_home; self.data.qvel[:]=self.initial_qvel_home
        self.data.ctrl[:]=self.initial_ctrl_home; mujoco.mj_forward(self.model,self.data)
        for _ in range(NUM_SETTLE_STEPS): 
            self.data.ctrl[:]=self.initial_ctrl_home
            try: mujoco.mj_step(self.model,self.data)
            except mujoco.FatalError: break
            self.sync_viewer_if_active()
        mujoco.mj_forward(self.model,self.data)
        self.previous_x_qpos_in_episode=self.data.qpos[0]; self.current_initial_body_z_pos=self.data.qpos[2]
        self.last_commanded_clipped_sim_rad=self.data.ctrl.copy(); self._randomize_discrete_obstacle()
        mujoco.mj_forward(self.model,self.data); return self._get_observation()

    def reset(self): # Standard RL reset
        self.current_episode_steps=0
        # Seed random number generator with current time to ensure different terrain each episode
        seed = int(time.time() * 1000) % (2**32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        self.reset_to_home_keyframe()
        self.cumulative_positive_x_displacement=0.0; self.cumulative_negative_x_displacement=0.0
        self.previous_net_forward_displacement=0.0; return self._get_observation()

    def _apply_actions_and_step(self, policy_actions_rad): # Applies actions
        f_cmds=np.zeros_like(self.data.ctrl)
        for i,act_n in enumerate(ACTUATOR_NAMES_ORDERED):
            p=actuator_ctrl_props[act_n]; mid=p['mujoco_id']; jn=ACTUATOR_TO_JOINT_NAME_MAP[act_n]
            shr=sim_keyframe_home_qpos_map[jn]; adr=policy_actions_rad[i]; stu=shr+adr
            cst=np.clip(stu,p['ctrlrange'][0],p['ctrlrange'][1]); f_cmds[mid]=cst
        self.data.ctrl[:]=f_cmds; sum_e=0
        for _ in range(self.sim_steps_per_policy_step):
            try: mujoco.mj_step(self.model,self.data)
            except mujoco.FatalError: sum_e+=1; break
            self.sync_viewer_if_active()
        return f_cmds,sum_e

    def step(self, policy_actions_scaled_neg1_to_1): # RL step
        self.current_episode_steps+=1; pol_rad=policy_actions_scaled_neg1_to_1*ACTION_AMPLITUDE_RAD
        f_cmds,sum_e=self._apply_actions_and_step(pol_rad); obs=self._get_observation()
        cx=self.data.qpos[0]; dx=cx-self.previous_x_qpos_in_episode
        if dx>0: self.cumulative_positive_x_displacement+=dx
        elif dx<0: self.cumulative_negative_x_displacement+=abs(dx)
        self.previous_x_qpos_in_episode=cx; fvx=self.data.qvel[0]; reward_forward_velocity=450.0*fvx
        cnd=self.cumulative_positive_x_displacement-self.cumulative_negative_x_displacement
        dnd=cnd-self.previous_net_forward_displacement; self.previous_net_forward_displacement=cnd
        reward_net_cumulative_progress=0.0; _=dnd>0.0005 and (reward_net_cumulative_progress:=20.0*dnd); penalty_backward_velocity=0.0; _=fvx<-0.005 and (penalty_backward_velocity:=-9.0*abs(fvx))
        
        # New reward term for instantaneous step displacement
        reward_step_displacement = 0.0
        if dx > 0: reward_step_displacement = 50.0 * dx # Reward for forward displacement
        elif dx < 0.0005: reward_step_displacement = -1.0  # Penalty for backward displacement (dx is negative)

        reward_alive=0.005; penalty_sideways_velocity=-0.3*abs(self.data.qvel[1]); penalty_y_position_stability=-0.15*abs(self.data.qpos[1]-self.initial_body_y_pos_home)
        
        z_dev_settled = self.data.qpos[2] - self.current_initial_body_z_pos
        z_dev_ideal = self.data.qpos[2] - self.initial_body_z_pos_on_flat
        penalty_z_position_stability = 0.0
        if z_dev_settled < -0.03: penalty_z_position_stability -= (Z_POS_STABILITY_PENALTY_COEF*0.5) * (abs(z_dev_settled) - 0.03)**2
        h_td = 0.05 
        if abs(z_dev_ideal) > h_td: penalty_z_position_stability -= (Z_POS_STABILITY_PENALTY_COEF*0.25) * (abs(z_dev_ideal) - h_td)**2
        
        cy,cp,cr=quat_to_ypr(self.data.qpos[3:7]); penalty_orientation_error=0.0; opf=-0.08
        if abs(cr)>ORIENTATION_PENALTY_THRESHOLD_RAD: penalty_orientation_error+=opf*(abs(cr)-ORIENTATION_PENALTY_THRESHOLD_RAD)**2
        if abs(cp)>ORIENTATION_PENALTY_THRESHOLD_RAD: penalty_orientation_error+=opf*(abs(cp)-ORIENTATION_PENALTY_THRESHOLD_RAD)**2
        if abs(cy)>YAW_PENALTY_THRESHOLD_RAD: penalty_orientation_error+=opf*(abs(cy)-YAW_PENALTY_THRESHOLD_RAD)**2
        ads=np.sum([(f_cmds[actuator_ctrl_props[n]['mujoco_id']]-self.last_commanded_clipped_sim_rad[actuator_ctrl_props[n]['mujoco_id']])**2 for n in ACTUATOR_NAMES_ORDERED])
        penalty_action_smoothness=-0.005*ads;cwo=False # Discrete obstacles off
        
        # New penalty for low joint velocity magnitude
        joint_vel_magnitude = np.sum(np.abs(self.data.qvel[7:15])) # Assuming joint velocities are qvel[7] to qvel[14]
        penalty_low_joint_velocity = -0.05 * np.exp(-joint_vel_magnitude * 5.0) # Exponential penalty for low velocity

        tot_r = (
            reward_forward_velocity +
            reward_net_cumulative_progress +
            penalty_backward_velocity +
            reward_alive + 0.01 + # Increased reward for staying alive
            penalty_sideways_velocity +
            penalty_y_position_stability +
            penalty_z_position_stability +
            penalty_orientation_error +
            penalty_action_smoothness +
            reward_step_displacement + # Add the new reward term
            penalty_low_joint_velocity # Add the new penalty term
        )
        done=False; info={"sim_target_rad":f_cmds.copy(),"termination_reason":"max_steps"}
        if sum_e>0: tot_r-=50.0; done=True; info["mj_error"]=True; info["termination_reason"]="mj_error" # Increased penalty
        if not done and (abs(cr)>ORIENTATION_TERMINATION_LIMIT_RAD or abs(cp)>ORIENTATION_TERMINATION_LIMIT_RAD or abs(cy)>ORIENTATION_TERMINATION_LIMIT_RAD*1.5):
            tot_r-=100.0; done=True; info["termination_reason"]="orientation_limit" # Increased penalty
        if not done and self.cumulative_positive_x_displacement>MIN_FWD_DISPLACEMENT_FOR_BACKWARD_CHECK and self.cumulative_negative_x_displacement > 0.85*self.cumulative_positive_x_displacement:
            tot_r-=50.0; done=True; info["termination_reason"]="too_much_backward" # Increased penalty
        
        pdi=20; _=g_dynamic_render_active and (pdi:=5)
        # if self.current_episode_steps%pdi==0 or done or self.current_episode_steps==1:
        #      print(
        #         f"  DBG S{self.current_episode_steps}: TotR={tot_r:.2f} || "
        #         f"FwdVel={reward_forward_velocity:.2f} ({fvx:.3f}) "
        #         f"Prog={reward_net_cumulative_progress:.2f} "
        #         f"BwdP={penalty_backward_velocity:.2f} "
        #         f"YStab={penalty_y_position_stability:.2f} (Y:{self.data.qpos[1]:.2f}) "
        #         f"ZStab={penalty_z_position_stability:.2f} (Z:{self.data.qpos[2]:.2f} "
        #         f"SetTrgZ:{self.current_initial_body_z_pos:.2f} "
        #         f"IdealZ:{self.initial_body_z_pos_on_flat:.2f}) "
        #         f"OrientP={penalty_orientation_error:.2f} "
        #         f"SmoothP={penalty_action_smoothness:.2f} || "
        #         f"Term: {info['termination_reason'] if done else 'No'}"
        #     )
        self.last_commanded_clipped_sim_rad=f_cmds.copy(); return obs,tot_r,done,info

    def launch_viewer_internal(self): # Standard viewer launch
        if self.viewer is None:
            try: self.viewer=mujoco.viewer.launch_passive(self.model,self.data)
            except Exception as e: self.viewer=None
        elif self.viewer and not self.viewer.is_running(): self.viewer=None; self.launch_viewer_internal()
    def close_viewer_internal(self): # Standard viewer close
        if self.viewer and self.viewer.is_running():
            try: self.viewer.close()
            except Exception: pass
            finally: self.viewer=None
    def sync_viewer_if_active(self): # Standard viewer sync
        global g_dynamic_render_active,g_render_lock
        with g_render_lock:
            if g_dynamic_render_active:
                if self.viewer is None or not self.viewer.is_running(): self.launch_viewer_internal()
                if self.viewer and self.viewer.is_running():
                    try: self.viewer.sync()
                    except Exception: self.close_viewer_internal()
            elif self.viewer and self.viewer.is_running(): self.close_viewer_internal()

def toggle_render(): # Standard render toggle
    global g_dynamic_render_active,g_render_lock
    with g_render_lock: g_dynamic_render_active=not g_dynamic_render_active
    print(f"\nDynamic rendering toggled {'ON' if g_dynamic_render_active else 'OFF'}.")

def train(): # Main training loop
    print("Starting RL Training (Terrain)...")
    global sim_keyframe_home_qpos_map,ACTUATOR_TO_JOINT_NAME_MAP,JOINT_NAME_TO_QPOS_IDX_MAP
    os.makedirs(JSON_OUTPUT_DIR,exist_ok=True); os.makedirs(PTH_OUTPUT_DIR,exist_ok=True)
    print(f"JSON outputs: {JSON_OUTPUT_DIR}, PTH outputs: {PTH_OUTPUT_DIR}\n--- Verifying XML & Mappings ---")
    try: s_dir=os.path.dirname(os.path.abspath(__file__))
    except NameError: s_dir=os.getcwd()
    bpfm=os.path.dirname(s_dir); mda=os.path.join(bpfm,'our_robot'); xml_p=os.path.join(mda,XML_FILE_NAME)
    if not os.path.exists(xml_p): raise FileNotFoundError(f"CRITICAL: XML '{XML_FILE_NAME}' not found at '{xml_p}'.")
    print(f"Loading model from: {xml_p}")
    _vm=mujoco.MjModel.from_xml_path(xml_p); _vd=mujoco.MjData(_vm)
    for an in ACTUATOR_NAMES_ORDERED:
        aid=mujoco.mj_name2id(_vm,mujoco.mjtObj.mjOBJ_ACTUATOR,an); assert aid!=-1
        jid=_vm.actuator_trnid[aid,0]; jn=mujoco.mj_id2name(_vm,mujoco.mjtObj.mjOBJ_JOINT,jid); assert jn
        ACTUATOR_TO_JOINT_NAME_MAP[an]=jn; JOINT_NAME_TO_QPOS_IDX_MAP[jn]=_vm.jnt_qposadr[jid]
    kid=mujoco.mj_name2id(_vm,mujoco.mjtObj.mjOBJ_KEY,'home'); assert kid!=-1
    mujoco.mj_resetDataKeyframe(_vm,_vd,kid)
    print("Home keyframe qpos (sim_keyframe_home_qpos_map - from flat ground):")
    for an in ACTUATOR_NAMES_ORDERED:
        jn=ACTUATOR_TO_JOINT_NAME_MAP[an]; qix=JOINT_NAME_TO_QPOS_IDX_MAP[jn]
        sim_keyframe_home_qpos_map[jn]=_vd.qpos[qix]
        print(f"  {jn:<15} (Act: {an:<18}): {sim_keyframe_home_qpos_map[jn]:.4f} rad (Real Home: {real_robot_home_deg_map[an]:.1f} deg)")
    del _vm,_vd; print("--- Verification & Global Map Init Complete ---")
    env=QuadrupedEnv(xml_p); 
    if KEYBOARD_LIB_AVAILABLE:
        try: keyboard.add_hotkey('ctrl+l',toggle_render); print("INFO: Ctrl+L hotkey active.")
        except Exception as e: print(f"WARNING: Failed to set Ctrl+L hotkey: {e}.")
    sd=env.state_dim; ad=env.action_dim; dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:{dev}. State_dim:{sd}, Action_dim:{ad}")
    clr=INITIAL_LEARNING_RATE; cec=INITIAL_ENTROPY_COEF
    agent=ActorCritic(sd,ad,INITIAL_ACTION_STD_INIT).to(dev); opt=optim.Adam(agent.parameters(),lr=clr)
    s_mem,a_mem,lp_mem,r_mem,v_mem,m_mem=[],[],[],[],[],[]
    ep_r=deque(maxlen=100); avgr_hist=deque(maxlen=AVGR_HISTORY_LEN); tot_steps=0
    try:
        for ep in range(1,NUM_EPISODES+1):
            st=env.reset(); cep_r=0; term_rsn="max_steps"
            for step_n in range(MAX_STEPS_PER_EPISODE):
                tot_steps+=1; st_t=torch.FloatTensor(st).unsqueeze(0).to(dev)
                with torch.no_grad():
                    a_dist,v_t=agent(st_t); a_t=a_dist.sample(); lp_t=a_dist.log_prob(a_t).sum(axis=-1)
                a_np=a_t.squeeze(0).cpu().numpy(); nst,rew,done,info=env.step(a_np)
                s_mem.append(st);a_mem.append(a_np);lp_mem.append(lp_t.item());r_mem.append(rew);v_mem.append(v_t.item());m_mem.append(1-done)
                st=nst; cep_r+=rew
                if done: term_rsn=info.get("termination_reason","unknown"); break
            ep_r.append(cep_r); cavg_r=np.mean(ep_r) if ep_r else 0.0; avgr_hist.append(cavg_r)
            print(f"Ep:{ep}, Steps:{step_n+1}, R:{cep_r:.2f}, AvgR:{cavg_r:.2f} (Term:{term_rsn}), LR:{clr:.1e}, EntC:{cec:.4f}, ActStd:{torch.exp(agent.action_log_std.mean()).item():.3f}")
            if len(s_mem)>=POLICY_UPDATE_INTERVAL:
                rets,advs=[],[]; l_val=0.0
                if not done:
                    with torch.no_grad(): _,l_val_t=agent(torch.FloatTensor(nst).unsqueeze(0).to(dev)); l_val=l_val_t.item()
                gae=0.0
                for i in reversed(range(len(r_mem))):
                    nv=v_mem[i+1] if i+1<len(v_mem) else l_val; bv=nv*m_mem[i]
                    delta=r_mem[i]+GAMMA*bv-v_mem[i]; gae=delta+GAMMA*GAE_LAMBDA*m_mem[i]*gae
                    rets.insert(0,gae+v_mem[i]); advs.insert(0,gae)
                s_t=torch.FloatTensor(np.array(s_mem)).to(dev); a_t=torch.FloatTensor(np.array(a_mem)).to(dev)
                r_t=torch.FloatTensor(rets).to(dev).unsqueeze(1); adv_t=torch.FloatTensor(advs).to(dev)
                adv_t=(adv_t-adv_t.mean())/(adv_t.std()+1e-8)
                for _ in range(NUM_EPOCHS_PER_UPDATE):
                    ad_n,v_n=agent(s_t); lpn=ad_n.log_prob(a_t).sum(axis=-1); ent=ad_n.entropy().mean()
                    al=-(lpn*adv_t).mean(); cl=nn.MSELoss()(v_n,r_t); loss=al+VALUE_LOSS_COEF*cl-cec*ent
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(agent.parameters(),MAX_GRAD_NORM);opt.step()
                s_mem,a_mem,lp_mem,r_mem,v_mem,m_mem=[],[],[],[],[],[]
            if ep%ADAPTATION_CHECK_INTERVAL==0 and len(avgr_hist)>=AVGR_HISTORY_LEN:
                fh_avg=np.mean(list(avgr_hist)[:AVGR_HISTORY_LEN//2]); sh_avg=np.mean(list(avgr_hist)[AVGR_HISTORY_LEN//2:])
                tr_d=sh_avg-fh_avg; sig_ct=0.10*abs(cavg_r) if abs(cavg_r)>10 else 1.0
                print(f"  Adapting Hypers: TrendDiff={tr_d:.2f}, CurrentAvgR={cavg_r:.2f}, Thresh={sig_ct:.2f}")
                if tr_d<-sig_ct:
                    print("    Trend: WORSENING...")
                    clr=max(MIN_LR,clr*0.75)
                    cec=max(MIN_ENTROPY_COEF,cec*0.9)
                    with torch.no_grad():
                        agent.action_log_std.data=torch.clamp(agent.action_log_std.data-math.log(1.05),MIN_ACTION_LOG_STD,MAX_ACTION_LOG_STD)
                elif abs(tr_d)<sig_ct*0.3:
                    print("    Trend: STAGNANT...")
                    cec=min(MAX_ENTROPY_COEF,cec*1.05)
                    with torch.no_grad():
                        agent.action_log_std.data=torch.clamp(agent.action_log_std.data+math.log(1.03),MIN_ACTION_LOG_STD,MAX_ACTION_LOG_STD)
                    if isinstance(clr, float) and clr < MAX_LR * 0.1: clr = min(MAX_LR, clr * 1.05)
                elif tr_d>sig_ct:
                    print("    Trend: IMPROVING...")
                    if isinstance(clr, float) and clr > MIN_LR * 5: clr = max(MIN_LR, clr * 0.95)
                if not isinstance(clr, float) or not isinstance(cec, float): # safety check if clr/cec became boolean
                    print(f"Warning: LR or Entropy Coef became non-float. LR:{clr}, EntC:{cec}. Resetting to defaults.")
                    clr = INITIAL_LEARNING_RATE; cec = INITIAL_ENTROPY_COEF
                for pg in opt.param_groups: pg['lr']=clr
                print(f"    New LR:{clr:.1e}, New EntC:{cec:.4f}, New ActStd:{torch.exp(agent.action_log_std.mean()).item():.3f}")
            if ep%PTH_SAVE_INTERVAL==0: pth_p=PTH_SAVE_TEMPLATE.format(ep); torch.save(agent.state_dict(),pth_p); print(f"Model saved to {pth_p}"); json_p=JSON_OUTPUT_PATH_TEMPLATE.format(ep); generate_walk_json(agent,env,dev,json_p,JSON_MAX_STEPS_EPISODIC)
    except KeyboardInterrupt: print("Training interrupted by user (Ctrl+C).")
    finally:
        print("Training finished or interrupted.")
        if KEYBOARD_LIB_AVAILABLE:
            try: keyboard.remove_hotkey('ctrl+l'); print("INFO: Ctrl+L hotkey removed.")
            except Exception as e: print(f"WARNING: Could not remove hotkey: {e}")
        torch.save(agent.state_dict(),FINAL_PTH_SAVE_NAME); print(f"Final model saved as '{FINAL_PTH_SAVE_NAME}'")
        final_json_p=JSON_OUTPUT_PATH_TEMPLATE.format('final'); generate_walk_json(agent,env,dev,final_json_p,JSON_MAX_STEPS_FINAL)
        if env.viewer and env.viewer.is_running(): env.close_viewer_internal()

def generate_walk_json(agent,env,dev,json_path,num_steps): # Standard JSON generation
    print(f"\nGenerating walk (Terrain, {num_steps} steps) to {json_path}...")
    global g_dynamic_render_active
    if g_dynamic_render_active: print("INFO: Dynamic rendering ON.")
    else: print("INFO: Dynamic rendering OFF.")
    if env.hfield_id!=-1: env.model.hfield_data[:]=0.5; mujoco.mj_forward(env.model,env.data) # Flat for JSON
    real_robot_seq=[]; st=env.reset() # Resets to flat terrain
    if env.obstacle_geom_id!=-1: env.model.geom_rgba[env.obstacle_geom_id,3]=0.0; env.model.geom_pos[env.obstacle_geom_id,2]=-100.0; mujoco.mj_forward(env.model,env.data)
    for step_c in range(num_steps):
        st_t=torch.FloatTensor(st).unsqueeze(0).to(dev)
        with torch.no_grad(): ad,_=agent(st_t); am=ad.mean; anp=am.squeeze(0).cpu().numpy()
        nst,_,done,info=env.step(anp); s_cmds=info.get("sim_target_rad")
        if s_cmds is None: print(f"WARN: no sim_target_rad (step {step_c})."); continue
        td_dict={}
        for ano in ACTUATOR_NAMES_ORDERED:
            mid=actuator_ctrl_props[ano]['mujoco_id']; rdv=convert_sim_rad_to_real_deg(ano,s_cmds[mid],sim_keyframe_home_qpos_map,real_robot_home_deg_map,joint_scale_factors)
            td_dict[ano]=round(rdv,2) if rdv is not None else 0.0
        real_robot_seq.append({"duration":round(POLICY_DECISION_DT,3),"targets_deg":td_dict}); st=nst
        if done: print(f"Episode ended early during JSON gen (step {step_c+1}, Term: {info.get('termination_reason','unknown')})."); break
    if not real_robot_seq: print(f"WARN: No steps for JSON {json_path}."); return
    try:
        os.makedirs(os.path.dirname(json_path),exist_ok=True)
        with open(json_path,'w') as f_json: json.dump(real_robot_seq,f_json,indent=2) # Used f_json
        print(f"Saved {len(real_robot_seq)} steps to {json_path}")
    except IOError as e: print(f"ERROR writing JSON to {json_path}: {e}")
    env.reset()

if __name__ == "__main__":
    train()