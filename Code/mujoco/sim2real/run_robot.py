import os
import sys
import time
import math
import threading
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal 
from pynput import keyboard # For keyboard input

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sim2real_dir_parent = os.path.dirname(script_dir) 
code_root_dir = os.path.dirname(sim2real_dir_parent) 
if code_root_dir not in sys.path:
    sys.path.insert(0, code_root_dir)

try:
    from quadpilot import QuadPilotBody
except ImportError as e:
    print(f"ERROR: Could not import QuadPilotBody. Ensure 'quadpilot' is in {code_root_dir}.")
    print(f"Details: {e}")
    sys.exit(1)

# --- Configuration: Model & Policy ---
MODEL_PTH_TO_LOAD = './best/quadruped_ac_terrain_ep6800.pth' 
NN_ACTUATOR_ORDER = [
    "FR_tigh_actuator", "FR_knee_actuator", "FL_tigh_actuator", "FL_knee_actuator",
    "BR_tigh_actuator", "BR_knee_actuator", "BL_tigh_actuator", "BL_knee_actuator",
]
ACTION_DIM = len(NN_ACTUATOR_ORDER)
STATE_DIM = 3 + ACTION_DIM + 1 # 3 ori, N joints, 1 velocity (now X-vel estimate)
INITIAL_ACTION_STD_INIT = 0.3 
ACTION_AMPLITUDE_DEG = 50.0
ACTION_AMPLITUDE_RAD = math.radians(ACTION_AMPLITUDE_DEG)
CONTROL_LOOP_FREQUENCY = 12.5 
CONTROL_LOOP_DT = 1.0 / CONTROL_LOOP_FREQUENCY

# --- Configuration: Robot Specific ---
NUM_MOTORS = 8
IMU_ESP_TARGET_IP = '192.168.137.101' 
ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP = {
    "FL_tigh_actuator": 3, "FL_knee_actuator": 0, "FR_tigh_actuator": 1, "FR_knee_actuator": 2,
    "BR_tigh_actuator": 5, "BR_knee_actuator": 4, "BL_tigh_actuator": 7, "BL_knee_actuator": 6,
}
MOTOR_PINS_CONFIG = [
    (39, 40, 41, 42), (16, 15, 7, 6), (17, 18, 5, 4), (37, 38, 1, 2),
    (37, 38, 1, 2), (40, 39, 42, 41), (15, 16, 6, 7), (18, 17, 4, 5),
]

def get_robot_home_angles_deg_by_motor_idx():
    home_deg = [0.0] * NUM_MOTORS
    home_deg[ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP["FL_tigh_actuator"]] = 45.0
    home_deg[ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP["FL_knee_actuator"]] = 45.0
    home_deg[ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP["FR_tigh_actuator"]] = -45.0
    home_deg[ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP["FR_knee_actuator"]] = 45.0
    home_deg[ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP["BR_tigh_actuator"]] = 45.0
    home_deg[ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP["BR_knee_actuator"]] = -45.0
    home_deg[ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP["BL_tigh_actuator"]] = 45.0
    home_deg[ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP["BL_knee_actuator"]] = -45.0
    return home_deg

ROBOT_HOME_Q_DEG_BY_MOTOR_IDX = get_robot_home_angles_deg_by_motor_idx()
ROBOT_HOME_Q_RAD_BY_NN_ORDER = np.zeros(ACTION_DIM)
for i, act_name in enumerate(NN_ACTUATOR_ORDER):
    motor_idx = ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP[act_name]
    ROBOT_HOME_Q_RAD_BY_NN_ORDER[i] = np.radians(ROBOT_HOME_Q_DEG_BY_MOTOR_IDX[motor_idx])

ROBOT_JOINT_LIMITS_DEG_BY_MOTOR_IDX = [
    (-85.0, 85.0), (-85.0, 85.0), (-85.0, 85.0), (-85.0, 85.0),
    (-85.0, 85.0), (-85.0, 85.0), (-85.0, 85.0), (-85.0, 85.0),
] 

# --- Global State Variables ---
g_nn_model = None
g_torch_device = None
g_robot_commander = None
g_robot_monitor = None
g_imu_esp_idx_on_monitor = -1
g_nn_control_loop_active = False
g_nn_control_thread = None
g_shutdown_initiated = False
g_listener_stop_flag = threading.Event()
g_current_x_velocity_estimate_mps = 0.0 # For X-velocity integration


# --- Neural Network Model Definition & Functions ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 1024), nn.Tanh(), nn.Linear(1024, 512), nn.Tanh(),
            nn.Linear(512, action_dim), nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 1024), nn.Tanh(), nn.Linear(1024, 512), nn.Tanh(),
            nn.Linear(512, 1)
        )
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * math.log(action_std_init))
    def forward(self, state):
        action_mean = self.actor(state)
        action_std = torch.exp(self.action_log_std.expand_as(action_mean))
        return Normal(action_mean, action_std), self.critic(state)

def load_neural_network_model(model_path, state_dim, action_dim, action_std_init_val):
    global g_nn_model, g_torch_device
    g_torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using PyTorch device: {g_torch_device}")
    g_nn_model = ActorCritic(state_dim, action_dim, action_std_init_val).to(g_torch_device)
    if not os.path.exists(model_path): return False
    try:
        g_nn_model.load_state_dict(torch.load(model_path, map_location=g_torch_device, weights_only=True))
        g_nn_model.eval(); print(f"NN model '{model_path}' loaded."); return True
    except Exception as e: print(f"ERROR loading NN model: {e}"); return False

def predict_nn_action(current_state_np_array):
    if g_nn_model is None: return None
    state_tensor = torch.FloatTensor(current_state_np_array.astype(np.float32)).unsqueeze(0).to(g_torch_device)
    with torch.no_grad():
        action_distribution, _ = g_nn_model(state_tensor)
        return action_distribution.mean.squeeze(0).cpu().numpy()

# --- Robot Sensor Data Acquisition ---
def get_imu_esp_index(monitor_instance, target_ip):
    if not monitor_instance or not monitor_instance.ips: return -1
    try: return monitor_instance.ips.index(target_ip)
    except ValueError: return -1

def get_robot_orientation_ypr_rad_and_world_ax(monitor_instance, imu_esp_idx):
    """
    Returns (orientation_ypr_rad, world_ax_mps2) or (None, None) if data is unavailable.
    world_ax_mps2 is the linear X acceleration in the world frame (gravity compensated).
    """
    if imu_esp_idx == -1: return None, None
    if not monitor_instance or not monitor_instance.is_data_available_from_esp(imu_esp_idx) or \
       not monitor_instance.is_dmp_ready_for_esp(imu_esp_idx):
        return None, None
    
    dmp_data = monitor_instance.get_latest_dmp_data_for_esp(imu_esp_idx)
    if dmp_data:
        ypr_rad = None; world_ax = None
        if 'ypr_deg' in dmp_data:
            ypr_deg_data = dmp_data['ypr_deg']
            ypr_rad = np.array([math.radians(ypr_deg_data.get(k,0.0)) for k in ['yaw','pitch','roll']])
        if 'world_accel_mps2' in dmp_data: # This dict should contain 'ax', 'ay', 'az'
            world_ax = dmp_data['world_accel_mps2'].get('ax', 0.0) 
        if ypr_rad is not None and world_ax is not None: return ypr_rad, world_ax
    return None, None


def get_robot_joint_angles_deg_by_motor_idx(monitor_instance):
    all_motor_angles_deg = [0.0] * NUM_MOTORS; data_valid_overall = True
    if not monitor_instance or not monitor_instance.ips: return None
    for esp_idx in range(len(monitor_instance.ips)):
        if not monitor_instance.is_data_available_from_esp(esp_idx): data_valid_overall = False; continue
        esp_motor_data = monitor_instance.get_latest_motor_data_for_esp(esp_idx)
        if not esp_motor_data or 'angles' not in esp_motor_data: data_valid_overall = False; continue
        current_esp_angles = esp_motor_data['angles']; start_global_motor_idx = esp_idx * 4 
        for local_motor_idx in range(len(current_esp_angles)):
            global_motor_idx = start_global_motor_idx + local_motor_idx
            if global_motor_idx < NUM_MOTORS: all_motor_angles_deg[global_motor_idx] = current_esp_angles[local_motor_idx]
            else: data_valid_overall = False
    return np.array(all_motor_angles_deg) if data_valid_overall else None

def update_and_get_body_x_velocity_mps(world_ax_mps2, dt):
    global g_current_x_velocity_estimate_mps
    if world_ax_mps2 is not None:
        g_current_x_velocity_estimate_mps += world_ax_mps2 * dt
    # HIGHLY EXPERIMENTAL: Simple damping to prevent runaway integration, may need tuning or removal
    g_current_x_velocity_estimate_mps *= 0.99 
    return g_current_x_velocity_estimate_mps


# --- Core Control Logic & Robot Actions ---
def perform_single_nn_inference_step():
    global g_robot_commander, g_robot_monitor, g_imu_esp_idx_on_monitor
    
    if not g_robot_commander or not g_robot_monitor: return False

    orientation_ypr_rad, world_ax_mps2 = get_robot_orientation_ypr_rad_and_world_ax(g_robot_monitor, g_imu_esp_idx_on_monitor)
    current_q_deg_by_motor_idx = get_robot_joint_angles_deg_by_motor_idx(g_robot_monitor)

    if orientation_ypr_rad is None or current_q_deg_by_motor_idx is None or world_ax_mps2 is None:
        print("Sensor data missing for NN step (IMU or Joints)."); return False

    body_x_vel_mps = update_and_get_body_x_velocity_mps(world_ax_mps2, CONTROL_LOOP_DT) 

    current_q_rad_nn_order = np.zeros(ACTION_DIM)
    for i, act_name in enumerate(NN_ACTUATOR_ORDER):
        motor_idx = ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP[act_name]
        current_q_rad_nn_order[i] = np.radians(current_q_deg_by_motor_idx[motor_idx])
    
    joint_deltas_rad_nn_order = current_q_rad_nn_order - ROBOT_HOME_Q_RAD_BY_NN_ORDER
    
    nn_input_state = np.concatenate([orientation_ypr_rad, joint_deltas_rad_nn_order, [body_x_vel_mps]]).astype(np.float32)

    # --- Logging Input to NN ---
    print(f"\n--- NN Input State @ {time.strftime('%H:%M:%S')} ---")
    print(f"  Orientation (YPR_rad): Yaw={nn_input_state[0]:.3f}, Pitch={nn_input_state[1]:.3f}, Roll={nn_input_state[2]:.3f}")
    print(f"  IMU World Ax (m/s^2): {world_ax_mps2:.3f}")
    print( "  Joint Deltas from Home (rad):")
    for i in range(ACTION_DIM):
        print(f"    {NN_ACTUATOR_ORDER[i]:<18}: {nn_input_state[3+i]:.3f}")
    print(f"  Estimated Body X-Velocity (m/s): {nn_input_state[-1]:.3f}")
    # --- End Logging Input ---

    normalized_actions_nn_order = predict_nn_action(nn_input_state)
    if normalized_actions_nn_order is None: print("NN prediction failed."); return False
    
    # --- Logging NN Output Actions ---
    print( "--- NN Raw Output (Normalized Actions) ---")
    for i in range(ACTION_DIM):
        print(f"    {NN_ACTUATOR_ORDER[i]:<18}: {normalized_actions_nn_order[i]:.3f}")
    # --- End Logging NN Output ---

    action_deltas_rad_nn_order = normalized_actions_nn_order * ACTION_AMPLITUDE_RAD
    # target_q_rad_nn_order = action_deltas_rad_nn_order
    target_q_rad_nn_order = ROBOT_HOME_Q_RAD_BY_NN_ORDER + action_deltas_rad_nn_order
    
    target_q_deg_robot_order = [0.0] * NUM_MOTORS
    print( "--- Target Angles Sent to Robot (deg) ---")
    for i, act_name_nn_order in enumerate(NN_ACTUATOR_ORDER): # Iterate in NN's canonical order
        motor_idx = ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP[act_name_nn_order]
        
        target_angle_deg = np.degrees(target_q_rad_nn_order[i]) # This is absolute target for this actuator
        original_target_for_log = target_angle_deg 
        
        min_lim, max_lim = ROBOT_JOINT_LIMITS_DEG_BY_MOTOR_IDX[motor_idx]
        clipped_target_angle_deg = np.clip(target_angle_deg, min_lim, max_lim)
        
        target_q_deg_robot_order[motor_idx] = clipped_target_angle_deg
        
        # Log with physical motor index and NN actuator name
        print(f"    Motor {motor_idx} ({act_name_nn_order:<18}): {clipped_target_angle_deg:.2f} (Pre-clip: {original_target_for_log:.2f})")
    # --- End Logging Target Angles ---

    if not g_robot_commander.set_angles(target_q_deg_robot_order): print("Warning: Failed to send angles."); return False
    return True

def set_robot_to_defined_home_pose():
    global g_robot_commander, g_current_x_velocity_estimate_mps
    if g_nn_control_loop_active: print("NN loop active. Press 'S' to stop."); return
    if g_robot_commander:
        print("\nSetting robot to home pose...")
        g_current_x_velocity_estimate_mps = 0.0 # Reset X-vel estimate when going to home
        if g_robot_commander.set_angles(ROBOT_HOME_Q_DEG_BY_MOTOR_IDX):
            print(f"Home pose sent: {[f'{a:.1f}' for a in ROBOT_HOME_Q_DEG_BY_MOTOR_IDX]}")
        else: print("Failed to set home pose.")
    else: print("Robot commander not initialized.")

def run_continuous_nn_loop():
    global g_nn_control_loop_active; print("\n--- Starting Continuous NN Loop (Press 'S' to stop) ---")
    loop_count = 0; last_print_time = time.time()
    while g_nn_control_loop_active:
        step_start_time = time.time()
        if perform_single_nn_inference_step(): loop_count +=1
        else: print("Error in NN step. Pausing."); time.sleep(0.2)
        
        elapsed_this_step = time.time() - step_start_time
        sleep_duration = CONTROL_LOOP_DT - elapsed_this_step
        if sleep_duration > 0: time.sleep(sleep_duration)
        elif loop_count > 1 : print(f"Warning: Loop {loop_count} took {elapsed_this_step:.4f}s > {CONTROL_LOOP_DT:.4f}s.")
        
        if time.time() - last_print_time > 5.0 and g_nn_control_loop_active:
             print(f"Loop {loop_count} active. XVel_Est: {g_current_x_velocity_estimate_mps:.3f} m/s")
             last_print_time = time.time()
    print("--- Continuous NN Control Loop Stopped ---")

def safe_shutdown_robot():
    global g_robot_commander, g_robot_monitor, g_nn_control_loop_active, g_nn_control_thread, g_shutdown_initiated
    if g_shutdown_initiated: return; g_shutdown_initiated = True
    print("\nInitiating safe shutdown..."); g_nn_control_loop_active = False 
    if g_nn_control_thread and g_nn_control_thread.is_alive():
        g_nn_control_thread.join(timeout=2.0)
        if g_nn_control_thread.is_alive(): print("Warning: NN thread did not finish.")
    if g_robot_commander:
        print("Disabling & resetting motors...")
        try:
            g_robot_commander.set_all_control_status(False); time.sleep(0.1)
            g_robot_commander.reset_all(); time.sleep(0.1)
        except Exception as e_s: print(f"Error: {e_s}")
        finally: g_robot_commander.close(); g_robot_commander = None
    if g_robot_monitor: g_robot_monitor.close(); g_robot_monitor = None
    print("Robot shutdown complete.")

def initialize_all():
    global g_robot_commander, g_robot_monitor, g_imu_esp_idx_on_monitor, g_current_x_velocity_estimate_mps
    g_current_x_velocity_estimate_mps = 0.0 
    print("Initializing QuadPilotBody instances...");
    g_robot_commander = QuadPilotBody(listen_for_broadcasts=False)
    g_robot_monitor = QuadPilotBody(listen_for_broadcasts=True)
    time.sleep(1.5) 
    if not g_robot_commander.ips or not g_robot_monitor.ips: print("FATAL: No ESPs."); return False
    print(f"Cmd ESPs: {g_robot_commander.ips}, Mon ESPs: {g_robot_monitor.ips}")
    g_imu_esp_idx_on_monitor = get_imu_esp_index(g_robot_monitor, IMU_ESP_TARGET_IP)
    if g_imu_esp_idx_on_monitor == -1: print(f"FATAL: IMU ESP ({IMU_ESP_TARGET_IP}) not in {g_robot_monitor.ips}"); return False
    print(f"IMU: ESP index {g_imu_esp_idx_on_monitor} (IP: {IMU_ESP_TARGET_IP}).")
    try:
        print("Setting up motors..."); time.sleep(0.1)
        if not g_robot_commander.set_control_params(P=0.9,I=0.0,D=0.9,dead_zone=5,pos_thresh=5): raise Exception("Set CtrParams")
        if not g_robot_commander.set_all_pins(MOTOR_PINS_CONFIG): raise Exception("Set Pins")
        time.sleep(0.5);
        if not g_robot_commander.reset_all(): raise Exception("Reset All")
        time.sleep(0.2);
        if not g_robot_commander.set_all_control_status(True): raise Exception("Set CtrlStatus")
        print("Motor setup complete."); time.sleep(0.1)
    except Exception as e: print(f"Motor setup failed: {e}"); return False
    if not load_neural_network_model(MODEL_PTH_TO_LOAD, STATE_DIM, ACTION_DIM, INITIAL_ACTION_STD_INIT): return False
    print("\nWaiting for initial IMU data..."); wait_start = time.time()
    while time.time() - wait_start < 10:
        ori, ax = get_robot_orientation_ypr_rad_and_world_ax(g_robot_monitor, g_imu_esp_idx_on_monitor)
        if ori is not None and ax is not None: print("Target IMU ready (orientation and X-accel)."); return True
        time.sleep(0.5); print("Still waiting for IMU...")
    print("Warning: Timed out waiting for IMU data."); return True 

# --- Keyboard Listener ---
def on_key_press(key):
    global g_nn_control_loop_active, g_nn_control_thread, g_listener_stop_flag
    if g_shutdown_initiated: return False; char = None
    if hasattr(key, 'char'): char = key.char
    try:
        if key == keyboard.Key.esc:
            print("ESC pressed. Shutdown..."); safe_shutdown_robot(); g_listener_stop_flag.set(); return False 
        if char:
            char = char.lower()
            if   char == 'a': set_robot_to_defined_home_pose()
            elif char == 'd':
                if g_nn_control_loop_active: print("Loop active ('W'). Press 'S' first.")
                else: perform_single_nn_inference_step() # Logging is inside this function
            elif char == 'w':
                if g_nn_control_loop_active: print("Loop already active.")
                else:
                    g_nn_control_loop_active = True
                    g_nn_control_thread = threading.Thread(target=run_continuous_nn_loop, daemon=True); g_nn_control_thread.start()
            elif char == 's':
                if g_nn_control_loop_active:
                    print("\n'S' pressed. Stopping loop..."); g_nn_control_loop_active = False
                    if g_nn_control_thread and g_nn_control_thread.is_alive(): g_nn_control_thread.join(timeout=1.0) 
                    print("Loop should be stopped.")
                else: print("Loop not active.")
    except Exception as e: print(f"Error in on_key_press: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Quadruped NN Control: X-Vel Est. & Keyboard Menu ---")
    if not initialize_all(): print("Init failed. Exiting."); safe_shutdown_robot(); sys.exit(1)
    print("\nSetting robot to initial home pose..."); set_robot_to_defined_home_pose() 
    print("\nKeyboard Menu:"); print("  A: Home Pose (resets X-vel estimate)"); print("  D: Single NN Step (verbose logging)"); print("  W: Start Continuous Loop"); print("  S: Stop Loop"); print("  Esc: Shutdown & Exit"); print("-------------------------------------------")
    print(f"NOTE: X-vel est. by IMU X-accel integration WILL DRIFT."); print(f"      NN model used true X-vel. Using IMU-derived X-vel is EXPERIMENTAL.")
    listener = keyboard.Listener(on_press=on_key_press); listener.start()
    print("Keyboard listener active. Ready for commands.")
    try: g_listener_stop_flag.wait() 
    except KeyboardInterrupt: print("\nCtrl+C in main. Shutdown...")
    finally:
        print("Main loop ending. Final shutdown...");
        if listener.is_alive(): listener.stop()
        safe_shutdown_robot(); print("Application finished.")