# --- Path Setup to Find the 'quadpilot' Library ---
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
code_root_dir = os.path.dirname(os.path.dirname(script_dir))
if code_root_dir not in sys.path:
    print(f"Adding '{code_root_dir}' to system path to find 'quadpilot'")
    sys.path.insert(0, code_root_dir)

# --- USER-REQUESTED IMPORT BLOCK (PRESERVED EXACTLY) ---
import time
import math
import threading
import numpy as np
from argparse import ArgumentParser
from stable_baselines3 import PPO
from environments.ScaleActionEnvironment import ScaleActionWrapper
from environments.WalkEnvironment import WalkEnvironmentV0
from environments.JumpEnvironment import JumpEnvironmentV0
from .RealTimePlotter import RealTimePlotter
from .ScaleActions import ScaleActions

# --- ADDITIONAL IMPORTS FOR REAL ROBOT CONTROL ---
try:
    from pynput import keyboard
except ImportError:
    print("FATAL ERROR: pynput library not found. Please run 'pip install pynput'")
    sys.exit(1)
try:
    from quadpilot import QuadPilotBody
except ImportError as e:
    print(f"FATAL ERROR: Could not import QuadPilotBody from '{code_root_dir}'. Details: {e}")
    sys.exit(1)

# --- Configuration ---
MODEL_PATH = r"C:\Users\rncb0\Code\cuadruped\Code\mujoco\best_model\best_model_final_walk.zip"
NN_ACTUATOR_ORDER = [ "FR_tigh_actuator", "FR_knee_actuator", "FL_tigh_actuator", "FL_knee_actuator", "BR_tigh_actuator", "BR_knee_actuator", "BL_tigh_actuator", "BL_knee_actuator" ]
ACTION_DIM = len(NN_ACTUATOR_ORDER)
EXPECTED_OBS_DIM = 33
CONTROL_LOOP_FREQUENCY = 10.0
CONTROL_LOOP_DT = 1.0 / CONTROL_LOOP_FREQUENCY
NUM_MOTORS = 8
IMU_ESP_TARGET_IP = '192.168.137.101'
ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP = { "FL_tigh_actuator": 3, "FL_knee_actuator": 0, "FR_tigh_actuator": 1, "FR_knee_actuator": 2, "BR_tigh_actuator": 5, "BR_knee_actuator": 4, "BL_tigh_actuator": 7, "BL_knee_actuator": 6 }
MOTOR_PINS_CONFIG = [ (39, 40, 41, 42), (16, 15, 7, 6), (17, 18, 5, 4), (37, 38, 1, 2), (37, 38, 1, 2), (40, 39, 42, 41), (15, 16, 6, 7), (18, 17, 4, 5) ]
ROBOT_JOINT_LIMITS_DEG_BY_MOTOR_IDX = [ (-70.0, 70.0), (-70.0, 70.0), (-70.0, 70.0), (-70.0, 70.0), (-70.0, 70.0), (-70.0, 70.0), (-70.0, 70.0), (-70.0, 70.0) ]

def get_robot_home_angles_deg_by_motor_idx():
    home_deg = [0.0] * NUM_MOTORS
    home_deg[ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP["FL_tigh_actuator"]] = 45.0; home_deg[ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP["FL_knee_actuator"]] = 45.0
    home_deg[ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP["FR_tigh_actuator"]] = -45.0; home_deg[ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP["FR_knee_actuator"]] = 45.0
    home_deg[ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP["BR_tigh_actuator"]] = 45.0; home_deg[ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP["BR_knee_actuator"]] = -45.0
    home_deg[ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP["BL_tigh_actuator"]] = 45.0; home_deg[ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP["BL_knee_actuator"]] = -45.0
    return home_deg

ROBOT_HOME_Q_DEG_BY_MOTOR_IDX = get_robot_home_angles_deg_by_motor_idx()
ROBOT_HOME_Q_RAD_BY_NN_ORDER = np.zeros(ACTION_DIM)
for i, act_name in enumerate(NN_ACTUATOR_ORDER):
    motor_idx = ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP[act_name]
    ROBOT_HOME_Q_RAD_BY_NN_ORDER[i] = np.radians(ROBOT_HOME_Q_DEG_BY_MOTOR_IDX[motor_idx])

g_sb3_model, g_robot_commander, g_robot_monitor = None, None, None
g_imu_esp_idx_on_monitor, g_control_thread = -1, None
g_shutdown_initiated = False
g_control_loop_active = threading.Event()
g_listener_stop_flag = threading.Event()
g_current_x_velocity_estimate_mps = 0.0

def get_imu_esp_index(monitor_instance, target_ip):
    if not monitor_instance or not monitor_instance.ips: return -1
    try: return monitor_instance.ips.index(target_ip)
    except ValueError: return -1

def get_robot_state_for_policy():
    for _ in range(10): # Try for up to 100ms
        if g_imu_esp_idx_on_monitor == -1: return None, False
        dmp_data = g_robot_monitor.get_latest_dmp_data_for_esp(g_imu_esp_idx_on_monitor)
        all_motor_angles_deg = [0.0] * NUM_MOTORS
        data_valid = True
        for esp_idx in range(len(g_robot_monitor.ips)):
            esp_motor_data = g_robot_monitor.get_latest_motor_data_for_esp(esp_idx)
            if not esp_motor_data or 'angles' not in esp_motor_data:
                data_valid = False; break
            for i in range(len(esp_motor_data['angles'])):
                all_motor_angles_deg[esp_idx * 4 + i] = esp_motor_data['angles'][i]
        
        if data_valid and dmp_data and 'ypr_deg' in dmp_data and 'world_accel_mps2' in dmp_data:
            ypr_deg = dmp_data['ypr_deg']
            orientation = np.array([math.radians(ypr_deg.get(k, 0.0)) for k in ['yaw', 'pitch', 'roll']])
            world_ax = dmp_data['world_accel_mps2'].get('ax', 0.0) 
            current_q_rad = np.zeros(ACTION_DIM)
            for i, act_name in enumerate(NN_ACTUATOR_ORDER):
                motor_idx = ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP[act_name]
                current_q_rad[i] = np.radians(all_motor_angles_deg[motor_idx])
            global g_current_x_velocity_estimate_mps
            g_current_x_velocity_estimate_mps += world_ax * CONTROL_LOOP_DT
            g_current_x_velocity_estimate_mps *= 100
            joint_deltas = current_q_rad - ROBOT_HOME_Q_RAD_BY_NN_ORDER
            real_obs = np.concatenate([orientation, joint_deltas, [g_current_x_velocity_estimate_mps]]).astype(np.float32)
            full_obs = np.zeros(EXPECTED_OBS_DIM, dtype=np.float32)
            full_obs[:len(real_obs)] = real_obs
            return full_obs, True
        time.sleep(0.01)
    return None, False

def run_robot_control_loop():
    print(f"\n--- Starting Continuous NN Control Loop. Press 'S' to stop ---")
    scaler = ScaleActions()
    loop_count = 0
    while g_control_loop_active.is_set():
        step_start_time = time.time()
        current_obs, is_valid = get_robot_state_for_policy()
        if not is_valid:
            print("Warning: Timed out waiting for sensor data. Skipping step.")
            time.sleep(CONTROL_LOOP_DT)
            continue

        action_batch, _ = g_sb3_model.predict(np.expand_dims(current_obs, axis=0), deterministic=False)
        action = action_batch[0]

        # The NN outputs an *offset* from the home pose.
        nn_offset_angles = scaler.to_real_robot_degrees(action)
        
        robot_command = [0.0] * NUM_MOTORS
        for i, act_name in enumerate(NN_ACTUATOR_ORDER):
            motor_idx = ROBOT_ACTUATOR_TO_MOTOR_IDX_MAP[act_name]

            # === THE CRITICAL FIX IS HERE ===
            # We add the NN's desired offset to the motor's home angle
            # to get the final absolute target angle for the real robot.
            home_angle = ROBOT_HOME_Q_DEG_BY_MOTOR_IDX[motor_idx]
            nn_offset = nn_offset_angles[i]
            absolute_target_angle =  nn_offset + home_angle
            # ===============================

            min_lim, max_lim = ROBOT_JOINT_LIMITS_DEG_BY_MOTOR_IDX[motor_idx]
            robot_command[motor_idx] = np.clip(absolute_target_angle, min_lim, max_lim)

        print(f"Step {loop_count}: Sending angles {np.round(robot_command, 1)}")
        g_robot_commander.set_angles(robot_command)
        
        elapsed_this_step = time.time() - step_start_time
        sleep_duration = CONTROL_LOOP_DT - elapsed_this_step
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        
        loop_count += 1
    print("--- Continuous NN Control Loop Stopped ---")

def initialize_all():
    global g_robot_commander, g_robot_monitor, g_imu_esp_idx_on_monitor
    print("Initializing QuadPilotBody instances...")
    g_robot_commander = QuadPilotBody(listen_for_broadcasts=False)
    g_robot_monitor = QuadPilotBody(listen_for_broadcasts=True)
    time.sleep(2.0)
    if not g_robot_commander.ips or not g_robot_monitor.ips:
        print("FATAL: No ESP32 boards detected."); return False
    print(f"Commander IPs: {g_robot_commander.ips}, Monitor IPs: {g_robot_monitor.ips}")
    g_imu_esp_idx_on_monitor = get_imu_esp_index(g_robot_monitor, IMU_ESP_TARGET_IP)
    if g_imu_esp_idx_on_monitor == -1:
        print(f"FATAL: IMU ESP ({IMU_ESP_TARGET_IP}) not found."); return False
    print(f"IMU found at ESP index {g_imu_esp_idx_on_monitor}.")
    try:
        print("Setting up motors...")
        g_robot_commander.set_control_params(P=0.9, I=0.0, D=0.9, dead_zone=5, pos_thresh=5)
        g_robot_commander.set_all_pins(MOTOR_PINS_CONFIG); time.sleep(0.5)
        g_robot_commander.reset_all(); time.sleep(0.2)
        g_robot_commander.set_all_control_status(True)
        print("Motor setup complete.")
    except Exception as e:
        print(f"FATAL: Motor setup failed: {e}"); return False
    print("Waiting for initial sensor data...")
    _, is_valid = get_robot_state_for_policy()
    if is_valid:
        print("Initial sensor data received. System is ready."); return True
    else:
        print("FATAL: Timed out waiting for initial sensor data."); return False

def set_robot_to_home_pose():
    if g_control_loop_active.is_set():
        print("Cannot go to home pose while loop is active. Press 'S' first.")
        return
    if g_robot_commander:
        print("\nSetting robot to home pose..."); g_robot_commander.set_angles(ROBOT_HOME_Q_DEG_BY_MOTOR_IDX)

def safe_shutdown_robot():
    global g_shutdown_initiated
    if g_shutdown_initiated: return; g_shutdown_initiated = True
    print("\nInitiating safe shutdown...")
    g_control_loop_active.clear()
    if g_control_thread and g_control_thread.is_alive():
        g_control_thread.join(timeout=1.0)
    if g_robot_commander:
        print("Disabling motors...")
        try:
            g_robot_commander.set_angles([0] * NUM_MOTORS); time.sleep(1.0)
            g_robot_commander.set_all_control_status(False)
        finally:
            g_robot_commander.close()
    if g_robot_monitor: g_robot_monitor.close()
    print("Robot shutdown complete.")

def on_key_press(key):
    global g_control_loop_active, g_control_thread
    if g_shutdown_initiated: return False
    char = getattr(key, 'char', None)
    if char: char = char.lower()
    if key == keyboard.Key.esc:
        print("ESC pressed. Shutting down..."); g_listener_stop_flag.set(); return False
    if char == 'y':
        if not g_control_loop_active.is_set():
            g_control_loop_active.set()
            g_control_thread = threading.Thread(target=run_robot_control_loop, daemon=True)
            g_control_thread.start()
    elif char == 's':
        if g_control_loop_active.is_set():
            print("\n'S' pressed. Stopping loop..."); g_control_loop_active.clear()
    elif char == 'a':
        set_robot_to_home_pose()

if __name__ == "__main__":
    print("--- Quadruped Real-World Policy Control ---")
    if not initialize_all():
        safe_shutdown_robot(); sys.exit(1)
    g_sb3_model = PPO.load(MODEL_PATH)
    print("Model loaded successfully.")
    set_robot_to_home_pose(); time.sleep(1)
    print("\n" + "="*55 + "\nKeyboard Menu:\n  'Y': Start Loop\n  'S': Stop Loop\n  'A': Home Pose\n  'ESC': Exit\n" + "="*55 + "\n")
    listener = keyboard.Listener(on_press=on_key_press); listener.start()
    print("Keyboard listener active. Ready for commands.")
    try:
        g_listener_stop_flag.wait()
    except KeyboardInterrupt:
        print("\nCtrl+C detected.")
    finally:
        safe_shutdown_robot()
        print("Application finished.")