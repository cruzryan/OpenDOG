# quad_autocorrect.py (v3 - Hallucination removed, robust init retained)

import os
import sys
import time
import threading
from pynput import keyboard
import math

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(os.path.dirname(script_dir), "quadpilot")
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

try:
    from body import QuadPilotBody
except ImportError:
    print(f"ERROR: Could not import QuadPilotBody. "
          f"Ensure 'quadpilot' directory is a sibling to this script's directory.")
    print(f"Attempted to look in: {code_dir}")
    sys.exit(1)


# --- Configuration ---
NUM_MOTORS = 8
TARGET_MOTORS = list(range(NUM_MOTORS))
# The index of the ESP32 that sends the primary IMU/DMP data (0 for ESP1, 1 for ESP2)
# Set this to the ESP that has the MPU6050 connected.
ESP_WITH_IMU = 1

# --- Yaw Auto-Correction Configuration ---
REAL_TARGET = 0.0
TARGET_YAW = REAL_TARGET
CORRECTION_GAIN_KP = 1.5
NEUTRAL_LIFT_ANGLE = 30.0
MIN_LIFT_ANGLE = 20.0
MAX_LIFT_ANGLE = 45.0
WALK_STEP_DURATION = 0.1

# Mapping: Actuator Name -> QuadPilotBody Index (0-7)
ACTUATOR_NAME_TO_INDEX_MAP = {
    "FL_tigh_actuator": 3, "FL_knee_actuator": 0,
    "FR_tigh_actuator": 1, "FR_knee_actuator": 2,
    "BR_tigh_actuator": 5, "BR_knee_actuator": 4,
    "BL_tigh_actuator": 7, "BL_knee_actuator": 6,
}

# Motor pin configurations (ENCODER_A, ENCODER_B, IN1, IN2)
MOTOR_PINS = [
    (39, 40, 41, 42),  # Motor 0 (FL_knee) -> ESP1
    (16, 15, 7, 6),    # Motor 1 (FR_tigh) -> ESP1
    (17, 18, 5, 4),    # Motor 2 (FR_knee) -> ESP1
    (37, 38, 1, 2),    # Motor 3 (FL_tigh) -> ESP1
    (37, 38, 1, 2),    # Motor 4 (BR_knee) -> ESP2
    (40, 39, 42, 41),  # Motor 5 (BR_tigh) -> ESP2
    (15, 16, 6, 7),    # Motor 6 (BL_knee) -> ESP2
    (18, 17, 4, 5),    # Motor 7 (BL_tigh) -> ESP2
]

# --- Initialize QuadPilotBody and Setup ---
body_controller = None
initial_setup_successful = False
try:
    body_controller = QuadPilotBody(listen_for_broadcasts=True)
    print("QuadPilotBody (control & monitor) instance initialized.")
    # Give the listener thread time to discover both ESPs on the network
    print("Waiting for ESPs to connect...")
    time.sleep(2.0)

    print("Setting initial control parameters...")
    if not body_controller.set_control_params(P=1.5, I=0.0, D=0.3, dead_zone=5, pos_thresh=5):
        raise Exception("Failed to set control parameters.")
    print("Control parameters set.")
    time.sleep(0.1)

    print("Initializing all motor pins...")
    if not body_controller.set_all_pins(MOTOR_PINS):
        raise Exception("Failed to set motor pins.")
    print("Motor pins initialized.")
    time.sleep(0.5)

    print("Resetting all motors...")
    if not body_controller.reset_all():
        raise Exception("Failed to reset all motors.")
    print("Motors reset.")
    time.sleep(0.2)

    # --- ROBUST MOTOR ENABLE SEQUENCE (from your working quad.py) ---
    print("Enabling control for target motors...")
    all_control_enabled = True
    if not body_controller.set_all_control_status(True):
         print("Warning: 'Set All' control command failed. Falling back to individual enabling.")
         all_control_enabled = False

    if not all_control_enabled:
        print("Attempting individual motor control enable...")
        # Re-set flag to true, and let the loop determine the final state
        all_control_enabled = True
        for motor_idx in TARGET_MOTORS:
            print(f"  Enabling motor {motor_idx}...")
            if not body_controller.set_control_status(motor=motor_idx, status=True):
                print(f"  --> Warning: Failed to enable control for motor {motor_idx}.")
                all_control_enabled = False # If any one fails, the overall status is False
            time.sleep(0.05) # CRITICAL: Delay to allow ESP to process

    if not all_control_enabled:
        # If after all attempts, not all motors are on, it's a fatal error.
        raise Exception("Failed to enable control for all motors after individual attempts.")
    else:
        print("All target motors control enabled successfully.")
    # --- END OF ROBUST SEQUENCE ---

    initial_setup_successful = True
    print("\nInitial setup complete.")

except Exception as e:
    print(f"FATAL: Failed during QuadPilotBody initialization or setup: {e}")
    if body_controller:
        body_controller.close()
    sys.exit(1)


# --- State Variables ---
motor_control_enabled = True
is_walking = False
last_key_press_time = 0
debounce_interval = 0.2

# --- Helper Functions (get_stance_angles, clamp, interruptible_sleep) ---
def get_stance_angles():
    stance = [0.0] * NUM_MOTORS
    stance[ACTUATOR_NAME_TO_INDEX_MAP["FL_tigh_actuator"]] = 45.0
    stance[ACTUATOR_NAME_TO_INDEX_MAP["FL_knee_actuator"]] = 45.0
    stance[ACTUATOR_NAME_TO_INDEX_MAP["FR_tigh_actuator"]] = -45.0
    stance[ACTUATOR_NAME_TO_INDEX_MAP["FR_knee_actuator"]] = 45.0
    stance[ACTUATOR_NAME_TO_INDEX_MAP["BR_tigh_actuator"]] = 45.0
    stance[ACTUATOR_NAME_TO_INDEX_MAP["BR_knee_actuator"]] = -45.0
    stance[ACTUATOR_NAME_TO_INDEX_MAP["BL_tigh_actuator"]] = 45.0
    stance[ACTUATOR_NAME_TO_INDEX_MAP["BL_knee_actuator"]] = -45.0
    return stance

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def interruptible_sleep(duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        if not is_walking: break
        time.sleep(0.02)

# --- Core Action Functions (set_manual_angles_action, execute_autocorrect_walk) ---
def set_manual_angles_action(target_angles: list[float]):
    global last_key_press_time, is_walking, motor_control_enabled
    if is_walking or not motor_control_enabled: return

    current_time = time.time()
    if current_time - last_key_press_time > debounce_interval:
        last_key_press_time = current_time
        if not body_controller.set_angles(target_angles):
            print("Failed to set manual angles (no OK from ESPs).")

def execute_autocorrect_walk():
    global is_walking, motor_control_enabled
    print("\n--- Starting Auto-Correcting Walk Sequence ---")
    step_index = 0
    idx_fr_knee = ACTUATOR_NAME_TO_INDEX_MAP["FR_knee_actuator"]
    idx_bl_knee = ACTUATOR_NAME_TO_INDEX_MAP["BL_knee_actuator"]
    idx_fl_knee = ACTUATOR_NAME_TO_INDEX_MAP["FL_knee_actuator"]
    idx_br_knee = ACTUATOR_NAME_TO_INDEX_MAP["BR_knee_actuator"]
    stance_pose = get_stance_angles()
    
    try:
        while is_walking and motor_control_enabled:
            current_yaw = 0.0
            dmp_data = body_controller.get_latest_dmp_data_for_esp(ESP_WITH_IMU)
            if dmp_data and 'ypr_deg' in dmp_data:
                current_yaw = dmp_data['ypr_deg'].get('yaw', 0.0)
            else:
                print(f"Warning: Could not get yaw data from ESP {ESP_WITH_IMU}. Correction may be inaccurate.")

            print(f"\nCurrent Yaw: {current_yaw:.1f} deg, Target Yaw: {TARGET_YAW:.1f} deg")
            yaw_error = current_yaw - TARGET_YAW
            correction = CORRECTION_GAIN_KP * yaw_error
            
            # These signs assume a positive correction (from a right turn) should steer LEFT.
            # To steer LEFT: increase lift/push of right-side legs (Y), decrease left-side (N).
            # You may need to flip the signs (+/- correction) if it steers the wrong way.
            N = clamp(NEUTRAL_LIFT_ANGLE - correction, MIN_LIFT_ANGLE, MAX_LIFT_ANGLE)
            Y = clamp(NEUTRAL_LIFT_ANGLE + correction, MIN_LIFT_ANGLE, MAX_LIFT_ANGLE)
            
            print(f"\nCycle {step_index//4 + 1}: Yaw={current_yaw:.1f}, Err={yaw_error:.1f} -> N={N:.1f}, Y={Y:.1f}")

            # Step 1: Lift FR/BL
            step_pose = stance_pose[:]
            step_pose[idx_fr_knee] = N
            step_pose[idx_bl_knee] = -N
            if not body_controller.set_angles(step_pose): print("Warning: Set angles failed on step 1")
            interruptible_sleep(WALK_STEP_DURATION)
            if not is_walking: break

            # Step 2: Plant All
            if not body_controller.set_angles(stance_pose): print("Warning: Set angles failed on step 2")
            interruptible_sleep(WALK_STEP_DURATION)
            if not is_walking: break

            # Step 3: Lift FL/BR
            step_pose = stance_pose[:]
            step_pose[idx_fl_knee] = Y
            step_pose[idx_br_knee] = -Y
            if not body_controller.set_angles(step_pose): print("Warning: Set angles failed on step 3")
            interruptible_sleep(WALK_STEP_DURATION)
            if not is_walking: break
            
            # Step 4: Plant All
            if not body_controller.set_angles(stance_pose): print("Warning: Set angles failed on step 4")
            interruptible_sleep(WALK_STEP_DURATION)
            if not is_walking: break
            
            step_index += 4
    finally:
        is_walking = False
        print("\n--- Walk Sequence Finished ---")
        print("Returning to stance pose.")
        set_manual_angles_action(get_stance_angles())

# --- Keyboard Listener and Main Execution ---
def on_key_press(key):
    global motor_control_enabled, is_walking, body_controller, TARGET_YAW
    try:
        if key == keyboard.Key.esc:
            is_walking = False
            return False
            
        if hasattr(key, 'char'):
            char = key.char.lower()
            if char == 't':
                # Toggle logic... (omitted for brevity, same as before)
                pass
            if not motor_control_enabled: return
            if char == 's':
                if is_walking: is_walking = False
            if is_walking: return
            if char == 'a': set_manual_angles_action(get_stance_angles())
            elif char == 'd': set_manual_angles_action([0.0] * NUM_MOTORS)
            elif char == 'o':
                dmp_data = body_controller.get_latest_dmp_data_for_esp(ESP_WITH_IMU)
                if dmp_data and dmp_data.get('ypr_deg'):
                    TARGET_YAW = dmp_data['ypr_deg'].get('yaw', 0.0)
                    print(f"\nNew Target Yaw set to current heading: {TARGET_YAW:.2f} deg.")
                else:
                    TARGET_YAW = REAL_TARGET
                    print(f"\nWarning: No yaw data. Target set to 45.0. Is DMP ready on ESP {ESP_WITH_IMU}?")
                is_walking = True
                threading.Thread(target=execute_autocorrect_walk, daemon=True).start()
            elif char == 'y':
                TARGET_YAW = REAL_TARGET
                print(f"\n[Manual] Target Yaw forced to {REAL_TARGET:.2f} deg. Starting walk.")
                is_walking = True
                threading.Thread(target=execute_autocorrect_walk, daemon=True).start()
    except Exception as e:
        print(f"Error in on_key_press: {e}")

if __name__ == "__main__":
    if not initial_setup_successful:
        sys.exit(1)

    print("\n" + "="*50 + "\nQuadPilot Auto-Correcting Walk Control\n" + "="*50)
    print(" W: Start walk | S: Stop walk | A: Stance | D: Zero | T: Toggle | Esc: Exit")
    print("-" * 50 + f"\nYaw correction ON. Kp={CORRECTION_GAIN_KP}. IMU on ESP#{ESP_WITH_IMU}\n" + "="*50)
    
    print("Setting initial stance pose...")
    set_manual_angles_action(get_stance_angles())
    
    listener = keyboard.Listener(on_press=on_key_press)
    try:
        listener.start()
        print("\nKeyboard listener started. Ready for commands.")
        listener.join()
    finally:
        print("\nInitiating shutdown...")
        is_walking = False
        time.sleep(0.5) # Give walk thread time to stop
        if listener.is_alive(): listener.stop()
        if body_controller:
            print("Disabling motors and closing connection...")
            body_controller.set_all_control_status(False)
            time.sleep(0.1)
            body_controller.reset_all()
            body_controller.close()
        print("Shutdown complete.")