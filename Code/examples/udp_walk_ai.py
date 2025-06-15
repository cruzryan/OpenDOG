# quad_autocorrect_policy.py (v4 - PyTorch Policy Integration)

import os
import sys
import time
import threading
from pynput import keyboard
import math

# --- PyTorch Imports ---
# Ensure you have PyTorch installed: pip install torch
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("FATAL ERROR: PyTorch is not installed.")
    print("Please install it by running: pip install torch")
    sys.exit(1)


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
MODEL_PATH = 'walk_policy.pth' # Path to your trained model file

# --- Yaw Auto-Correction Configuration ---
REAL_TARGET = 0.0
TARGET_YAW = REAL_TARGET
# CORRECTION_GAIN_KP is now replaced by the neural network
NEUTRAL_LIFT_ANGLE = 30.0 # Still useful for reference and potential clamping
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


# --- Neural Network Definition (must match the training script) ---
class WalkPolicy(nn.Module):
    def __init__(self):
        super(WalkPolicy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 64),      # Input layer (1 neuron) -> Hidden layer (64 neurons)
            nn.ReLU(),             # Activation function
            nn.Linear(64, 64),     # Hidden layer -> Hidden layer
            nn.ReLU(),             # Activation function
            nn.Linear(64, 2)       # Hidden layer -> Output layer (2 neurons for N and Y)
        )

    def forward(self, x):
        return self.network(x)


# --- Load Trained Policy Model ---
walk_policy_model = None
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. "
                                f"Please ensure it's in the same directory as the script.")
    print(f"Loading walk policy from '{MODEL_PATH}'...")
    walk_policy_model = WalkPolicy()
    walk_policy_model.load_state_dict(torch.load(MODEL_PATH))
    walk_policy_model.eval()  # Set the model to evaluation mode (very important!)
    print("Walk policy model loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load the PyTorch model: {e}")
    sys.exit(1)


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

    # --- ROBUST MOTOR ENABLE SEQUENCE ---
    print("Enabling control for target motors...")
    all_control_enabled = True
    if not body_controller.set_all_control_status(True):
         print("Warning: 'Set All' control command failed. Falling back to individual enabling.")
         all_control_enabled = False

    if not all_control_enabled:
        print("Attempting individual motor control enable...")
        all_control_enabled = True
        for motor_idx in TARGET_MOTORS:
            print(f"  Enabling motor {motor_idx}...")
            if not body_controller.set_control_status(motor=motor_idx, status=True):
                print(f"  --> Warning: Failed to enable control for motor {motor_idx}.")
                all_control_enabled = False
            time.sleep(0.05)

    if not all_control_enabled:
        raise Exception("Failed to enable control for all motors after individual attempts.")
    else:
        print("All target motors control enabled successfully.")

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


# --- Helper Functions ---
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

def get_policy_action(yaw_error: float) -> tuple[float, float]:
    """
    Uses the loaded neural network to determine the N and Y lift angles.
    """
    # Convert the yaw_error float into a 2D tensor for the network
    input_tensor = torch.FloatTensor([[yaw_error]])
    
    with torch.no_grad(): # Disable gradient calculation for inference
        predicted_N, predicted_Y = walk_policy_model(input_tensor).squeeze().tolist()
        
    return predicted_N, predicted_Y

# --- Core Action Functions ---
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
    print("\n--- Starting Auto-Correcting Walk Sequence (using NN Policy) ---")
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
            
            # --- THIS IS THE MODIFIED SECTION ---
            yaw_error = current_yaw - REAL_TARGET
            
            # Get the N and Y angles from our trained policy model
            N, Y = get_policy_action(yaw_error)

            # Safety clamp: Even though the model should learn the limits, it's wise
            # to enforce them here to prevent accidental damage.
            N = clamp(N, MIN_LIFT_ANGLE, MAX_LIFT_ANGLE)
            Y = clamp(Y, MIN_LIFT_ANGLE, MAX_LIFT_ANGLE)
            # --- END OF MODIFICATION ---
            
            print(f"\nCycle {step_index//4 + 1}: Yaw={current_yaw:.1f}, Err={yaw_error:.1f} -> Policy says N={N:.1f}, Y={Y:.1f}")

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
                motor_control_enabled = not motor_control_enabled
                body_controller.set_all_control_status(motor_control_enabled)
                status_str = "ENABLED" if motor_control_enabled else "DISABLED"
                print(f"\n--- Motor control {status_str} ---")
            
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
                    print(f"\nWarning: No yaw data. Target set to {REAL_TARGET}. Is DMP ready on ESP {ESP_WITH_IMU}?")
                is_walking = True
                threading.Thread(target=execute_autocorrect_walk, daemon=True).start()
            elif char == 'y': # Kept 'y' key for consistency
                TARGET_YAW = REAL_TARGET
                print(f"\n[Manual] Target Yaw forced to {REAL_TARGET:.2f} deg. Starting walk.")
                is_walking = True
                threading.Thread(target=execute_autocorrect_walk, daemon=True).start()
    except Exception as e:
        print(f"Error in on_key_press: {e}")

if __name__ == "__main__":
    if not initial_setup_successful or not walk_policy_model:
        sys.exit(1)

    print("\n" + "="*50 + "\nQuadPilot Auto-Correcting Walk Control (NN Policy)\n" + "="*50)
    print(" o: Start walk (sets current yaw as target) | y: Start walk (sets 0 as target)")
    print(" s: Stop walk | a: Stance | d: Zero | t: Toggle Motors | Esc: Exit")
    print("-" * 50 + f"\nYaw correction ON (Policy: {MODEL_PATH}). IMU on ESP#{ESP_WITH_IMU}\n" + "="*50)
    
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
        time.sleep(0.5)
        if listener.is_alive(): listener.stop()
        if body_controller:
            print("Disabling motors and closing connection...")
            body_controller.set_all_control_status(False)
            time.sleep(0.1)
            body_controller.reset_all()
            body_controller.close()
        print("Shutdown complete.")