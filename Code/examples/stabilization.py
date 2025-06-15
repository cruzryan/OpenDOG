# stabilization_final_candidate.py

import os
import sys
import time
import threading
from pynput import keyboard
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

try:
    from quadpilot import QuadPilotBody
except ImportError:
    print(f"ERROR: QuadPilotBody import failed. Ensure 'quadpilot.py' is in: {code_dir}")
    sys.exit(1)

# --- Configuration ---
NUM_MOTORS = 8
ACTUATOR_NAME_TO_INDEX_MAP = {
    "FL_tigh_actuator": 3, "FL_knee_actuator": 0,
    "FR_tigh_actuator": 1, "FR_knee_actuator": 2,
    "BR_tigh_actuator": 5, "BR_knee_actuator": 4,
    "BL_tigh_actuator": 7, "BL_knee_actuator": 6,
}
PRINT_ORDER = [
    "FR_tigh_actuator", "FR_knee_actuator", "FL_tigh_actuator", "FL_knee_actuator",
    "BR_tigh_actuator", "BR_knee_actuator", "BL_tigh_actuator", "BL_knee_actuator"
]
MOTOR_PINS = [
    (39, 40, 41, 42), (16, 15, 7, 6), (17, 18, 5, 4), (37, 38, 1, 2),
    (37, 38, 1, 2), (40, 39, 42, 41), (15, 16, 6, 7), (18, 17, 4, 5),
]

# --- Stabilization Configuration ---
STABILIZATION_KP = -2.0 # As per your feedback
PRIMARY_IMU_ESP_INDEX = 1 # Corrected ESP index

KNEE_FL_IDX = ACTUATOR_NAME_TO_INDEX_MAP["FL_knee_actuator"]
KNEE_FR_IDX = ACTUATOR_NAME_TO_INDEX_MAP["FR_knee_actuator"]
KNEE_BL_IDX = ACTUATOR_NAME_TO_INDEX_MAP["BL_knee_actuator"]
KNEE_BR_IDX = ACTUATOR_NAME_TO_INDEX_MAP["BR_knee_actuator"]
THIGH_FL_IDX = ACTUATOR_NAME_TO_INDEX_MAP["FL_tigh_actuator"]
THIGH_FR_IDX = ACTUATOR_NAME_TO_INDEX_MAP["FR_tigh_actuator"]
THIGH_BL_IDX = ACTUATOR_NAME_TO_INDEX_MAP["BL_tigh_actuator"]
THIGH_BR_IDX = ACTUATOR_NAME_TO_INDEX_MAP["BR_tigh_actuator"]

KNEE_ANGLE_POS_MIN = 20.0
KNEE_ANGLE_POS_MAX = 55.0 
KNEE_ANGLE_NEG_MIN = -55.0 
KNEE_ANGLE_NEG_MAX = -20.0
THIGH_FR_MIN = -65.0 # e.g. home -45 +/- 15
THIGH_FR_MAX = -25.0
THIGH_FL_MIN = 25.0  # e.g. home +45 +/- 15
THIGH_FL_MAX = 65.0
THIGH_BR_MIN = 25.0  # e.g. home +45 +/- 15
THIGH_BR_MAX = 65.0
THIGH_BL_MIN = 25.0  # e.g. home +45 +/- 15
THIGH_BL_MAX = 65.0

# --- Global Variable for Body Controller ---
body_controller = None # Will hold the Phase 2 instance

# --- Initialization Phase 1: Command-Only Setup ---
initial_command_setup_successful = False
print("--- PHASE 1: Initial Command-Only Setup (like quad.py) ---")
try:
    temp_body_controller = QuadPilotBody(listen_for_broadcasts=False)
    print("QuadPilotBody (command-only instance) initialized for Phase 1.")
    time.sleep(0.2)

    print("Phase 1: Setting initial control parameters...")
    if not temp_body_controller.set_control_params(P=0.9, I=0.0, D=0.9, dead_zone=5, pos_thresh=5):
        raise Exception("Phase 1: Failed to set control parameters.")
    print("Phase 1: Control parameters set.")
    time.sleep(0.1)

    print("Phase 1: Initializing all motor pins...")
    if not temp_body_controller.set_all_pins(MOTOR_PINS):
        raise Exception("Phase 1: Failed to set motor pins.")
    print("Phase 1: Motor pins initialized.")
    time.sleep(0.5) 

    print("Phase 1: Resetting all motors...")
    if not temp_body_controller.reset_all():
        raise Exception("Phase 1: Failed to reset all motors.")
    print("Phase 1: Motors reset.")
    time.sleep(0.2)

    print("Phase 1: Enabling control for target motors...")
    all_control_enabled_phase1 = True
    if not temp_body_controller.set_all_control_status(True):
         print("Phase 1 Warning: Failed to enable control for one or more motors via set_all_control_status.")
         all_control_enabled_phase1 = False
    
    if not all_control_enabled_phase1: # Fallback logic from your quad.py
        print("Phase 1: Attempting individual motor control enable...")
        temp_target_motors = list(range(NUM_MOTORS))
        for motor_idx in temp_target_motors:
            if not temp_body_controller.set_control_status(motor=motor_idx, status=True):
                print(f"Phase 1 Warning: Failed to enable control for motor {motor_idx} individually.")
                all_control_enabled_phase1 = False 
            time.sleep(0.05)
    
    if not all_control_enabled_phase1:
        raise Exception("Phase 1: Not all motors control could be enabled.")
    else:
        print("Phase 1: All target motors control enabled.")

    initial_command_setup_successful = True
    print("--- PHASE 1: Command-Only Setup Successful. Closing command instance. ---")
    temp_body_controller.close()
    temp_body_controller = None 
    time.sleep(0.5) 

except Exception as e:
    print(f"FATAL: PHASE 1 Setup Failed: {e}")
    if 'temp_body_controller' in locals() and temp_body_controller:
        temp_body_controller.close()
    sys.exit(1)

# --- Initialization Phase 2: Listener Setup (for Stabilization) ---
if not initial_command_setup_successful:
    print("FATAL: Phase 1 was not successful. Exiting.")
    sys.exit(1)

print("\n--- PHASE 2: Listener Instance Setup (for Stabilization) ---")
initial_listener_setup_successful = False
try:
    body_controller = QuadPilotBody(listen_for_broadcasts=True) # Global body_controller
    print("QuadPilotBody (listener instance) initialized for Phase 2.")
    time.sleep(1.0) 

    # Assuming ESPs retain state from Phase 1.
    # Check if data is coming from the IMU ESP
    if not body_controller.is_data_available_from_esp(PRIMARY_IMU_ESP_INDEX):
        print(f"Phase 2 Warning: No data received from IMU ESP ({PRIMARY_IMU_ESP_INDEX+1}) yet. Waiting briefly...")
        time.sleep(1.0)
        if not body_controller.is_data_available_from_esp(PRIMARY_IMU_ESP_INDEX):
             print(f"Phase 2 Warning: Still no data from IMU ESP ({PRIMARY_IMU_ESP_INDEX+1}). Stabilization might fail.")

    initial_listener_setup_successful = True
    print("--- PHASE 2: Listener Instance Setup Assumed Successful. ---")

except Exception as e:
    print(f"FATAL: PHASE 2 Setup Failed: {e}")
    if body_controller:
        body_controller.close()
    sys.exit(1)

# --- State Variables ---
motor_control_enabled = True 
is_stabilizing = False
stabilization_thread = None
last_key_press_time = 0
debounce_interval = 0.2

# --- Helper Functions ---
def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def get_home_angles(): 
    home = [0.0] * NUM_MOTORS
    home[THIGH_FL_IDX] = 45.0; home[KNEE_FL_IDX] = 45.0
    home[THIGH_FR_IDX] = -45.0; home[KNEE_FR_IDX] = 45.0
    home[THIGH_BR_IDX] = 45.0; home[KNEE_BR_IDX] = -45.0
    home[THIGH_BL_IDX] = 45.0; home[KNEE_BL_IDX] = -45.0
    return home

def print_target_angles(angles_list: list[float], prefix="Target Angles:"):
    print(f"\n{prefix}")
    for name in PRINT_ORDER:
        index = ACTUATOR_NAME_TO_INDEX_MAP.get(name)
        if index is not None and 0 <= index < len(angles_list):
            print(f"  {name} (idx {index}): {angles_list[index]:.2f}")

def set_manual_angles_action(target_angles: list[float]):
    global last_key_press_time, is_stabilizing, motor_control_enabled, body_controller
    if is_stabilizing: print("LOG: Manual set blocked: Stabilization active."); return
    if not motor_control_enabled: print("LOG: Manual set blocked: Motor control disabled."); return
    current_time = time.time()
    if current_time - last_key_press_time > debounce_interval:
        last_key_press_time = current_time
        print("\nLOG: Setting manual angles...")
        if body_controller.set_angles(target_angles):
            print_target_angles(target_angles, prefix="Manual Angles Set:")
            print("LOG: Manual angles set successfully.")
        else: print("LOG: Failed to set manual angles.")

# --- Stabilization Loop ---
def stabilization_loop_fn():
    global is_stabilizing, motor_control_enabled, body_controller
    print(f"LOG STAB LOOP: Entered. Targeting ESP {PRIMARY_IMU_ESP_INDEX+1}.")
    base_home_angles = get_home_angles()
    current_target_angles = list(base_home_angles) 
    loop_count = 0
    while is_stabilizing and motor_control_enabled:
        loop_count += 1
        if not body_controller: print("LOG STAB LOOP: body_controller is None!"); is_stabilizing = False; break

        esp_motor_data_packet = body_controller.get_latest_motor_data_for_esp(PRIMARY_IMU_ESP_INDEX)
        dmp_flag_from_motor_packet = esp_motor_data_packet.get('dmp_ready', False) if esp_motor_data_packet else False
        diagnostic_dmp_data = body_controller.get_latest_dmp_data_for_esp(PRIMARY_IMU_ESP_INDEX)
        
        if loop_count % 40 == 1: # Reduced frequency for diagnostic print
            print(f"LOG STAB DIAG [{loop_count}]: ESP {PRIMARY_IMU_ESP_INDEX+1}: MotPktRecv:{esp_motor_data_packet is not None}, DMPFlagMotPkt:{dmp_flag_from_motor_packet}, DMPData:{diagnostic_dmp_data is not None}")

        if not dmp_flag_from_motor_packet:
            if loop_count % 80 == 1: print(f"LOG STAB LOOP [{loop_count}]: ESP {PRIMARY_IMU_ESP_INDEX+1} 'dmp_ready' in mot_pkt False. Waiting...")
            time.sleep(0.1); continue
        if not diagnostic_dmp_data or 'ypr_deg' not in diagnostic_dmp_data:
            if loop_count % 40 == 1: print(f"LOG STAB LOOP [{loop_count}]: ESP {PRIMARY_IMU_ESP_INDEX+1} 'dmp_ready' True, but no YPR. Data: {diagnostic_dmp_data}")
            time.sleep(0.05); continue

        roll = diagnostic_dmp_data['ypr_deg'].get('roll', 0.0)
        if loop_count % 20 == 1 or abs(roll) > 1.0 : 
            print(f"LOG STAB LOOP [{loop_count}]: ESP {PRIMARY_IMU_ESP_INDEX+1} - Roll: {roll:.2f} deg")

        adjustment_val = STABILIZATION_KP * roll 
        current_target_angles = list(base_home_angles) 
        # Right Side Actuators (FR, BR)
        current_target_angles[THIGH_FR_IDX] = clamp(base_home_angles[THIGH_FR_IDX] + adjustment_val, THIGH_FR_MIN, THIGH_FR_MAX)
        current_target_angles[KNEE_FR_IDX]  = clamp(base_home_angles[KNEE_FR_IDX]  + adjustment_val, KNEE_ANGLE_POS_MIN, KNEE_ANGLE_POS_MAX)
        current_target_angles[THIGH_BR_IDX] = clamp(base_home_angles[THIGH_BR_IDX] + adjustment_val, THIGH_BR_MIN, THIGH_BR_MAX)
        current_target_angles[KNEE_BR_IDX]  = clamp(base_home_angles[KNEE_BR_IDX]  - adjustment_val, KNEE_ANGLE_NEG_MIN, KNEE_ANGLE_NEG_MAX)
        # Left Side Actuators (FL, BL)
        current_target_angles[THIGH_FL_IDX] = clamp(base_home_angles[THIGH_FL_IDX] - adjustment_val, THIGH_FL_MIN, THIGH_FL_MAX)
        current_target_angles[KNEE_FL_IDX]  = clamp(base_home_angles[KNEE_FL_IDX]  - adjustment_val, KNEE_ANGLE_POS_MIN, KNEE_ANGLE_POS_MAX)
        current_target_angles[THIGH_BL_IDX] = clamp(base_home_angles[THIGH_BL_IDX] - adjustment_val, THIGH_BL_MIN, THIGH_BL_MAX)
        current_target_angles[KNEE_BL_IDX]  = clamp(base_home_angles[KNEE_BL_IDX]  + adjustment_val, KNEE_ANGLE_NEG_MIN, KNEE_ANGLE_NEG_MAX)
        
        if loop_count % 50 == 1 or abs(roll) > 1.0 : 
            print_target_angles(current_target_angles, prefix=f"STAB LOOP [{loop_count}] SetAngles (R={roll:.1f}):")

        if not body_controller.set_angles(current_target_angles) and (loop_count % 50 == 1 or abs(roll) > 1.0): 
            print(f"LOG STAB LOOP [{loop_count}]: WARN - set_angles no OK.")
        time.sleep(0.02)

    print(f"LOG STAB LOOP: Exited. is_stabilizing={is_stabilizing}")
    if motor_control_enabled and body_controller:
        print("LOG STAB LOOP: Setting to home pose on exit.")
        if body_controller.set_angles(get_home_angles()):
             print_target_angles(get_home_angles(), prefix="STAB LOOP: Home Pose Set on Exit:")
        else: print("LOG STAB LOOP: Failed to set home pose on exit.")

# --- Keyboard Listener Callback ---
def on_key_press(key):
    global motor_control_enabled, body_controller, is_stabilizing, stabilization_thread
    try:
        if key == keyboard.Key.esc: print("LOG: Esc pressed."); is_stabilizing = False; return False 
        if hasattr(key, 'char') and key.char:
            char = key.char.lower(); print(f"\nLOG: Key '{char}' pressed.")
            if char == 't':
                print("LOG: 't' for motor toggle.")
                if is_stabilizing: print("LOG: Stab active, stopping it."); is_stabilizing = False
                if stabilization_thread and stabilization_thread.is_alive(): stabilization_thread.join(timeout=0.5)
                stabilization_thread = None
                motor_control_enabled = not motor_control_enabled
                print(f"LOG: Motor ctrl: {'ENABLED' if motor_control_enabled else 'DISABLED'}")
                if body_controller:
                    if motor_control_enabled:
                        print("LOG: Enabling motors (Phase 2)...")
                        if not body_controller.reset_all(): print("LOG: P2 reset_all failed.")
                        time.sleep(0.2)
                        if not body_controller.set_all_control_status(True): print("LOG: P2 set_all_ctrl(T) failed.")
                        else: print("LOG: P2 Motors re-enabled."); set_manual_angles_action(get_home_angles()) if not is_stabilizing else None
                    else: # Disabling
                        print("LOG: Disabling motors (Phase 2)..."); is_stabilizing = False
                        if not body_controller.set_all_control_status(False): print("LOG: P2 set_all_ctrl(F) failed.")
                else: print("LOG: body_controller is None for 't' toggle.")
                return
            if not motor_control_enabled: print(f"LOG: Motor ctrl disabled. Key '{char}' ignored."); return
            if char == 'b':
                print("LOG: 'b' for stab toggle.")
                is_stabilizing = not is_stabilizing
                if is_stabilizing:
                    if not body_controller: print("LOG: No body_controller for stab."); is_stabilizing = False; return
                    print("LOG: Starting stab..."); set_manual_angles_action(get_home_angles()) # Set home before starting
                    time.sleep(0.1) 
                    if not (stabilization_thread and stabilization_thread.is_alive()):
                        stabilization_thread = threading.Thread(target=stabilization_loop_fn, daemon=True); stabilization_thread.start()
                    else: print("LOG: Stab thread already running?")
                else: # Stopping stabilization
                    print("LOG: Stopping stab.")
                    if stabilization_thread and stabilization_thread.is_alive(): stabilization_thread.join(timeout=1.0)
                    stabilization_thread = None; print("LOG: Stab stopped by 'b'.")
                return
            if is_stabilizing: print(f"LOG: Stab active. Key '{char}' ignored."); return
            if char == 'a': print("\nLOG: 'a' for home."); set_manual_angles_action(get_home_angles())
            elif char == 'd': print("\nLOG: 'd' for zero."); set_manual_angles_action([0.0] * NUM_MOTORS)
    except Exception as e: print(f"LOG: Key press error: {e}"); traceback.print_exc()

# --- Main Execution ---
if __name__ == "__main__":
    if not initial_command_setup_successful or not initial_listener_setup_successful:
        print("LOG: Exiting due to failed initial setup in one of the phases.")
        if body_controller: body_controller.close()
        sys.exit(1)

    print(f"\nQuadPilot Stabilize (2-Phase, Thighs+Knees, ESP{PRIMARY_IMU_ESP_INDEX+1}, KP={STABILIZATION_KP})")
    print("-----------------------------------------------------------------------------")
    print(" T: Toggle motor enable/disable | A: Home pose | D: Zero pose");
    print(f" B: Toggle IMU stabilization  | Esc: Exit")
    print("-----------------------------------------------------------------------------")

    if body_controller:
        print("LOG: Setting initial home pose (P2 controller)...")
        if body_controller.set_angles(get_home_angles()):
            print_target_angles(get_home_angles(), prefix="Initial Home Pose:")
            print("LOG: Initial home pose set.")
        else: print("LOG: WARN - Failed to set initial home pose (P2).")
    else: print("LOG: ERROR - body_controller None after setup. No initial pose.")

    listener = keyboard.Listener(on_press=on_key_press)
    try:
        listener.start(); print("\nLOG: Keyboard listener started.")
        listener.join() 
    except KeyboardInterrupt: print("\nLOG: Ctrl+C detected.")
    except Exception as e: print(f"LOG: Main loop error: {e}"); traceback.print_exc()
    finally:
        print("\nLOG: Initiating shutdown...")
        is_stabilizing = False 
        if stabilization_thread and stabilization_thread.is_alive(): stabilization_thread.join(timeout=1.0)
        if listener.is_alive(): listener.stop()
        if body_controller:
            print("LOG: Final motor safety (P2 instance)...")
            try:
                print("LOG: Disabling motor control..."); body_controller.set_all_control_status(False)
                # print("LOG: Resetting motors..."); body_controller.reset_all() # Optional reset
            except Exception as e: print(f"LOG: Error during motor safety ops: {e}")
            finally: print("LOG: Closing QuadPilotBody (P2)..."); body_controller.close()
        print("LOG: Shutdown complete.")