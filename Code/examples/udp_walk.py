# quad.py

import os
import sys
import time
import threading
import json # For loading walk sequence
from pynput import keyboard

# Add the code directory (parent of this script's directory) to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir) # This is the parent directory
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

try:
    # Assuming quadpilot.py is in the 'code_dir' (parent directory)
    from quadpilot import QuadPilotBody
except ImportError:
    print(f"ERROR: Could not import QuadPilotBody from 'quadpilot.py'. "
          f"Ensure 'quadpilot.py' is in the directory: {code_dir}")
    sys.exit(1)


# --- Configuration ---
JSON_WALK_FILE = os.path.join(script_dir, 'walk.json') # Expect walk.json in the same dir as quad.py
NUM_MOTORS = 8
TARGET_MOTORS = list(range(NUM_MOTORS))

# Mapping: JSON Actuator Name -> QuadPilotBody Index (0-7)
ACTUATOR_NAME_TO_INDEX_MAP = {
    "FL_tigh_actuator": 3, "FL_knee_actuator": 0,
    "FR_tigh_actuator": 1, "FR_knee_actuator": 2,
    "BR_tigh_actuator": 5, "BR_knee_actuator": 4,
    "BL_tigh_actuator": 7, "BL_knee_actuator": 6,
}

# Desired print order for angles
PRINT_ORDER = [
    "FR_tigh_actuator", "FR_knee_actuator", "FL_tigh_actuator", "FL_knee_actuator",
    "BR_tigh_actuator", "BR_knee_actuator", "BL_tigh_actuator", "BL_knee_actuator"
]

if len(ACTUATOR_NAME_TO_INDEX_MAP) != NUM_MOTORS:
    print("Warning: ACTUATOR_NAME_TO_INDEX_MAP does not contain exactly 8 entries!")

# Motor pin configurations (ENCODER_A, ENCODER_B, IN1, IN2)
# Order corresponds to motor indices 0-7
MOTOR_PINS = [
    (39, 40, 41, 42),  # Motor 0 (FL_knee) -> ESP1_M0
    (16, 15, 7, 6),    # Motor 1 (Now FL_tigh, was FR_tigh) -> ESP1_M1
    (17, 18, 5, 4),    # Motor 2 (FR_knee) -> ESP1_M2
    (37, 38, 1, 2),    # Motor 3 (Now FR_tigh, was FL_tigh) -> ESP1_M3
    (37, 38, 1, 2),    # Motor 4 (BR_knee) -> ESP2_M0
    (40, 39, 42, 41),  # Motor 5 (BR_tigh) -> ESP2_M1
    (15, 16, 6, 7),    # Motor 6 (BL_knee) -> ESP2_M2
    (18, 17, 4, 5),    # Motor 7 (BL_tigh) -> ESP2_M3
]

# --- Load Walk Sequence ---
walk_sequence = None
try:
    with open(JSON_WALK_FILE, 'r') as f:
        walk_sequence = json.load(f)
    print(f"Successfully loaded walk sequence from '{JSON_WALK_FILE}' with {len(walk_sequence)} steps.")
except FileNotFoundError:
    print(f"Error: Walk sequence file not found at '{JSON_WALK_FILE}'. 'W' key will not function.")
except json.JSONDecodeError as e:
    print(f"Error: Could not decode JSON from '{JSON_WALK_FILE}': {e}. 'W' key will not function.")
except Exception as e:
    print(f"An unexpected error occurred loading '{JSON_WALK_FILE}': {e}. 'W' key will not function.")


# --- Initialize QuadPilotBody and Setup ---
body_controller = None
initial_setup_successful = False
try:
    # Initialize with listen_for_broadcasts=False for a command-focused instance
    body_controller = QuadPilotBody(listen_for_broadcasts=False)
    print("QuadPilotBody (command instance) initialized.")

    print("Setting initial control parameters...")
    if not body_controller.set_control_params(P=0.9, I=0.001, D=0.3, dead_zone=10, pos_thresh=5):
        raise Exception("Failed to set control parameters.")
    print("Control parameters set.")
    time.sleep(0.1)

    print("Initializing all motor pins...")
    if not body_controller.set_all_pins(MOTOR_PINS):
        raise Exception("Failed to set motor pins.")
    print("Motor pins initialized.")
    time.sleep(0.5) # Give time for pins to settle

    print("Resetting all motors...")
    if not body_controller.reset_all():
        raise Exception("Failed to reset all motors.")
    print("Motors reset.")
    time.sleep(0.2)

    print("Enabling control for target motors...")
    all_control_enabled = True
    # Using set_all_control_status which iterates internally
    if not body_controller.set_all_control_status(True):
         print("Warning: Failed to enable control for one or more motors via set_all_control_status.")
         # Fallback to individual enabling if needed, or just accept partial success
         all_control_enabled = False # Or handle more gracefully

    if not all_control_enabled:
        # Optionally, try individual enabling here if the above failed for some motors
        print("Attempting individual motor control enable...")
        for motor_idx in TARGET_MOTORS:
            if not body_controller.set_control_status(motor=motor_idx, status=True):
                print(f"Warning: Failed to enable control for motor {motor_idx} individually.")
                all_control_enabled = False # Can decide how critical this is
            time.sleep(0.05)

    if not all_control_enabled:
         print("Warning: Not all motors might be under control.")
    else:
        print("All target motors control enabled.")

    initial_setup_successful = True
    print("Initial setup complete.")

except Exception as e:
    print(f"FATAL: Failed during QuadPilotBody initialization or initial setup: {e}")
    if body_controller:
        body_controller.close()
    sys.exit(1)


# --- State Variables ---
motor_control_enabled = True # Assume enabled after successful init
is_walking = False
last_key_press_time = 0
debounce_interval = 0.2 # Debounce for manual angle setting (A, D keys)

# --- Helper Functions ---
def get_home_angles():
    """Returns the defined home pose angles."""
    home = [0.0] * NUM_MOTORS
    home[ACTUATOR_NAME_TO_INDEX_MAP["FL_tigh_actuator"]] = 45.0
    home[ACTUATOR_NAME_TO_INDEX_MAP["FL_knee_actuator"]] = 45.0
    home[ACTUATOR_NAME_TO_INDEX_MAP["FR_tigh_actuator"]] = -45.0
    home[ACTUATOR_NAME_TO_INDEX_MAP["FR_knee_actuator"]] = 45.0
    home[ACTUATOR_NAME_TO_INDEX_MAP["BR_tigh_actuator"]] = 45.0
    home[ACTUATOR_NAME_TO_INDEX_MAP["BR_knee_actuator"]] = -45.0
    home[ACTUATOR_NAME_TO_INDEX_MAP["BL_tigh_actuator"]] = 45.0
    home[ACTUATOR_NAME_TO_INDEX_MAP["BL_knee_actuator"]] = -45.0
    return home

def print_target_angles(angles_list: list[float]):
    """Prints angles based on PRINT_ORDER and ACTUATOR_NAME_TO_INDEX_MAP."""
    print("Target Motor Angles:")
    for name in PRINT_ORDER:
        index = ACTUATOR_NAME_TO_INDEX_MAP.get(name)
        if index is not None and 0 <= index < len(angles_list):
            print(f"  {name}: {angles_list[index]:.1f} degrees")
        else:
            print(f"  {name}: Not Mapped or Index Out of Bounds")

def set_manual_angles_action(target_angles: list[float]):
    global last_key_press_time, is_walking, motor_control_enabled, body_controller
    if is_walking:
        print("Cannot set manual angles while walking.")
        return
    if not motor_control_enabled:
        print("Motor control is disabled. Cannot set angles.")
        return

    current_time = time.time()
    if current_time - last_key_press_time > debounce_interval:
        last_key_press_time = current_time
        print("\nSetting manual angles...")
        if body_controller.set_angles(target_angles):
            print_target_angles(target_angles)
            print("Manual angles set successfully.")
        else:
            print("Failed to set manual angles (no OK from ESPs).")
    # else: print("Debounced manual angle set.")


def execute_walk_sequence_action(sequence, name_map):
    global is_walking, motor_control_enabled, body_controller
    if not motor_control_enabled:
        print("Motor control disabled. Cannot start walk sequence.")
        is_walking = False
        return
    if not sequence:
        print("No walk sequence loaded or sequence is empty.")
        is_walking = False
        return

    print("\n--- Starting Walk Sequence ---")
    # Use home pose as the starting point for interpolation/last known angles
    last_angles_sent = get_home_angles()[:] # Make a copy

    try:
        for i, step_data in enumerate(sequence):
            if not is_walking or not motor_control_enabled: # Check flags
                print("\nWalk sequence interrupted (flag changed).")
                break

            duration = step_data.get("duration", 0.1)
            targets_in_step = step_data.get("targets_deg", {})
            current_step_target_angles = last_angles_sent[:] # Start with previous angles

            print(f"\n--- Step {i+1}/{len(sequence)} --- Duration: {duration:.2f}s")
            print("Targets for this step:")

            for name, target_angle_val in targets_in_step.items():
                motor_idx = name_map.get(name)
                if motor_idx is not None and 0 <= motor_idx < NUM_MOTORS:
                    current_step_target_angles[motor_idx] = float(target_angle_val)
                    # Print only defined targets for this step
                    for p_name in PRINT_ORDER:
                        if p_name == name:
                             print(f"  {name}: {float(target_angle_val):.1f} degrees")
                else:
                    print(f"Warning: Actuator '{name}' in walk sequence not in map.")

            if not body_controller.set_angles(current_step_target_angles):
                print(f"Warning: Failed to set angles for step {i+1}. Continuing sequence...")
                # Decide if you want to stop the sequence on failure or just warn

            last_angles_sent = current_step_target_angles[:] # Update for next iteration

            # Interruptible sleep
            start_step_time = time.time()
            sleep_chunk = 0.02 # Check flags every 20ms
            while time.time() - start_step_time < duration:
                if not is_walking or not motor_control_enabled:
                    break
                time.sleep(min(sleep_chunk, duration - (time.time() - start_step_time) + 0.001))
            
            if not is_walking or not motor_control_enabled: # Check again after sleep
                print("\nWalk sequence stopping after step.")
                break
        
        print("--- Walk Sequence Finished (or stopped) ---")

    except Exception as e:
        print(f"Error during walk sequence execution: {e}")
    finally:
        is_walking = False


# --- Keyboard Listener Callback ---
def on_key_press(key):
    global motor_control_enabled, is_walking, body_controller

    try:
        if key == keyboard.Key.esc:
            print("\nEscape pressed, exiting application...")
            is_walking = False # Signal walk thread to stop
            return False # Stop pynput listener

        if hasattr(key, 'char') and key.char:
            char = key.char.lower()

            if char == 't': # Toggle motor control
                if is_walking:
                    print("Cannot toggle motor control while walking. Stop sequence ('s') first.")
                    return

                motor_control_enabled = not motor_control_enabled
                status_text = "ENABLED" if motor_control_enabled else "DISABLED"
                print(f"\nMotor control TOGGLED to: {status_text}")

                if motor_control_enabled:
                    print("Resetting motors and enabling control...")
                    if not body_controller.reset_all(): print("Warning: Reset all failed.")
                    time.sleep(0.2)
                    if not body_controller.set_all_control_status(True):
                        print("Warning: Failed to enable all motor controls.")
                    else:
                        print("Motors re-enabled. Setting to home pose.")
                        set_manual_angles_action(get_home_angles())
                else:
                    print("Disabling all motor controls and resetting...")
                    if not body_controller.set_all_control_status(False):
                        print("Warning: Failed to disable all motor controls.")
                    time.sleep(0.1)
                    if not body_controller.reset_all(): print("Warning: Reset all after disable failed.")
                return

            # Actions below require motor control to be enabled
            if not motor_control_enabled:
                print(f"Motor control is disabled. Press 't' to enable. Key '{char}' ignored.")
                return

            if char == 's': # Stop walking
                if is_walking:
                    print("\n's' pressed. Stopping walk sequence...")
                    is_walking = False
                else:
                    print("\n's' pressed, but not currently walking.")
                return

            # Actions below are blocked if walking
            if is_walking:
                print(f"Walk sequence active. Key '{char}' ignored. Press 's' to stop walking first.")
                return

            if char == 'a': # Set home pose
                print("\nAttempting to set home pose ('a')...")
                set_manual_angles_action(get_home_angles())
            elif char == 'd': # Set zero pose
                print("\nAttempting to set zero pose ('d')...")
                set_manual_angles_action([0.0] * NUM_MOTORS)
            elif char == 'w': # Start walk sequence
                if walk_sequence:
                    is_walking = True # Set flag *before* starting thread
                    print("\n'w' pressed. Starting walk sequence in a new thread...")
                    walk_thread = threading.Thread(
                        target=execute_walk_sequence_action,
                        args=(walk_sequence, ACTUATOR_NAME_TO_INDEX_MAP),
                        daemon=True
                    )
                    walk_thread.start()
                else:
                    print("\nWalk sequence not loaded. Cannot start.")

    except Exception as e:
        print(f"Error in on_key_press: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    if not initial_setup_successful:
        print("Exiting due to failed initial setup.")
        sys.exit(1)

    print("\nQuadPilot Keyboard Control (Commands)")
    print("-------------------------------------")
    print(" T: Toggle motor enable/disable (resets & re-homes if enabling)")
    print(" A: Set motors to approximate home/standing pose")
    print(" D: Set motors to zero position")
    print(" W: Start walk sequence from walk.json")
    print(" S: Stop walk sequence (if running)")
    print(" Esc: Exit")
    print("-------------------------------------")

    print("Setting initial approximate home pose...")
    set_manual_angles_action(get_home_angles()) # Use the action function for debounce & checks

    listener = keyboard.Listener(on_press=on_key_press)
    try:
        listener.start()
        print("\nKeyboard listener started. Ready for commands.")
        listener.join() # Blocks until listener stops (on_key_press returns False)

    except KeyboardInterrupt:
        print("\nCtrl+C detected in main thread.")
    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
    finally:
        print("\nInitiating shutdown sequence...")
        is_walking = False # Ensure walk thread stops
        time.sleep(0.2) # Give walk thread a moment to notice the flag

        if listener.is_alive():
            print("Stopping keyboard listener...")
            listener.stop()
            listener.join(timeout=1.0)

        if body_controller:
            print("Performing final motor safety operations...")
            try:
                if not body_controller.set_all_control_status(False):
                    print("Warning: Failed to disable all motor controls during shutdown.")
                time.sleep(0.2)
                if not body_controller.reset_all():
                    print("Warning: Failed to reset all motors during shutdown.")
            except Exception as e:
                print(f"Error during motor safety operations: {e}")
            finally:
                print("Closing QuadPilotBody connection...")
                body_controller.close()
        
        print("Shutdown complete. Exiting.")