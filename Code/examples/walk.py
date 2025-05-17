import os
import sys
import time
import threading
import json # Import json library
from pynput import keyboard
# Assuming QuadPilotBody is in the parent directory relative to this script's dir
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

# Attempt to import QuadPilotBody AFTER adjusting sys.path
try:
    from quadpilot import QuadPilotBody
except ImportError as e:
    print(f"Error importing QuadPilotBody: {e}")
    print("Please ensure 'quadpilot.py' is in the correct directory (expected: parent of script directory).")
    sys.exit(1)


# --- Configuration ---
JSON_WALK_FILE = os.path.join(script_dir, 'walk.json') # Path to the walk sequence
NUM_MOTORS = 8
TARGET_MOTORS = list(range(NUM_MOTORS)) # All motors

# --- Mapping: JSON Actuator Name -> QuadPilotBody Index ---
# **IMPORTANT**: Verify this mapping matches how your simulation generated
# the JSON keys and how your QuadPilotBody orders motors for set_angles!
# Based on your 'a' key example code for home pose:
#   angles[3] = FL Hip, angles[0] = FL Knee
#   angles[1] = FR Hip, angles[2] = FR Knee
#   angles[5] = BR Hip, angles[4] = BR Knee
#   angles[7] = BL Hip, angles[6] = BL Knee
# Mapping this to typical simulation actuator names:
ACTUATOR_NAME_TO_INDEX_MAP = {
    "FL_tigh_actuator": 3,  # Front Left Hip/Thigh
    "FL_knee_actuator": 0,  # Front Left Knee
    "FR_tigh_actuator": 1,  # Front Right Hip/Thigh
    "FR_knee_actuator": 2,  # Front Right Knee
    "BR_tigh_actuator": 5,  # Back Right Hip/Thigh
    "BR_knee_actuator": 4,  # Back Right Knee
    "BL_tigh_actuator": 7,  # Back Left Hip/Thigh
    "BL_knee_actuator": 6,  # Back Left Knee
}
# Check if all 8 motors are mapped
if len(ACTUATOR_NAME_TO_INDEX_MAP) != NUM_MOTORS:
    print("Warning: ACTUATOR_NAME_TO_INDEX_MAP does not contain exactly 8 entries!")

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


# --- Initialize QuadPilotBody ---
try:
    body = QuadPilotBody()
    print("QuadPilotBody initialized.")
except Exception as e:
    print(f"FATAL: Failed to initialize QuadPilotBody: {e}")
    sys.exit(1)

# --- State Variables ---
motor_control_enabled = True
is_walking = False # Flag to indicate if walk sequence is running
last_key_press_time = 0
debounce_interval = 0.2 # For manual keys 'a' and 'd'

# --- Functions ---

# Function to set angle for target motors (for manual 'a'/'d')
def set_manual_angles(angles):
    global last_key_press_time, is_walking
    if is_walking:
        print("Cannot set manual angles while walking sequence is active.")
        return
    current_time = time.time()
    if current_time - last_key_press_time > debounce_interval:
        last_key_press_time = current_time
        print(f"\nSetting manual angles: {angles}")
        try:
            body.set_angles(angles)
        except Exception as e:
            print(f"Failed to set manual angle: {e}")

# Function to execute the walk sequence (runs in a separate thread)
def execute_walk_sequence(sequence, name_to_index_map):
    global is_walking, motor_control_enabled
    if not motor_control_enabled:
        print("Motor control disabled, cannot start walk sequence.")
        is_walking = False # Ensure flag is reset if called incorrectly
        return

    print("\n--- Starting Walk Sequence ---")
    last_angles = [0] * NUM_MOTORS # Store last sent angles

    try:
        for i, step in enumerate(sequence):
            if not motor_control_enabled or not is_walking: # Check flags each step
                print("Walk sequence interrupted (motor control disabled or flag reset).")
                break

            duration = step.get("duration", 0.1) # Default duration if missing
            targets_deg = step.get("targets_deg", {})

            # Prepare the angle list for this step
            current_step_angles = last_angles[:] # Start with previous angles

            for name, target_angle in targets_deg.items():
                if name in name_to_index_map:
                    index = name_to_index_map[name]
                    if 0 <= index < NUM_MOTORS:
                        current_step_angles[index] = int(round(target_angle)) # Use rounded int angle
                    else:
                        print(f"Warning: Index {index} for '{name}' out of bounds.")
                else:
                    print(f"Warning: Actuator name '{name}' from JSON not found in map.")

            # Send angles directly, bypassing manual debounce
            # print(f"Step {i+1}/{len(sequence)}: Setting {current_step_angles}, Duration: {duration:.2f}s") # Optional: Verbose step log
            body.set_angles(current_step_angles)
            last_angles = current_step_angles # Update last sent angles

            # Wait for the specified duration
            time.sleep(duration)

        print("--- Walk Sequence Finished ---")

    except Exception as e:
        print(f"Error during walk sequence execution: {e}")
    finally:
        # VERY IMPORTANT: Reset the flag when the sequence thread finishes or errors out
        is_walking = False
        print("is_walking flag set to False.")


# Function to handle key presses
def on_press(key):
    global motor_control_enabled, is_walking
    try:
        char = key.char
    except AttributeError:
        # Special keys (like shift, ctrl, etc.) ignored
        return

    if char == 't':
        if is_walking:
            print("\nCannot toggle motor control while walking. Stop sequence first (not implemented) or restart.")
            return
        motor_control_enabled = not motor_control_enabled
        print(f"\nMotor control {'enabled' if motor_control_enabled else 'disabled'}")
        try:
            # Reset motors before changing control status bulk
            body.reset_all()
            time.sleep(0.5) # Allow time for reset
            for motor_idx in TARGET_MOTORS:
                body.set_control_status(motor=motor_idx, status=motor_control_enabled)
            print(f"Control status set to {motor_control_enabled} for motors {TARGET_MOTORS}")
            if motor_control_enabled:
                 # Optionally set to home pose after enabling
                 print("Setting to home pose after enabling control.")
                 home_angles = [0] * NUM_MOTORS
                 home_angles[3] = -45; home_angles[0] = 45  # FL
                 home_angles[1] = 45;  home_angles[2] = 45  # FR
                 home_angles[5] = 45;  home_angles[4] = -45 # BR
                 home_angles[7] = 45;  home_angles[6] = -45 # BL
                 set_manual_angles(home_angles) # Use manual func with debounce here

        except Exception as e:
            print(f"Failed to toggle control status: {e}")

    # --- Process other keys only if motor control is enabled ---
    elif motor_control_enabled:
        if char == 'a':
            if is_walking:
                print("\nCannot set home pose while walking sequence is active.")
                return
            angles = [0] * NUM_MOTORS
            angles[3] = -45; angles[0] = 45  # FL
            angles[1] = 45;  angles[2] = 45  # FR
            angles[5] = 45;  angles[4] = -45 # BR
            angles[7] = 45;  angles[6] = -45 # BL
            set_manual_angles(angles) # Use manual func with debounce

        elif char == 'd':
            if is_walking:
                print("\nCannot set zero pose while walking sequence is active.")
                return
            angles = [0] * NUM_MOTORS
            set_manual_angles(angles) # Use manual func with debounce

        elif char == 'w':
            if walk_sequence is None:
                print("\nWalk sequence not loaded. Cannot start.")
                return
            if is_walking:
                print("\nWalk sequence already running.")
                return
            # --- Start walk sequence in a new thread ---
            is_walking = True # Set flag immediately
            print("\nStarting walk sequence thread...")
            walk_thread = threading.Thread(
                target=execute_walk_sequence,
                args=(walk_sequence, ACTUATOR_NAME_TO_INDEX_MAP),
                daemon=True # Allows main program to exit even if thread is running
            )
            walk_thread.start()
            # --------------------------------------------

        elif char == 's':
            # 's' key currently has no function related to the sequence
            # If you want 's' to stop the sequence, you would set is_walking = False
            # Note: Stopping cleanly might require more sophisticated thread communication
            if is_walking:
                 print("\nStopping walk sequence...")
                 is_walking = False # Signal thread to stop
            else:
                 print("\n's' key pressed, but not walking.")
            # The sequence thread will check the flag and exit on its next iteration


# --- Main ---
if __name__ == "__main__":
    print("\nQuadPilot Keyboard Control")
    print("--------------------------")
    print(" T: Toggle motor enable/disable (resets motors)")
    print(" A: Set motors to approximate home/standing pose")
    print(" D: Set motors to zero position")
    print(" W: Start walk sequence from walk.json")
    print(" S: Stop walk sequence (if running)")
    print(" Ctrl+C: Exit")
    print("--------------------------")

    # Enable control for target motors initially (optional, depends on desired start state)
    if motor_control_enabled:
        try:
            print("Enabling control for target motors on startup...")
            body.reset_all()
            time.sleep(0.5)
            for motor_idx in TARGET_MOTORS:
                body.set_control_status(motor=motor_idx, status=True)
            print("Done.")
            # Set home pose on startup
            print("Setting initial home pose...")
            home_angles = [0] * NUM_MOTORS
            home_angles[3] = -45; home_angles[0] = 45  # FL
            home_angles[1] = 45;  home_angles[2] = 45  # FR
            home_angles[5] = 45;  home_angles[4] = -45 # BR
            home_angles[7] = 45;  home_angles[6] = -45 # BL
            set_manual_angles(home_angles)
        except Exception as e:
            print(f"Failed during initial motor setup: {e}")


    # Start the keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("\nKeyboard listener started. Ready for commands.")

    # Keep the main thread alive
    try:
        # You could join the listener thread, but sleeping allows
        # KeyboardInterrupt to be caught more reliably here.
        while listener.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Stopping...")
    finally:
        print("Stopping keyboard listener...")
        listener.stop()
        listener.join() # Wait for listener thread to finish

        # Ensure motors are disabled on exit
        print("Disabling motor control...")
        is_walking = False # Make sure sequence thread exits if running
        time.sleep(0.1) # Give thread a moment to potentially react
        try:
            for motor_idx in TARGET_MOTORS:
                # Attempt to disable even if it fails for some
                try:
                    body.set_control_status(motor=motor_idx, status=False)
                except Exception as inner_e:
                    print(f"Minor error disabling motor {motor_idx}: {inner_e}")
            time.sleep(0.3) # Allow time for commands
        except Exception as e:
            print(f"Failed to disable control during cleanup: {e}")
        print("Cleanup complete. Exiting.")