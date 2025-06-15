import mujoco
import mujoco.viewer
import time
import os
import math
import numpy as np
import json
import sys # For exiting on error

# --- Constants and Configuration ---
# (Keep these easily accessible at the top or load from a config file)
ACTUATOR_NAMES = [
    "FR_tigh_actuator", "FR_knee_actuator",
    "BR_tigh_actuator", "BR_knee_actuator",
    "FL_tigh_actuator", "FL_knee_actuator",
    "BL_tigh_actuator", "BL_knee_actuator",
]
JSON_INPUT_PATH = './walk.json' # Path for the INPUT JSON sequence file
XML_FILENAME = 'walking_scene.xml' # Name of the MuJoCo model file
XML_SEARCH_DIRS = ['../our_robot', './our_robot'] # Relative paths to search
KEYFRAME_NAME = 'home' # Name of the keyframe defining sim home pose

# SIM-TO-REAL MAPPING CONFIGURATION
# (Ensure these accurately reflect your simulation and real robot)
REAL_ROBOT_HOME_DEG_MAP = {
    "FR_tigh_actuator": -45.0, "FR_knee_actuator": 45.0,
    "BR_tigh_actuator": 45.0, "BR_knee_actuator": -45.0,
    "FL_tigh_actuator": 45.0, "FL_knee_actuator": 45.0,
    "BL_tigh_actuator": 45.0, "BL_knee_actuator": -45.0,
}
JOINT_SCALE_FACTORS = {
    "FR_tigh_actuator": 1.0, "FR_knee_actuator": 1.0,
    "BR_tigh_actuator": -1.0, "BR_knee_actuator": -1.0,
    "FL_tigh_actuator": -1.0, "FL_knee_actuator": 1.0,
    "BL_tigh_actuator": -1.0, "BL_knee_actuator": -1.0,
}

# --- Utility Functions ---

def find_xml_path(filename, search_dirs):
    """Searches for the XML file in specified relative directories."""
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    paths_tried = []
    for rel_dir in search_dirs:
        path = os.path.abspath(os.path.join(script_dir, rel_dir, filename))
        paths_tried.append(path)
        if os.path.exists(path):
            print(f"Found model XML at: {path}")
            return path

    # Fallback: Check relative to current working directory
    cwd_path = os.path.join(os.getcwd(), filename)
    paths_tried.append(cwd_path)
    if os.path.exists(cwd_path):
         print(f"Found model XML at: {cwd_path} (relative to cwd)")
         return cwd_path

    raise FileNotFoundError(f"Could not find '{filename}'. Searched paths: {paths_tried}")

def convert_real_deg_to_sim_rad(actuator_name, real_deg_value, sim_home_rad_map, real_home_deg_map, scale_factors):
    """Converts real robot degree target to simulation radian command."""
    sim_home_rad = sim_home_rad_map.get(actuator_name)
    real_home_deg = real_home_deg_map.get(actuator_name)
    scale = scale_factors.get(actuator_name)

    if sim_home_rad is None or real_home_deg is None or scale is None:
        # print(f"Debug: Missing mapping for '{actuator_name}' in real->sim conversion.")
        return None

    if not isinstance(real_deg_value, (int, float)):
         # print(f"Debug: Invalid real_deg_value type ({type(real_deg_value)}) for '{actuator_name}'.")
         return None

    if scale == 0: return None # Avoid division by zero

    real_delta_deg = real_deg_value - real_home_deg
    sim_delta_rad = math.radians(real_delta_deg) / scale
    sim_target_rad = sim_home_rad + sim_delta_rad
    return sim_target_rad

def format_angle_output(name, target_real_deg, target_sim_rad):
    """Formats actuator name, target Real Deg, and calculated Sim Rad for printing."""
    padded_name = f"{name:<18}"
    real_deg_str = f"{target_real_deg:6.1f}d" if target_real_deg is not None else " N/A "
    sim_rad_str = f"{target_sim_rad:6.3f}r" if target_sim_rad is not None else " N/A "
    sim_deg_str = f"({math.degrees(target_sim_rad):6.1f}d)" if target_sim_rad is not None else ""
    return f"{padded_name}= REAL_TGT={real_deg_str} -> SIM_CMD={sim_rad_str} {sim_deg_str}"

# --- Core Logic Functions ---

def initialize_mujoco(xml_path):
    """Loads the MuJoCo model and creates the data object."""
    print(f"Loading MuJoCo model from: {xml_path}")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        print("MuJoCo model and data initialized successfully.")
        return model, data
    except Exception as e:
        print(f"ERROR: Failed to load MuJoCo model from {xml_path}: {e}", file=sys.stderr)
        sys.exit(1) # Exit if model fails to load

def map_actuators(model, actuator_names_list):
    """Maps actuator names to IDs and validates them."""
    print("\nMapping Actuator Names to IDs...")
    actuator_id_map = {}
    valid_actuator_names = []
    for name in actuator_names_list:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id == -1:
            print(f"Warning: Actuator '{name}' not found in the model. Skipping.")
            continue
        if not (0 <= actuator_id < model.nu):
            print(f"ERROR: Actuator ID {actuator_id} for '{name}' is out of range [0, {model.nu-1}]. Check model or ACTUATOR_NAMES.", file=sys.stderr)
            continue # Skip invalid IDs but maybe don't exit immediately? Or sys.exit(1)
        actuator_id_map[name] = actuator_id
        valid_actuator_names.append(name)
        print(f"  '{name}' -> ID: {actuator_id}")

    if not valid_actuator_names:
        print("ERROR: No valid actuators found based on ACTUATOR_NAMES in the model. Cannot proceed.", file=sys.stderr)
        sys.exit(1)
    print(f"Successfully mapped {len(valid_actuator_names)} valid actuators.")
    return actuator_id_map, valid_actuator_names

def determine_sim_home_pose(model, data, key_name, actuator_id_map, valid_act_names):
    """Sets initial pose from keyframe and extracts sim home angles (rad)."""
    print(f"\nDetermining Simulation Home Pose using keyframe '{key_name}'...")
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    sim_home_rad_map = {}
    initial_ctrl = np.zeros(model.nu) # Default to zeros

    if key_id == -1:
        print(f"Warning: Keyframe '{key_name}' not found. Using default initial control values.")
        # If no keyframe, initial_ctrl might be zeros or defaults from XML
        initial_ctrl = data.ctrl[:model.nu].copy() # Use whatever is in data initially
    else:
        try:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
            initial_ctrl = data.ctrl[:model.nu].copy() # Get ctrl values AFTER resetting to keyframe
            print(f"Loaded keyframe '{key_name}' successfully.")
        except Exception as e:
            print(f"Warning: Failed to reset to keyframe '{key_name}': {e}. Using current data.ctrl.")
            initial_ctrl = data.ctrl[:model.nu].copy()

    print("Extracting Simulation Home Angles (Radians) from initial control state:")
    for name in valid_act_names:
        act_id = actuator_id_map[name]
        sim_home_rad_map[name] = initial_ctrl[act_id]
        print(f"  {name}: {sim_home_rad_map[name]:.4f}")

    if not sim_home_rad_map:
        print("ERROR: Failed to determine simulation home angles. Cannot proceed.", file=sys.stderr)
        sys.exit(1)

    return sim_home_rad_map, initial_ctrl

def verify_mappings(valid_act_names, real_home_map, scale_map):
    """Checks if the real robot and scale factor maps cover all valid actuators."""
    print("\nVerifying Real Robot Mapping Configuration...")
    missing_real_home = [name for name in valid_act_names if name not in real_home_map]
    missing_scale = [name for name in valid_act_names if name not in scale_map]
    has_error = False
    if missing_real_home:
        print(f"ERROR: Real robot home map (REAL_ROBOT_HOME_DEG_MAP) is incomplete. Missing: {missing_real_home}", file=sys.stderr)
        has_error = True
    if missing_scale:
        print(f"ERROR: Joint scale factor map (JOINT_SCALE_FACTORS) is incomplete. Missing: {missing_scale}", file=sys.stderr)
        has_error = True

    if has_error:
        sys.exit(1)
    else:
        print("Mapping configuration covers all valid actuators.")

def load_and_process_sequence(json_path, sim_home_map, real_home_map, scale_map, act_id_map, valid_acts, model):
    """Loads sequence from JSON, converts degrees to radians, clamps, and returns it."""
    print(f"\nLoading and Processing Sequence from: {json_path}")
    try:
        with open(json_path, 'r') as f:
            raw_sequence = json.load(f)
        print(f"Loaded {len(raw_sequence)} steps from JSON.")
    except FileNotFoundError:
        print(f"ERROR: JSON input file not found at '{json_path}'", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON format in '{json_path}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error loading JSON '{json_path}': {e}", file=sys.stderr)
        sys.exit(1)

    processed_sequence = [] # Stores tuples: (sim_rad_dict, duration, original_real_deg_dict)
    print("Converting real degrees to simulation radians and clamping...")
    for i, step_data in enumerate(raw_sequence):
        if not isinstance(step_data, dict) or 'duration' not in step_data or 'targets_deg' not in step_data:
            print(f"Warning: Skipping invalid step format at index {i} in JSON.")
            continue

        duration = float(step_data['duration'])
        real_targets_deg = step_data['targets_deg']
        sim_targets_rad_clamped = {}
        processed_any_target = False

        for name, real_deg_val in real_targets_deg.items():
            if name not in valid_acts:
                # print(f"Debug: Actuator '{name}' from JSON step {i} not active in model. Skipping.")
                continue

            sim_rad_val = convert_real_deg_to_sim_rad(name, real_deg_val, sim_home_map, real_home_map, scale_map)

            if sim_rad_val is not None:
                act_id = act_id_map[name]
                ctrlrange = model.actuator_ctrlrange[act_id]
                clamped_val = np.clip(sim_rad_val, ctrlrange[0], ctrlrange[1])

                if abs(clamped_val - sim_rad_val) > 1e-4:
                    print(f"  Info: Step {i}, Actuator '{name}': Sim target {sim_rad_val:.3f}r clamped to {clamped_val:.3f}r (Range: [{ctrlrange[0]:.3f}, {ctrlrange[1]:.3f}])")

                sim_targets_rad_clamped[name] = clamped_val
                processed_any_target = True
            else:
                 print(f"Warning: Could not convert target for '{name}' (val: {real_deg_val}) in JSON step {i}. Check mappings or value.")

        if not processed_any_target and len(real_targets_deg)>0 :
            print(f"Warning: Step {i} resulted in no valid simulation targets after conversion/filtering. Skipping step.")
            continue
        elif not processed_any_target and len(real_targets_deg)==0:
             print(f"Info: Step {i} has no targets defined. Will hold previous pose for duration {duration}s.")
             # We still add it so the duration is respected, but with empty targets.
             # The simulation loop needs to handle this by not updating unspecified actuators.

        processed_sequence.append((sim_targets_rad_clamped, duration, real_targets_deg))

    if not processed_sequence:
        print("ERROR: No valid control steps could be processed from the JSON file.", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully processed {len(processed_sequence)} steps for simulation.")
    return processed_sequence


def run_simulation(model, data, sequence, initial_ctrl, act_id_map, valid_acts, sim_home_map, real_home_map, scale_map):
    """Executes the simulation loop using the processed sequence."""
    print(f"\nStarting Simulation...")
    current_sequence_index = 0
    state_start_time = data.time # Use simulation time for state transitions
    current_target_ctrl = initial_ctrl.copy() # Start with initial/home controls

    # Apply initial targets from the first sequence step
    if sequence:
        first_sim_targets, _, first_real_targets = sequence[0]
        print("Applying initial simulation targets from sequence step 1:")
        initial_target_strings = []
        for name in valid_acts:
            act_id = act_id_map[name]
            # Update target if specified in the first step, otherwise keep initial value
            if name in first_sim_targets:
                target_sim_rad = first_sim_targets[name]
                current_target_ctrl[act_id] = target_sim_rad
            else:
                 target_sim_rad = current_target_ctrl[act_id] # Keep the initial one

            target_real_deg = first_real_targets.get(name) # Get original degree for printing
            initial_target_strings.append(format_angle_output(name, target_real_deg, target_sim_rad))

        half_idx = (len(initial_target_strings) + 1) // 2
        print("  " + " | ".join(initial_target_strings[:half_idx]))
        if half_idx < len(initial_target_strings):
            print("  " + " | ".join(initial_target_strings[half_idx:]))
    else:
        print("Warning: Sequence is empty. Holding initial pose.")
        # Code to print initial pose could go here if desired

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            wall_clock_start_time = time.time()

            while viewer.is_running():
                step_start_time = time.time()
                sim_time = data.time

                # --- Sequence Advancement Logic ---
                if current_sequence_index < len(sequence):
                    _, current_duration, _ = sequence[current_sequence_index]

                    # Check if it's time to move to the next state
                    if sim_time >= state_start_time + current_duration:
                        current_sequence_index += 1
                        if current_sequence_index < len(sequence):
                            next_sim_targets, next_duration, next_real_targets = sequence[current_sequence_index]
                            print(f"\nSim Time: {sim_time:.3f}s. Transitioning to sequence step {current_sequence_index + 1} (duration: {next_duration:.2f}s)")

                            target_strings = []
                            # Update controls based on the *next* step's targets
                            for name in valid_acts:
                                act_id = act_id_map[name]
                                # IMPORTANT: Only update if the target exists in the next step's dictionary
                                if name in next_sim_targets:
                                    target_sim_rad = next_sim_targets[name]
                                    current_target_ctrl[act_id] = target_sim_rad # Apply new target
                                else:
                                    # If actuator not mentioned, keep its *current* target value
                                    target_sim_rad = current_target_ctrl[act_id]

                                target_real_deg = next_real_targets.get(name) # Get original degree for printing
                                target_strings.append(format_angle_output(name, target_real_deg, target_sim_rad))

                            print("  New Targets:")
                            half_idx = (len(target_strings) + 1) // 2
                            print("  " + " | ".join(target_strings[:half_idx]))
                            if half_idx < len(target_strings):
                                print("  " + " | ".join(target_strings[half_idx:]))

                            state_start_time = sim_time # Reset timer for the new state

                        else:
                            # Reached the end of the sequence
                            print(f"\nSim Time: {sim_time:.3f}s. Sequence finished. Holding final pose.")
                            # Add an infinite hold state implicitly by not advancing index anymore
                            # Ensure the last command persists: make the last step infinite duration conceptually
                            last_sim_targets, _, last_real_targets = sequence[-1]
                            sequence.append((last_sim_targets, float('inf'), last_real_targets)) # Append hold state

                # --- Apply Controls ---
                if model.nu > 0:
                    data.ctrl[:model.nu] = current_target_ctrl

                # --- Simulation Step ---
                mujoco.mj_step(model, data)

                # --- Viewer Sync & Real-time Pace ---
                viewer.sync()
                time_spent = time.time() - step_start_time
                sleep_time = model.opt.timestep - time_spent
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except mujoco.FatalError as e:
        print(f"\nMuJoCo Fatal Error: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during simulation: {e}", file=sys.stderr)
    finally:
        print("\nSimulation finished or viewer closed.")
        if 'wall_clock_start_time' in locals():
             wall_clock_duration = time.time() - wall_clock_start_time
             print(f"Total wall clock time: {wall_clock_duration:.2f} seconds.")
        if data: # Check if data exists
             print(f"Final simulation time: {data.time:.3f} seconds.")


# --- Main Execution ---
def main():
    """Main function to orchestrate the setup and simulation."""
    try:
        # 1. Setup: Find XML, Initialize MuJoCo
        xml_path = find_xml_path(XML_FILENAME, XML_SEARCH_DIRS)
        model, data = initialize_mujoco(xml_path)

        # 2. Mapping: Identify actuators, determine home pose, verify config
        actuator_id_map, valid_actuator_names = map_actuators(model, ACTUATOR_NAMES)
        sim_home_angles_rad_map, initial_home_ctrl = determine_sim_home_pose(
            model, data, KEYFRAME_NAME, actuator_id_map, valid_actuator_names
        )
        verify_mappings(valid_actuator_names, REAL_ROBOT_HOME_DEG_MAP, JOINT_SCALE_FACTORS)

        # 3. Sequence Processing: Load JSON, convert degrees to radians, clamp
        processed_sequence = load_and_process_sequence(
            JSON_INPUT_PATH,
            sim_home_angles_rad_map,
            REAL_ROBOT_HOME_DEG_MAP,
            JOINT_SCALE_FACTORS,
            actuator_id_map,
            valid_actuator_names,
            model
        )

        # 4. Simulation: Run the main loop
        run_simulation(
            model,
            data,
            processed_sequence,
            initial_home_ctrl,
            actuator_id_map,
            valid_actuator_names,
            sim_home_angles_rad_map, # Pass maps needed for printing inside loop
            REAL_ROBOT_HOME_DEG_MAP,
            JOINT_SCALE_FACTORS
        )

    except FileNotFoundError as e:
         print(f"Setup Error: {e}", file=sys.stderr)
         sys.exit(1)
    except ValueError as e:
         print(f"Configuration or Data Error: {e}", file=sys.stderr)
         sys.exit(1)
    except RuntimeError as e:
         print(f"Runtime Error: {e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
         print(f"An unexpected error occurred in main: {e}", file=sys.stderr)
         import traceback
         traceback.print_exc() # Print full traceback for unexpected errors
         sys.exit(1)


if __name__ == "__main__":
    main()