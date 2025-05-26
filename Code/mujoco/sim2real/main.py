import mujoco
import mujoco.viewer
import time
import os
import math
import numpy as np
import json # Import the JSON library

# --- Configuration ---
ACTUATOR_NAMES = [
    "FR_tigh_actuator", "FR_knee_actuator",
    "BR_tigh_actuator", "BR_knee_actuator",
    "FL_tigh_actuator", "FL_knee_actuator",
    "BL_tigh_actuator", "BL_knee_actuator",
]
JSON_OUTPUT_PATH = './walk.json' # Path for the output JSON file

# --- SIM-TO-REAL MAPPING CONFIGURATION ---
real_robot_home_deg_map = {
    "FR_tigh_actuator": -45.0,
    "FR_knee_actuator": 45.0,
    "BR_tigh_actuator": 45.0,
    "BR_knee_actuator": -45.0,
    "FL_tigh_actuator": 45.0,
    "FL_knee_actuator": 45.0,
    "BL_tigh_actuator": 45.0,
    "BL_knee_actuator": -45.0,
}
joint_scale_factors = {
    "FR_tigh_actuator": 1.0,
    "FR_knee_actuator": 1.0,
    "BR_tigh_actuator": 1.0,
    "BR_knee_actuator": 1.0,
    "FL_tigh_actuator": 1.0,
    "FL_knee_actuator": 1.0,
    "BL_tigh_actuator": 1.0,
    "BL_knee_actuator": 1.0,
}

# --- Conversion Function ---
def convert_sim_rad_to_real_deg(actuator_name, sim_rad_value, sim_home_rad_map, real_home_deg_map, scale_factors):
    """Converts simulation radian target to real robot degree command."""
    # Use .get for safer dictionary access with defaults
    sim_home_rad = sim_home_rad_map.get(actuator_name)
    real_home_deg = real_home_deg_map.get(actuator_name)
    scale = scale_factors.get(actuator_name, 1.0) # Default scale to 1 if missing

    if sim_home_rad is None or real_home_deg is None:
        # print(f"Warning: Missing mapping for '{actuator_name}' in conversion.")
        return None # Cannot convert if home poses are missing

    if not isinstance(sim_rad_value, (int, float)):
         # print(f"Warning: Invalid sim_rad_value type ({type(sim_rad_value)}) for '{actuator_name}'.")
         return None # Cannot convert non-numeric types

    sim_delta_rad = sim_rad_value - sim_home_rad
    real_delta_deg = scale * math.degrees(sim_delta_rad)
    real_target_deg = real_home_deg + real_delta_deg
    return real_target_deg


# --- Sequence Definition Function (NOW calculates and returns real targets too) ---
def create_control_sequence(sim_home_rad_map, real_home_deg_map, scale_factors, actuator_id_map, model):
    """
    Creates a sequence containing sim targets (rad), real targets (deg), and duration.
    """
    # --- Gait Parameters (User Provided Values) ---
    thigh_forward_delta = 0.10
    thigh_backward_delta = -0.10
    back_knee_lift_flex_delta = -0.35
    back_knee_stance_extend_delta = 0.2
    front_knee_lift_flex_delta = -0.50
    front_knee_stance_extend_delta = 0.15
    phase_duration = 0.40
    initial_hold = 1.0
    num_steps = 12

    def get_home(actuator_name):
        return sim_home_rad_map.get(actuator_name, 0.0)

    sequence = [] # Will store tuples: (sim_rad_dict, real_deg_dict, duration)

    # --- Initial Hold Step ---
    sim_initial_targets_rad = sim_home_rad_map.copy()
    real_initial_targets_deg = {}
    for name, sim_rad_val in sim_initial_targets_rad.items():
         real_deg = convert_sim_rad_to_real_deg(name, sim_rad_val, sim_home_rad_map, real_home_deg_map, scale_factors)
         if real_deg is not None:
             real_initial_targets_deg[name] = real_deg
    sequence.append((sim_initial_targets_rad, real_initial_targets_deg, initial_hold))


    # --- Shuffle Steps ---
    for step in range(num_steps):
        swing_pair_is_FR_BL = (step % 2 == 0)
        step_targets_rad = sim_home_rad_map.copy() # Calculate sim targets based on home

        # Calculate simulation radian targets for this step
        if swing_pair_is_FR_BL:
            step_targets_rad["FR_tigh_actuator"] = get_home("FR_tigh_actuator") + thigh_forward_delta
            step_targets_rad["FR_knee_actuator"] = get_home("FR_knee_actuator") + front_knee_lift_flex_delta
            step_targets_rad["BL_tigh_actuator"] = get_home("BL_tigh_actuator") + thigh_forward_delta
            step_targets_rad["BL_knee_actuator"] = get_home("BL_knee_actuator") + back_knee_lift_flex_delta
            step_targets_rad["FL_tigh_actuator"] = get_home("FL_tigh_actuator") + thigh_backward_delta
            step_targets_rad["FL_knee_actuator"] = get_home("FL_knee_actuator") + front_knee_stance_extend_delta
            step_targets_rad["BR_tigh_actuator"] = get_home("BR_tigh_actuator") + thigh_backward_delta
            step_targets_rad["BR_knee_actuator"] = get_home("BR_knee_actuator") + back_knee_stance_extend_delta
        else:
            step_targets_rad["FL_tigh_actuator"] = get_home("FL_tigh_actuator") + thigh_forward_delta
            step_targets_rad["FL_knee_actuator"] = get_home("FL_knee_actuator") + front_knee_lift_flex_delta
            step_targets_rad["BR_tigh_actuator"] = get_home("BR_tigh_actuator") + thigh_forward_delta
            step_targets_rad["BR_knee_actuator"] = get_home("BR_knee_actuator") + back_knee_lift_flex_delta
            step_targets_rad["FR_tigh_actuator"] = get_home("FR_tigh_actuator") + thigh_backward_delta
            step_targets_rad["FR_knee_actuator"] = get_home("FR_knee_actuator") + front_knee_stance_extend_delta
            step_targets_rad["BL_tigh_actuator"] = get_home("BL_tigh_actuator") + thigh_backward_delta
            step_targets_rad["BL_knee_actuator"] = get_home("BL_knee_actuator") + back_knee_stance_extend_delta

        # Clamp simulation radian targets
        clamped_targets_rad = {}
        for name, value in step_targets_rad.items():
             if name in actuator_id_map:
                 act_id = actuator_id_map[name]
                 if 0 <= act_id < model.nu:
                     ctrlrange = model.actuator_ctrlrange[act_id]
                     clamped_value = np.clip(value, ctrlrange[0], ctrlrange[1])
                     clamped_targets_rad[name] = clamped_value
                 else: clamped_targets_rad[name] = value # Use unclamped if ID invalid
             else: clamped_targets_rad[name] = value # Use unclamped if name not mapped

        # Convert clamped simulation radians to real robot degrees
        step_targets_real_deg = {}
        for name, sim_rad_val in clamped_targets_rad.items():
             real_deg = convert_sim_rad_to_real_deg(name, sim_rad_val, sim_home_rad_map, real_home_deg_map, scale_factors)
             if real_deg is not None:
                step_targets_real_deg[name] = real_deg

        # Append the tuple for this step
        sequence.append((clamped_targets_rad.copy(), step_targets_real_deg.copy(), phase_duration))

    # --- Return to Home Step ---
    sim_home_targets_rad = sim_home_rad_map.copy()
    real_home_targets_deg = {}
    for name, sim_rad_val in sim_home_targets_rad.items():
         real_deg = convert_sim_rad_to_real_deg(name, sim_rad_val, sim_home_rad_map, real_home_deg_map, scale_factors)
         if real_deg is not None:
             real_home_targets_deg[name] = real_deg
    sequence.append((sim_home_targets_rad, real_home_targets_deg, 1.0)) # Duration 1.0 sec


    print(f"Generated sequence with {len(sequence)} steps total.")
    return sequence


# --- Helper Function to Format Angles ---
def format_angle_output(name, rad_value, sim_home_map, real_home_map, scale_factors):
    """Formats actuator name and angle in Sim Rad, Sim Deg, and calculated Real Robot Deg."""
    if not isinstance(rad_value, (int, float)):
        return f"{name}=InvalidVal"
    sim_deg_value = math.degrees(rad_value)
    real_deg_value = convert_sim_rad_to_real_deg(name, rad_value, sim_home_map, real_home_map, scale_factors)
    real_deg_str = f"{real_deg_value:6.1f}d" if real_deg_value is not None else " N/A "
    padded_name = f"{name:<18}"
    return f"{padded_name}= {rad_value:6.3f}r ({sim_deg_value:6.1f}d) -> REAL= {real_deg_str}"

# --- Main Script Logic ---

# --- XML Path Handling ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
xml_file_name = 'walking_scene.xml'
xml_path = os.path.join(script_dir, '../our_robot', xml_file_name)
if not os.path.exists(xml_path):
    alt_xml_path = os.path.join(os.path.dirname(script_dir), 'our_robot', xml_file_name)
    if os.path.exists(alt_xml_path): xml_path = alt_xml_path
    else:
        cwd_xml_path = os.path.join(os.getcwd(), 'our_robot', xml_file_name)
        if os.path.exists(cwd_xml_path): xml_path = cwd_xml_path
        else: raise FileNotFoundError(f"Could not find '{xml_file_name}'. Checked paths relative to {script_dir} and {os.getcwd()}")
print(f"Loading model from: {xml_path}")

# --- MuJoCo Setup ---
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# --- Actuator ID Mapping ---
actuator_id_map = {}
print("Mapping Actuator Names to IDs:")
valid_actuator_names = []
for i, name in enumerate(ACTUATOR_NAMES):
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id == -1:
        print(f"Warning: Actuator '{name}' not found in the XML model. Skipping.")
        continue
    if actuator_id < 0 or actuator_id >= model.nu:
         print(f"Error: Actuator ID {actuator_id} for '{name}' is out of valid range [0, {model.nu-1}]. Check ACTUATOR_NAMES list.")
         continue
    actuator_id_map[name] = actuator_id
    valid_actuator_names.append(name)
    print(f"  '{name}' -> ID: {actuator_id}")

if not valid_actuator_names:
     raise ValueError("No valid actuators found based on ACTUATOR_NAMES in the model. Cannot proceed.")

# --- Keyframe Loading & SIMULATION Home Pose Map ---
key_name = 'home'
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
if key_id == -1: raise ValueError(f"Keyframe '{key_name}' not found.")
mujoco.mj_resetDataKeyframe(model, data, key_id)
print(f"\nLoaded keyframe '{key_name}'.")

sim_home_angles_rad_map = {}
initial_home_ctrl = data.ctrl[:model.nu].copy()
print("\nExtracting Simulation Home Angles (Radians):")
for name in valid_actuator_names:
    act_id = actuator_id_map[name]
    if 0 <= act_id < len(initial_home_ctrl):
        sim_home_angles_rad_map[name] = initial_home_ctrl[act_id]
        print(f"  {name}: {sim_home_angles_rad_map[name]:.3f}")
    else: print(f"Warning: Could not get sim home angle for {name}.")

if not sim_home_angles_rad_map: raise ValueError("Failed to extract sim home angles.")

# --- Verify Mapping Configuration ---
print("\nVerifying Real Robot Mapping Configuration:")
missing_real_home = [name for name in valid_actuator_names if name not in real_robot_home_deg_map]
missing_scale_factor = [name for name in valid_actuator_names if name not in joint_scale_factors]
if missing_real_home: raise ValueError(f"Real robot home map incomplete. Missing: {missing_real_home}")
if missing_scale_factor: raise ValueError(f"Joint scale factor map incomplete. Missing: {missing_scale_factor}")
print("Mapping configuration seems complete for valid actuators.")


# --- Create Sequence (Now returns sim_rad, real_deg, duration) ---
full_control_sequence = create_control_sequence(
    sim_home_angles_rad_map,
    real_robot_home_deg_map,
    joint_scale_factors,
    actuator_id_map,
    model
)

# --- Prepare Sequence for JSON Export ---
real_robot_sequence_for_json = []
for _, real_targets_deg, duration in full_control_sequence:
     # Round real degrees for cleaner JSON output (optional)
     rounded_real_targets = {name: round(deg, 2) for name, deg in real_targets_deg.items()}
     real_robot_sequence_for_json.append({
         "duration": round(duration, 3), # Round duration as well
         "targets_deg": rounded_real_targets
     })

# --- Save Real Robot Sequence to JSON ---
try:
    print(f"\nSaving real robot command sequence to {JSON_OUTPUT_PATH}...")
    with open(JSON_OUTPUT_PATH, 'w') as f:
        json.dump(real_robot_sequence_for_json, f, indent=2) # Use indent for readability
    print("Save successful.")
except IOError as e:
    print(f"ERROR: Could not write to JSON file '{JSON_OUTPUT_PATH}': {e}")
except Exception as e:
    print(f"An unexpected error occurred during JSON saving: {e}")


# --- Simulation Loop ---
current_sequence_index = 0
state_start_time = data.time

# Initialize target control array using simulation home values
current_target_ctrl = np.copy(initial_home_ctrl)

# Apply initial targets from sequence step 1 and print detailed info
if full_control_sequence:
    # Get the SIMULATION targets for the first step
    first_sim_target_dict, _, _ = full_control_sequence[0]
    print("\nApplying initial simulation targets from sequence step 1:")
    initial_target_strings = []
    for name in valid_actuator_names:
        act_id = actuator_id_map[name]
        target_rad = first_sim_target_dict.get(name, current_target_ctrl[act_id])
        current_target_ctrl[act_id] = target_rad # Update the sim control array
        initial_target_strings.append(format_angle_output(name, target_rad, sim_home_angles_rad_map, real_robot_home_deg_map, joint_scale_factors))
    print("  " + " | ".join(initial_target_strings[:len(initial_target_strings)//2]))
    print("  " + " | ".join(initial_target_strings[len(initial_target_strings)//2:]))
else:
    print("Warning: Control sequence is empty. Holding initial home pose.")
    home_target_strings = [format_angle_output(name, sim_home_angles_rad_map.get(name, 0.0), sim_home_angles_rad_map, real_robot_home_deg_map, joint_scale_factors)
                           for name in valid_actuator_names]
    print("  " + " | ".join(home_target_strings[:len(home_target_strings)//2]))
    print("  " + " | ".join(home_target_strings[len(home_target_strings)//2:]))


print(f"\nStarting simulation...")
# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
  wall_clock_start_time = time.time()

  while viewer.is_running():
    step_start_time = time.time()
    sim_time = data.time

    # --- Sequence Logic ---
    # Use the full_control_sequence which contains (sim_rad, real_deg, duration)
    if current_sequence_index < len(full_control_sequence):
        # Unpack the tuple for the current step
        current_sim_targets, _, duration = full_control_sequence[current_sequence_index]

        # Check if time to move to the next state
        if sim_time >= state_start_time + duration:
            current_sequence_index += 1
            if current_sequence_index < len(full_control_sequence):
                # Get the SIMULATION targets for the next step
                next_sim_target_dict, _, next_duration = full_control_sequence[current_sequence_index]
                print(f"\nSim Time: {sim_time:.3f}s. Transitioning to sequence step {current_sequence_index + 1} (duration: {next_duration:.2f}s)")

                # Update the target control array and prepare print strings
                target_strings = []
                for name in valid_actuator_names:
                     if name in actuator_id_map:
                         act_id = actuator_id_map[name]
                         # Get target from next *simulation* dict
                         target_rad = next_sim_target_dict.get(name, current_target_ctrl[act_id])
                         current_target_ctrl[act_id] = target_rad # Update sim control array
                         # Format output using the current sim target rad
                         target_strings.append(format_angle_output(name, target_rad, sim_home_angles_rad_map, real_robot_home_deg_map, joint_scale_factors))

                # Print the NEW target angles (Sim Rad, Sim Deg, Real Deg)
                print("  New Targets:")
                print("  " + " | ".join(target_strings[:len(target_strings)//2]))
                print("  " + " | ".join(target_strings[len(target_strings)//2:]))

                state_start_time = sim_time
            else:
                # End of defined sequence steps
                if full_control_sequence[-1][2] != float('inf'): # Check duration of last element
                     print(f"\nSim Time: {sim_time:.3f}s. Control sequence finished. Holding last pose.")
                     # Print the final pose targets using helper function
                     final_target_strings = [format_angle_output(name, current_target_ctrl[actuator_id_map[name]], sim_home_angles_rad_map, real_robot_home_deg_map, joint_scale_factors)
                                             for name in valid_actuator_names if name in actuator_id_map]
                     print("  Final Targets:")
                     print("  " + " | ".join(final_target_strings[:len(final_target_strings)//2]))
                     print("  " + " | ".join(final_target_strings[len(final_target_strings)//2:]))
                     # Append infinite duration step - copy last targets
                     last_sim_targets, last_real_targets, _ = full_control_sequence[-1]
                     full_control_sequence.append( (last_sim_targets, last_real_targets, float('inf')) )

    # Apply the simulation target controls
    if model.nu > 0:
        data.ctrl[:model.nu] = current_target_ctrl


    # --- Simulation Step ---
    try:
        mujoco.mj_step(model, data)
    except mujoco.FatalError as e:
        print(f"\nMuJoCo fatal error encountered: {e}")
        break
    except Exception as e:
        print(f"\nAn unexpected error occurred during simulation step: {e}")
        break


    # --- Viewer Sync and Time Control ---
    viewer.sync()
    time_spent_in_step = time.time() - step_start_time
    time_to_sleep = model.opt.timestep - time_spent_in_step
    if time_to_sleep > 0:
      time.sleep(time_to_sleep)


print("\nSimulation finished or viewer closed.")
wall_clock_duration = time.time() - wall_clock_start_time
print(f"Total wall clock time: {wall_clock_duration:.2f} seconds.")
print(f"Final simulation time: {data.time:.3f} seconds.")