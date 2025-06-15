import mujoco
import mujoco.viewer
import time
import os
import math
import numpy as np
import json
import requests
import sys
import threading

# --- Configuration ---
ESP32_IPS = [
    "192.168.137.100",  # Replace with your first ESP32 IP if different
    "192.168.137.101",  # Replace with your second ESP32 IP if different
]
ESP32_EVENT_ENDPOINT = "/events" # Endpoint on ESP32 providing angles

NUM_MOTORS = 8 # Total number of motors (4 per ESP32)

# --- EMPIRICALLY CORRECTED IMPORTANT MAPPING based on your latest symptom ---
# This maps (ESP32_index, motor_index_on_esp32) to the MuJoCo_actuator_name
# that it empirically seems to control in the simulation.
# Symptom: Real FL Tigh (assumed to be from index (0,3)) moves Sim FR Tigh (controlled by FR_tigh_actuator)
# Symptom: Real FR Tigh (assumed to be from index (0,1)) moves Sim FL Tigh (controlled by FL_tigh_actuator)
ESP32_MOTOR_INDEX_TO_ACTUATOR_NAME = {
    (0, 0): "FL_knee_actuator", # Assuming these remain correct
    (0, 1): "FL_tigh_actuator", # <-- Swap 1: Real FR Tigh (0,1) maps to Sim FL Tigh actuator name
    (0, 2): "FR_knee_actuator", # Assuming these remain correct
    (0, 3): "FR_tigh_actuator", # <-- Swap 2: Real FL Tigh (0,3) maps to Sim FR Tigh actuator name
    (1, 0): "BR_knee_actuator", # Assuming these remain correct
    (1, 1): "BR_tigh_actuator", # Assuming these remain correct
    (1, 2): "BL_knee_actuator", # Assuming these remain correct
    (1, 3): "BL_tigh_actuator", # Assuming these remain correct
}

# --- ANGLE OFFSETS ---
# These offsets in degrees will be ADDED to the SIGN-CORRECTED fetched angles
# before applying to simulation. Define the zero point for the sim relative to robot sensors.
ANGLE_OFFSETS_DEG = {
    "FR_tigh_actuator": 25.0,
    "FR_knee_actuator": 45.0,
    "FL_tigh_actuator": -25.0,
    "FL_knee_actuator": 45.0,
    "BR_tigh_actuator": 25.0,
    "BR_knee_actuator": -45.0,
    "BL_tigh_actuator": 25.0,
    "BL_knee_actuator": -45.0
}

# --- JOINT SIGN CORRECTION ---
# Multiply fetched angle by this factor (+1 or -1) before applying offset.
# Use -1 for joints whose sensor reading increases when the sim joint angle should decrease, or vice versa.
# Based on symptom, front thighs were inverted *after* mapping. Keep corrections from previous step.
# You will need to verify signs for ALL joints once mapping is correct.
JOINT_SIGN_CORRECTION = {
    # These multipliers apply *after* the mapping is done.
    # If FR_tigh_actuator (in sim) moves in wrong direction when Real FR Tigh is moved, flip this sign.
    # If FL_tigh_actuator (in sim) moves in wrong direction when Real FL Tigh is moved, flip this sign.
    "FR_tigh_actuator": -1.0, # Correct sign for Sim FR Tigh joint
    "FL_tigh_actuator": -1.0, # Correct sign for Sim FL Tigh joint
    # Add other actuators here with 1.0 if their direction is correct, or -1.0 if inverted
    "FR_knee_actuator": 1.0, # Assuming correct direction for knees (adjust if needed)
    "FL_knee_actuator": 1.0,
    "BR_tigh_actuator": 1.0,
    "BR_knee_actuator": 1.0,
    "BL_tigh_actuator": 1.0,
    "BL_knee_actuator": 1.0,
}


# --- Shared State and Control ---
latest_angles_deg = {}
stop_fetch_threads = threading.Event()

# --- Angle Fetching Thread Function ---
def fetch_angles_thread(esp32_index, esp32_ip, stop_event, shared_angles_dict):
    """
    Thread function to continuously fetch angles from a single ESP32 using streaming.
    """
    url = f"http://{esp32_ip}:82{ESP32_EVENT_ENDPOINT}"
    request_timeout = 3.0 # seconds for connection and reading the first byte

    print(f"[{esp32_ip}] Fetch thread started. Connecting to {url}.")

    while not stop_event.is_set():
        try:
            with requests.get(url, stream=True, timeout=request_timeout) as response:
                response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)

                for line in response.iter_lines():
                    if stop_event.is_set():
                        print(f"[{esp32_ip}] Stop signal received, closing stream.")
                        break # Exit the line iteration loop

                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data:"):
                            try:
                                data = json.loads(decoded_line[5:])
                                motor_angles_from_esp32 = data.get("angles", []) # Expects a list of 4 angles
                                # print(f"[{esp32_ip}] Parsed data, got angles: {motor_angles_from_esp32}") # --- IMPORTANT DEBUG PRINT ---

                                if motor_angles_from_esp32 and isinstance(motor_angles_from_esp32, list):
                                    temp_angles = {}
                                    # print(f"[{esp32_ip}] Attempting to map received angles...") # --- IMPORTANT DEBUG PRINT ---
                                    for j in range(min(len(motor_angles_from_esp32), 4)): # Process up to 4 angles
                                        esp32_motor_key = (esp32_index, j)
                                        angle_value = motor_angles_from_esp32[j]

                                        if esp32_motor_key in ESP32_MOTOR_INDEX_TO_ACTUATOR_NAME:
                                            actuator_name = ESP32_MOTOR_INDEX_TO_ACTUATOR_NAME[esp32_motor_key]
                                            try:
                                                angle = float(angle_value)
                                                temp_angles[actuator_name] = angle
                                                # print(f"[{esp32_ip}]   Mapped ({esp32_index},{j}) -> '{actuator_name}': {angle:.1f} deg") # --- IMPORTANT DEBUG PRINT ---
                                            except (ValueError, TypeError):
                                                print(f"[{esp32_ip}] Warning: Invalid angle data for {actuator_name} (index {j}): {angle_value} (type: {type(angle_value)})")
                                        else:
                                             print(f"[{esp32_ip}] Warning: ESP32 motor key {esp32_motor_key} not found in mapping.") # Should not happen if mapping is complete


                                    if temp_angles:
                                        shared_angles_dict.update(temp_angles)
                                        print(f"[{esp32_ip}] Fetched and applied {len(temp_angles)} angles. Latest total fetched in dict: {len(shared_angles_dict)}")


                                break # Exit the line iteration loop after processing data

                            except json.JSONDecodeError:
                                print(f"[{esp32_ip}] Error: Could not decode JSON from line: {decoded_line}")
                            except Exception as e:
                                print(f"[{esp32_ip}] Warning: Error processing data line: {decoded_line[:100]}... Exception: {e}")

        except requests.exceptions.Timeout:
            pass
        except requests.exceptions.ConnectionError as e:
            pass
        except requests.exceptions.RequestException as e:
            pass
        except Exception as e:
            print(f"[{esp32_ip}] An unexpected error occurred during fetch: {e}. Reconnecting...")

        if not stop_event.is_set():
             stop_event.wait(0.1)


    print(f"[{esp32_ip}] Fetch thread stopping.")

# --- Main Script Logic ---
if __name__ == "__main__":

    # --- XML Path Handling ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    xml_file_name = 'walking_scene.xml'
    potential_paths = [
        os.path.join(script_dir, '../our_robot', xml_file_name),
        os.path.join(os.path.dirname(script_dir), 'our_robot', xml_file_name), # One dir up from script_dir
        os.path.join(os.getcwd(), 'our_robot', xml_file_name), # Relative to cwd
        os.path.join(script_dir, xml_file_name), # In the same dir as script
    ]
    xml_path = None
    for path in potential_paths:
        if os.path.exists(path):
            xml_path = path
            break

    if xml_path is None:
        print(f"Error: Could not find '{xml_file_name}'. Searched in:\n- " + "\n- ".join(potential_paths))
        sys.exit(1)

    print(f"Loading model from: {xml_path}")

    # --- MuJoCo Setup ---
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        print("MuJoCo model loaded.")
    except Exception as e:
        print(f"FATAL: Failed to load MuJoCo model '{xml_path}': {e}")
        sys.exit(1)

    # --- Actuator ID Mapping ---
    actuator_name_to_mujoco_id = {}
    mujoco_actuator_names_in_model = []
    print("\nMapping Actuator Names to MuJoCo IDs (from model):")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name is not None:
            actuator_name_to_mujoco_id[name] = i
            mujoco_actuator_names_in_model.append(name)
            print(f"  '{name}' -> ID: {i}")

    if not mujoco_actuator_names_in_model:
         print("Error: No actuators found in the model.")
         sys.exit(1)

    # --- Verify Mappings ---
    print("\nVerifying mapping configuration...")
    mujoco_model_actuator_names_set = set(mujoco_actuator_names_in_model)
    mapped_mujoco_names_set = set(ESP32_MOTOR_INDEX_TO_ACTUATOR_NAME.values())

    missing_in_mujoco = mapped_mujoco_names_set - mujoco_model_actuator_names_set
    if missing_in_mujoco:
        print(f"Warning: Mapped actuator names not found in XML: {missing_in_mujoco}")

    missing_in_mapping = mujoco_model_actuator_names_set - mapped_mujoco_names_set
    if missing_in_mapping:
        print(f"Warning: Actuators in XML not found in ESP32 mapping: {missing_in_mapping}. These will not be controlled.")

    if len(mapped_mujoco_names_set) != NUM_MOTORS:
         print(f"Warning: Mapping defines {len(mapped_mujoco_names_set)} actuator names ({list(mapped_mujoco_names_set)}), expected {NUM_MOTORS}. Check ESP32_MOTOR_INDEX_TO_ACTUATOR_NAME.")
    else:
         print("Mapping configuration seems complete for all 8 motors.")

    offset_names_set = set(ANGLE_OFFSETS_DEG.keys())
    mapped_but_no_offset = mapped_mujoco_names_set - offset_names_set
    if mapped_but_no_offset:
        print(f"Warning: The following mapped actuators do not have an offset defined in ANGLE_OFFSETS_DEG and will use 0.0: {mapped_but_no_offset}")
    sign_correction_names_set = set(JOINT_SIGN_CORRECTION.keys())
    mapped_but_no_sign = mapped_mujoco_names_set - sign_correction_names_set
    if mapped_but_no_sign:
         print(f"Warning: The following mapped actuators do not have a sign correction defined in JOINT_SIGN_CORRECTION and will use +1.0: {mapped_but_no_sign}")

    unused_offsets = offset_names_set - mapped_mujoco_names_set
    if unused_offsets:
         print(f"Warning: Offsets defined for actuators not in the mapping: {unused_offsets}. These offsets will not be used.")


    # --- Initial Pose ---
    mujoco.mj_resetData(model, data)
    print("\nResetting data to default XML pose (usually zero).")


    # --- Start Fetch Threads ---
    fetch_threads = []
    print("\nStarting angle fetching threads...")
    for i, ip in enumerate(ESP32_IPS):
        thread = threading.Thread(
            target=fetch_angles_thread,
            args=(i, ip, stop_fetch_threads, latest_angles_deg),
            daemon=True
        )
        fetch_threads.append(thread)
        thread.start()

    time.sleep(0.5)
    print("Angle fetching threads started.")


    # --- Simulation Loop ---
    print("\nStarting simulation viewer. Press Ctrl+C to exit.")
    sim_loop_interval = 0.01 # seconds

    debug_print_interval = 1.0 # Print angles every 1 second
    last_debug_print_time = time.time()

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                loop_start_time = time.time()
                sim_time = data.time

                # --- Apply Latest Fetched Angles with Sign Correction and Offset to MuJoCo ---
                angles_to_apply_deg = latest_angles_deg.copy()

                if angles_to_apply_deg:
                    for actuator_name, angle_deg in angles_to_apply_deg.items():
                        if actuator_name in actuator_name_to_mujoco_id:
                            mujoco_id = actuator_name_to_mujoco_id[actuator_name]

                            sign_correction = JOINT_SIGN_CORRECTION.get(actuator_name, 1.0)
                            corrected_angle_deg = angle_deg * sign_correction

                            offset_deg = ANGLE_OFFSETS_DEG.get(actuator_name, 0.0)
                            offsetted_angle_deg = corrected_angle_deg + offset_deg

                            angle_rad = math.radians(offsetted_angle_deg)

                            data.ctrl[mujoco_id] = angle_rad

                # --- Debugging Print (Periodically) ---
                current_time = time.time()
                if current_time - last_debug_print_time > debug_print_interval:
                    print(f"\nSim Time: {sim_time:.3f}s | Robot Angles Debug:")
                    debug_strings = []
                    # Iterate through actuators from the mapping for a consistent order
                    for esp32_motor_key, mapped_name in ESP32_MOTOR_INDEX_TO_ACTUATOR_NAME.items():
                         if mapped_name in actuator_name_to_mujoco_id:
                             mujoco_id = actuator_name_to_mujoco_id[mapped_name]

                             fetched_deg = latest_angles_deg.get(mapped_name, float('nan'))
                             fetched_str = f"{fetched_deg:6.1f}d" if not math.isnan(fetched_deg) else "  N/A "

                             sign_corr = JOINT_SIGN_CORRECTION.get(mapped_name, 1.0)
                             sign_corrected_deg = fetched_deg * sign_corr if not math.isnan(fetched_deg) else float('nan')
                             sign_corrected_str = f"{sign_corrected_deg:6.1f}d" if not math.isnan(sign_corrected_deg) else "  N/A "


                             offset_val_deg = ANGLE_OFFSETS_DEG.get(mapped_name, 0.0)
                             expected_target_deg = sign_corrected_deg + offset_val_deg if not math.isnan(sign_corrected_deg) else float('nan')
                             expected_target_str = f"{expected_target_deg:6.1f}d" if not math.isnan(expected_target_deg) else "  N/A "


                             target_rad = data.ctrl[mujoco_id] if mujoco_id < model.nu else float('nan')
                             actual_target_deg_from_ctrl = math.degrees(target_rad) if not math.isnan(target_rad) else float('nan')

                             joint_id = model.actuator_trnid[mujoco_id, 0]
                             actual_pos_rad = data.qpos[joint_id] if joint_id != -1 and joint_id < model.nq else float('nan')
                             actual_pos_deg = math.degrees(actual_pos_rad) if not math.isnan(actual_pos_rad) else float('nan')


                             debug_strings.append(f"{mapped_name:<18} (ESP:{esp32_motor_key[0]},Idx:{esp32_motor_key[1]}): Fetched={fetched_str} | SignCorr={sign_corrected_str} | OffsettedTarget={expected_target_str} (Ctrl={actual_target_deg_from_ctrl:6.1f}d) | Qpos={actual_pos_deg:6.1f}d")

                    half = len(debug_strings) // 2
                    if half > 0:
                        print("  " + " | ".join(debug_strings[:half]))
                        print("  " + " | ".join(debug_strings[half:]))
                    elif debug_strings:
                         print("  " + " | ".join(debug_strings))
                    else:
                         print("  (No actuators found or mapped for debug print)")

                    last_debug_print_time = current_time


                # --- Simulation Step ---
                try:
                    mujoco.mj_step(model, data)
                except mujoco.FatalError as e:
                    print(f"\nMuJoCo fatal error encountered during step: {e}")
                    viewer.close()
                    break
                except Exception as e:
                    print(f"\nAn unexpected error occurred during simulation step: {e}")
                    viewer.close()
                    break

                # --- Viewer Sync and Time Control ---
                viewer.sync()

                elapsed_time = time.time() - loop_start_time
                sleep_time = sim_loop_interval - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nCtrl+C detected.")
    except Exception as e:
        print(f"\nAn unhandled error occurred during the simulation loop: {e}")
    finally:
        print("Signaling fetch threads to stop...")
        stop_fetch_threads.set()

        print("Waiting for fetch threads to finish...")
        for i, thread in enumerate(fetch_threads):
            thread.join(timeout=1.0)
            if thread.is_alive():
                print(f"Warning: Thread for {ESP32_IPS[i]} did not stop cleanly.")
            else:
                print(f"Thread for {ESP32_IPS[i]} stopped.")

        print("Simulation finished or viewer closed.")
        print("Cleanup complete. Exiting.")