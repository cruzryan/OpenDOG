# Save this as: scripts/calibrate_br_hip.py

import os
import sys
import time
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir) + "\quadpilot" # This is the parent directory
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

try:
    # Assuming quadpilot.py is in the 'code_dir' (parent directory)
    from body import QuadPilotBody
except ImportError:
    print(f"ERROR: Could not import QuadPilotBody from 'quadpilot.py'. "
          f"Ensure 'quadpilot.py' is in the directory: {code_dir}")
    sys.exit(1)

# --- Motor & Calibration Configurations ---
MOTOR_CONFIGS = [
    (0, "Front Left (Knee)", 39, 40, 41, 42), (1, "Front Right (Hip)", 37, 38, 1, 2),
    (2, "Front Right (Knee)", 17, 18, 5, 4),  (3, "Front Left (Hip)", 16, 15, 7, 6),
    (4, "Back Right (Knee)", 37, 38, 1, 2),   (5, "Back Right (Hip)", 40, 39, 42, 41),
    (6, "Back Left (Knee)", 15, 16, 6, 7),    (7, "Back Left (Hip)", 18, 17, 4, 5),
]
pin_configs = [(enc_a, enc_b, in1, in2) for _, _, enc_a, enc_b, in1, in2 in MOTOR_CONFIGS]

COUNTS_PER_REV = 1975.0
CALIBRATION_MOTOR_INDEX = 5
CALIBRATION_MOTOR_NAME = MOTOR_CONFIGS[CALIBRATION_MOTOR_INDEX][1]

CALIBRATION_ANGLE_NEG_LIMIT = -90.0
CALIBRATION_ANGLE_ZERO = 0.0
CALIBRATION_ANGLE_REF_START = -45.0
CALIBRATION_ANGLE_REF_END = 45.0

CAL_KP, CAL_KI, CAL_KD = 0.9, 0.0, 0.3
CAL_DEAD_ZONE, CAL_POS_THRESHOLD = 5, 5

STABILITY_WINDOW_SIZE = 15
STABILITY_THRESHOLD_COUNTS = 2
VELOCITY_SMOOTHING_WINDOW = 8
MOVEMENT_VELOCITY_RATIO = 0.4
MIN_VELOCITY_THRESHOLD = 10.0

WAIT_FOR_DATA_TIMEOUT = 5.0
WAIT_FOR_STABILITY_TIMEOUT = 10.0
WAIT_FOR_MOVEMENT_TIMEOUT = 5.0


def perform_initial_setup(body: QuadPilotBody) -> bool:
    """Performs a robust, sequential setup to ensure all commands are processed."""
    try:
        print("\n[PHASE 0: ROBUST SETUP]")
        
        print("Step 1/5: Setting control parameters...")
        if not body.set_control_params(CAL_KP, CAL_KI, CAL_KD, CAL_DEAD_ZONE, CAL_POS_THRESHOLD):
            raise Exception("Failed to set control parameters.")
        time.sleep(0.1)

        print("Step 2/5: Initializing all motor pins...")
        if not body.set_all_pins(pin_configs):
            raise Exception("Failed to set motor pins.")
        time.sleep(0.5)

        print("Step 3/5: Resetting all motors...")
        if not body.reset_all():
            raise Exception("Failed to reset all motors.")
        time.sleep(0.2)

        print("Step 4/5: Setting fast UDP send interval...")
        if not body.set_send_interval(2):
            raise Exception("Failed to set send interval.")
        time.sleep(0.1)

        # --- THIS IS THE CRITICAL FIX ---
        # Instead of the fast parallel set_all_control_status, we do it sequentially.
        print("Step 5/5: Enabling control for each motor sequentially...")
        for motor_idx in range(8):
            print(f"  -> Enabling motor {motor_idx}...")
            if not body.set_control_status(motor_idx, True):
                # If even one fails, the entire setup is invalid.
                raise Exception(f"Failed to enable control for motor {motor_idx}.")
            time.sleep(0.05) # Small delay to allow ESP32 to process before next command

        print("\nInitial setup successful.")
        return True

    except Exception as e:
        print(f"FATAL: Setup failed: {e}")
        return False


def calibrate_br_hip(body: QuadPilotBody):
    motor_index = CALIBRATION_MOTOR_INDEX
    print(f"\n--- Starting Backlash Calibration for {CALIBRATION_MOTOR_NAME} (Motor {motor_index}) ---")

    def get_current_pos():
        ip_idx = 0 if motor_index < 4 else 1
        esp_motor_idx = motor_index % 4
        data = body.get_latest_motor_data_for_esp(ip_idx)
        return data['encoderPos'][esp_motor_idx] if data and data.get('encoderPos') else None

    def set_target_angle(angle):
        angles = [0.0] * 8
        angles[motor_index] = float(angle)
        if not body.set_angles(angles):
             print(f"Warning: set_angles command for angle {angle} may have failed.")

    def wait_for_stability(timeout=WAIT_FOR_STABILITY_TIMEOUT):
        start_time = time.time()
        print(f"Waiting for motor to stabilize (max {timeout}s)...")
        while time.time() - start_time < timeout:
            with body.data_access_lock:
                 history = list(body._encoder_history[motor_index])
            if len(history) >= STABILITY_WINDOW_SIZE:
                recent_pos = [p for t, p in history[-STABILITY_WINDOW_SIZE:]]
                if (max(recent_pos) - min(recent_pos)) <= STABILITY_THRESHOLD_COUNTS:
                    latest_pos = history[-1][1]
                    print(f"Motor stable at ~{latest_pos} counts.")
                    return latest_pos
            time.sleep(0.01)
        print("Warning: Motor did not stabilize within timeout.")
        return get_current_pos()

    def wait_for_movement_start(direction_sign, velocity_threshold, timeout=WAIT_FOR_MOVEMENT_TIMEOUT):
        start_time = time.time()
        print(f"Waiting for movement start (velocity {' >' if direction_sign > 0 else ' <'} {direction_sign * velocity_threshold:.1f} counts/s)...")
        while time.time() - start_time < timeout:
            vel = body.get_smoothed_velocity(motor_index, VELOCITY_SMOOTHING_WINDOW)
            if vel * direction_sign > velocity_threshold:
                pos = get_current_pos()
                if pos is not None:
                    print(f"Movement detected at {pos} counts (velocity: {vel:.1f} counts/s).")
                    return pos
            time.sleep(0.005)
        print("Warning: Timed out waiting for movement start.")
        return None
    
    start_wait_time = time.time()
    print("Waiting for initial data packets...")
    while len(body._encoder_history[motor_index]) < VELOCITY_SMOOTHING_WINDOW:
        if time.time() - start_wait_time > WAIT_FOR_DATA_TIMEOUT:
            print("FATAL: Timed out waiting for sufficient encoder data after setup.")
            return False
        time.sleep(0.1)
    print("Initial data received. Proceeding with calibration.")

    print("\n[PHASE 1: MEASURING REFERENCE VELOCITY]")
    set_target_angle(CALIBRATION_ANGLE_REF_START)
    wait_for_stability()
    set_target_angle(CALIBRATION_ANGLE_REF_END)
    max_velocity = 0.0
    start_time = time.time()
    while time.time() - start_time < 5.0:
        vel = abs(body.get_smoothed_velocity(motor_index, VELOCITY_SMOOTHING_WINDOW))
        if vel > max_velocity: max_velocity = vel
        if vel < max_velocity * 0.5 and max_velocity > MIN_VELOCITY_THRESHOLD * 2: break
        time.sleep(0.01)
    reference_velocity = max(max_velocity, MIN_VELOCITY_THRESHOLD)
    movement_threshold = max(reference_velocity * MOVEMENT_VELOCITY_RATIO, MIN_VELOCITY_THRESHOLD)
    print(f"Reference Velocity: {reference_velocity:.1f} counts/s | Movement Threshold: {movement_threshold:.1f} counts/s")
    wait_for_stability()

    print("\n[PHASE 2: MEASURING BACKLASH, NEGATIVE -> POSITIVE]")
    set_target_angle(CALIBRATION_ANGLE_NEG_LIMIT)
    pos_at_neg_limit = wait_for_stability(timeout=15.0)
    set_target_angle(CALIBRATION_ANGLE_ZERO)
    pos_start_move_to_zero = wait_for_movement_start(1, movement_threshold)
    backlash_neg_to_pos = abs(pos_start_move_to_zero - pos_at_neg_limit) if all(p is not None for p in [pos_start_move_to_zero, pos_at_neg_limit]) else None
    pos_at_zero = wait_for_stability()

    print("\n[PHASE 3: MEASURING BACKLASH, POSITIVE -> NEGATIVE]")
    set_target_angle(CALIBRATION_ANGLE_NEG_LIMIT)
    pos_start_move_to_neg = wait_for_movement_start(-1, movement_threshold)
    backlash_pos_to_neg = abs(pos_at_zero - pos_start_move_to_neg) if all(p is not None for p in [pos_start_move_to_neg, pos_at_zero]) else None
    wait_for_stability(timeout=15.0)

    print("\n" + "="*40 + "\n--- CALIBRATION RESULTS ---\n" + "="*40)
    if backlash_neg_to_pos is not None: print(f"Backlash (moving POSITIVE): {backlash_neg_to_pos} counts ({(backlash_neg_to_pos/COUNTS_PER_REV)*360.0:.2f} deg)")
    else: print("Backlash (moving POSITIVE): FAILED TO MEASURE")
    if backlash_pos_to_neg is not None: print(f"Backlash (moving NEGATIVE): {backlash_pos_to_neg} counts ({(backlash_pos_to_neg/COUNTS_PER_REV)*360.0:.2f} deg)")
    else: print("Backlash (moving NEGATIVE): FAILED TO MEASURE")
    
    measured = [b for b in [backlash_neg_to_pos, backlash_pos_to_neg] if b is not None]
    if measured:
        avg = sum(measured) / len(measured)
        print(f"\nAverage Backlash: {avg:.1f} counts ({(avg/COUNTS_PER_REV)*360.0:.2f} degrees)")
    else:
        print("\nCould not calculate average backlash.")
    print("="*40 + "\n\n--- Calibration Finished ---")
    return True

# --- Main Execution ---
if __name__ == "__main__":
    body = None
    try:
        print("Initializing QuadPilotBody for calibration (with broadcast listener)...")
        body = QuadPilotBody(listen_for_broadcasts=True)
        
        if not perform_initial_setup(body):
            sys.exit(1)
        
        calibrate_br_hip(body)

    except Exception as e:
        import traceback
        print(f"\nFATAL: An unhandled error occurred in main execution: {e}")
        traceback.print_exc()

    finally:
        print("\nCleaning up...")
        if body:
            try:
                 print(f"Disabling control for all motors...")
                 # Use the sequential method for shutdown safety too
                 for motor_idx in range(8):
                    body.set_control_status(motor_idx, False)
                    time.sleep(0.05)
                 body.close()
            except Exception as cleanup_e:
                 print(f"Error during cleanup: {cleanup_e}")
        print("Cleanup complete.")

        