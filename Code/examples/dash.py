# dash.py

import os
import sys
import time
import threading

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

# --- Motor Name Configuration (for display purposes) ---
MOTOR_NAMES = [
    (0, "FL_knee"), (1, "FR_tigh"), (2, "FR_knee"), (3, "FL_tigh"),
    (4, "BR_knee"), (5, "BR_tigh"), (6, "BL_knee"), (7, "BL_tigh"),
]
MOTOR_NAME_MAP = {idx: name for idx, name in MOTOR_NAMES}
NUM_MOTORS_PER_ESP = 4


def display_dashboard_data(body_monitor: QuadPilotBody):
    print("\nStarting Data Dashboard. Press Ctrl+C to exit.")
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=" * 70) # Increased width
            print("QUADRUPED REAL-TIME DATA DASHBOARD (DMP Enhanced)")
            print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)

            for esp_idx, esp_ip in enumerate(body_monitor.ips):
                print(f"\n--- ESP32 #{esp_idx + 1} ({esp_ip}) ---")

                if not body_monitor.is_data_available_from_esp(esp_idx):
                    print("  No data received yet from this ESP32.")
                    continue

                # Get motor data first to access 'dmp_ready' status
                esp_motor_data = body_monitor.get_latest_motor_data_for_esp(esp_idx)
                dmp_is_ready = False
                if esp_motor_data:
                    dmp_is_ready = esp_motor_data.get('dmp_ready', False) # Use dmp_ready from motor_data
                else: # Fallback if motor_data itself is None
                    dmp_is_ready = body_monitor.is_dmp_ready_for_esp(esp_idx)


                print(f"  DMP Status: {'READY' if dmp_is_ready else 'NOT READY / UNAVAILABLE'}")

                if dmp_is_ready:
                    # Use the new getter for structured DMP data
                    dmp_data = body_monitor.get_latest_dmp_data_for_esp(esp_idx)
                    if dmp_data:
                        quat = dmp_data.get('quaternion', {})
                        accel = dmp_data.get('world_accel_mps2', {})
                        ypr = dmp_data.get('ypr_deg', {})

                        print("  DMP Data:")
                        print(f"    Quaternion: W={quat.get('w', 0.0): >6.3f}, X={quat.get('x', 0.0): >6.3f}, Y={quat.get('y', 0.0): >6.3f}, Z={quat.get('z', 0.0): >6.3f}")
                        print(f"    World Accel (m/s^2): AX={accel.get('ax', 0.0): >6.3f}, AY={accel.get('ay', 0.0): >6.3f}, AZ={accel.get('az', 0.0): >6.3f}")
                        print(f"    YPR (degrees): Yaw={ypr.get('yaw', 0.0): >6.1f}, Pitch={ypr.get('pitch', 0.0): >6.1f}, Roll={ypr.get('roll', 0.0): >6.1f}")
                    else:
                        print("    DMP data not populated yet (or error in reception).")
                else:
                    # Optionally, try to display old IMU data if you want to keep that path for non-DMP MPU
                    # For now, we'll just indicate DMP is not ready.
                    # legacy_imu_data = body_monitor.get_latest_imu_data_for_esp(esp_idx) # This getter is now deprecated
                    # if legacy_imu_data:
                    #     print("  Legacy IMU Data (non-DMP):")
                    #     print(f"    Accel (raw?): X={legacy_imu_data.get('accel_x', 0.0): >6.2f}, Y={legacy_imu_data.get('accel_y', 0.0): >6.2f}, Z={legacy_imu_data.get('accel_z', 0.0): >6.2f}")
                    #     print(f"    Gyro (raw?):  X={legacy_imu_data.get('gyro_x', 0.0): >6.2f}, Y={legacy_imu_data.get('gyro_y', 0.0): >6.2f}, Z={legacy_imu_data.get('gyro_z', 0.0): >6.2f}")
                    pass # Handled by "DMP Status: NOT READY"

                # Motor Data
                print("\n  Motor Data (Angle, EncoderPos, TargetPos):")
                if esp_motor_data: # We fetched this earlier
                    for motor_local_idx in range(NUM_MOTORS_PER_ESP):
                        global_motor_idx = esp_idx * NUM_MOTORS_PER_ESP + motor_local_idx
                        motor_name = MOTOR_NAME_MAP.get(global_motor_idx, f"M{global_motor_idx}")

                        angle_val = esp_motor_data['angles'][motor_local_idx] if motor_local_idx < len(esp_motor_data.get('angles',[])) else "N/A"
                        enc_pos_val = esp_motor_data['encoderPos'][motor_local_idx] if motor_local_idx < len(esp_motor_data.get('encoderPos',[])) else "N/A"
                        tgt_pos_val = esp_motor_data['targetPos'][motor_local_idx] if motor_local_idx < len(esp_motor_data.get('targetPos',[])) else "N/A"

                        # Convert to string for consistent formatting if N/A
                        angle_str = f"{angle_val:>6.1f}" if isinstance(angle_val, (int, float)) else f"{str(angle_val):>6}"
                        enc_pos_str = f"{enc_pos_val:>7}" if isinstance(enc_pos_val, (int, float)) else f"{str(enc_pos_val):>7}"
                        tgt_pos_str = f"{tgt_pos_val:>7}" if isinstance(tgt_pos_val, (int, float)) else f"{str(tgt_pos_val):>7}"

                        print(f"    {motor_name:<8s} (Global M{global_motor_idx}): {angle_str}, {enc_pos_str}, {tgt_pos_str}")
                else:
                    print("    Motor data not populated yet for this ESP.")

            print("\n" + "=" * 70)
            print("Press Ctrl+C to stop the dashboard.")
            time.sleep(0.1) # Refresh rate of the dashboard display

    except KeyboardInterrupt:
        print("\nDashboard stopping due to Ctrl+C.")
    except Exception as e:
        print(f"\nAn error occurred in the dashboard display loop: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    monitor_instance = None
    print("Initializing QuadPilotBody for monitoring...")
    try:
        monitor_instance = QuadPilotBody(listen_for_broadcasts=True)
        print("QuadPilotBody (monitoring instance) initialized. Listener thread started.")
        time.sleep(0.5) # Give a moment for the listener to potentially receive first packets
        display_dashboard_data(monitor_instance)

    except Exception as e:
        print(f"FATAL: Failed to initialize or run dashboard: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if monitor_instance:
            print("Closing QuadPilotBody (monitoring instance)...")
            monitor_instance.close()
        print("Dashboard has shut down.")