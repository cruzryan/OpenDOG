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
# This should match or be consistent with your main control script's motor mapping
# For display, we primarily need names for motors 0-7.
# (Global_Motor_Index, "Name")
MOTOR_NAMES = [
    (0, "FL_knee"), (1, "FR_tigh"), (2, "FR_knee"), (3, "FL_tigh"),
    (4, "BR_knee"), (5, "BR_tigh"), (6, "BL_knee"), (7, "BL_tigh"),
]
# Create a simple lookup map
MOTOR_NAME_MAP = {idx: name for idx, name in MOTOR_NAMES}
NUM_MOTORS_PER_ESP = 4


def display_dashboard_data(body_monitor: QuadPilotBody):
    print("\nStarting Data Dashboard. Press Ctrl+C to exit.")
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=" * 60)
            print("QUADRUPED REAL-TIME DATA DASHBOARD")
            print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)

            for esp_idx, esp_ip in enumerate(body_monitor.ips):
                print(f"\n--- ESP32 #{esp_idx + 1} ({esp_ip}) ---")

                if not body_monitor.is_data_available_from_esp(esp_idx):
                    print("  No data received yet from this ESP32.")
                    continue

                # IMU Data
                mpu_avail = body_monitor.is_mpu_available_for_esp(esp_idx)
                print(f"  MPU6050 Status: {'AVAILABLE' if mpu_avail else 'NOT AVAILABLE'}")
                if mpu_avail:
                    imu_data = body_monitor.get_latest_imu_data_for_esp(esp_idx)
                    if imu_data:
                        print("  IMU Data:")
                        print(f"    Accel (g):  X={imu_data.get('accel_x', 0.0): >6.2f}, Y={imu_data.get('accel_y', 0.0): >6.2f}, Z={imu_data.get('accel_z', 0.0): >6.2f}")
                        print(f"    Gyro (dps): X={imu_data.get('gyro_x', 0.0): >6.2f}, Y={imu_data.get('gyro_y', 0.0): >6.2f}, Z={imu_data.get('gyro_z', 0.0): >6.2f}")
                        print(f"    Temp (C):   {imu_data.get('temp', 0.0):.2f}")
                    else:
                        print("    IMU data not populated yet (or error in reception).")
                
                # Motor Data
                print("\n  Motor Data (Angle, EncoderPos, TargetPos):")
                esp_motor_data = body_monitor.get_latest_motor_data_for_esp(esp_idx)
                if esp_motor_data:
                    for motor_local_idx in range(NUM_MOTORS_PER_ESP):
                        global_motor_idx = esp_idx * NUM_MOTORS_PER_ESP + motor_local_idx
                        motor_name = MOTOR_NAME_MAP.get(global_motor_idx, f"M{global_motor_idx}")
                        
                        angle = esp_motor_data['angles'][motor_local_idx] if motor_local_idx < len(esp_motor_data.get('angles',[])) else "N/A"
                        enc_pos = esp_motor_data['encoderPos'][motor_local_idx] if motor_local_idx < len(esp_motor_data.get('encoderPos',[])) else "N/A"
                        tgt_pos = esp_motor_data['targetPos'][motor_local_idx] if motor_local_idx < len(esp_motor_data.get('targetPos',[])) else "N/A"
                        
                        # Optional Debug Data
                        # dbg1 = esp_motor_data['debug'][motor_local_idx] if motor_local_idx < len(esp_motor_data.get('debug',[])) else "N/A"
                        # dbg2 = esp_motor_data['debugComputed'][motor_local_idx] if motor_local_idx < len(esp_motor_data.get('debugComputed',[])) else "N/A"
                        
                        print(f"    {motor_name:<8s} (Global M{global_motor_idx}): {angle:>6.1f}, {enc_pos:>7}, {tgt_pos:>7}")
                else:
                    print("    Motor data not populated yet for this ESP.")
            
            print("\n" + "=" * 60)
            print("Press Ctrl+C to stop the dashboard.")
            time.sleep(0.1) # Refresh rate of the dashboard display

    except KeyboardInterrupt:
        print("\nDashboard stopping due to Ctrl+C.")
    except Exception as e:
        print(f"\nAn error occurred in the dashboard display loop: {e}")

if __name__ == "__main__":
    monitor_instance = None
    print("Initializing QuadPilotBody for monitoring...")
    try:
        # IMPORTANT: listen_for_broadcasts=True for the dashboard
        monitor_instance = QuadPilotBody(listen_for_broadcasts=True)
        print("QuadPilotBody (monitoring instance) initialized. Listener thread started.")
        
        # Optionally, ask ESP32s to send data at a certain rate if they aren't already
        # This might conflict if udp_walk.py also tries to set this.
        # Best if ESP32s default to a reasonable broadcast rate.
        # Or, only one script (e.g., udp_walk.py) sets it once at the start.
        # print("Requesting ESP32s to send data every 50ms (if not already)...")
        # if not monitor_instance.set_send_interval(50):
        #     print("Warning: Failed to set send interval for one or both ESPs from dashboard.")
        # time.sleep(0.2) # Give time for command to be processed

        display_dashboard_data(monitor_instance)

    except Exception as e:
        print(f"FATAL: Failed to initialize or run dashboard: {e}")
    finally:
        if monitor_instance:
            print("Closing QuadPilotBody (monitoring instance)...")
            monitor_instance.close()
        print("Dashboard has shut down.")