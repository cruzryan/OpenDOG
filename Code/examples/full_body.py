import os
import sys
import time
import threading
from pynput import keyboard
import uuid


# Add the code directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from quadpilot import QuadPilotBody

# Initialize with two ESP32 IPs (192.168.137.100 for motors 0-3, 192.168.137.101 for motors 4-7)
body = QuadPilotBody()

# Motor indices
NUM_MOTORS = 8
TARGET_MOTORS = [0,1,2,3,4,5,6,7]  # Back Left Knee, Back Left Knee, Back Right Knee

# Set initial PID parameters for both ESP32s
body.set_control_params(1.5, 0.001, 0.3, 5, 5)
# Note: If motor overshoots, try reducing kp (e.g., 0.5) or increasing kd (e.g., 0.5)
print("Set control parameters")

# Motor configurations


MOTOR_CONFIGS = [
    (0, "Front Left (Knee)", 39, 40, 41, 42),  # IP1
    (1, "Front Right (Hip)", 37, 38, 1, 2),   # IP1

    (2, "Front Right (Knee)", 17, 18, 5, 4),  # IP1
    (3, "Front Left (Hip)", 16, 15, 7, 6),    # IP1

    
    (4, "Back Right (Knee)", 37, 38, 1, 2),    # IP2
    (5, "Back Right (Hip)", 40, 39, 42, 41),   # IP2

    (6, "Back Left (Knee)", 15, 16, 6, 7),   # IP2
    (7, "Back Left (Hip)", 18, 17, 4, 5),    # IP2
]

pin_configs = [(enc_a, enc_b, in1, in2) for _, _, enc_a, enc_b, in1, in2 in MOTOR_CONFIGS]

print("Initializing pins for all motors (only target motors will be active)")
try:
    body.set_all_pins(pin_configs)
    print("Set pins successful")
except Exception as e:
    print(f"Failed to set pins: {e}")
time.sleep(1)

# Reset all motors
print("Resetting all motors")
try:
    body.reset_all()
    print("Reset all successful")
except Exception as e:
    print(f"Failed to reset all: {e}")

# Initialize only the target motors
for motor_idx in TARGET_MOTORS:
    motor_config = MOTOR_CONFIGS[motor_idx]
    description = motor_config[1]
    print(f"Initializing {description} (Motor {motor_idx})")
    try:
        body.set_control_status(motor=motor_idx, status=True)
        print(f"Control status set to enabled for Motor {motor_idx}")
    except Exception as e:
        print(f"Failed to set control status for Motor {motor_idx}: {e}")
    time.sleep(0.5)

print("Target motors initialized and control enabled!")

# Debounce mechanism variables
last_key_press_time = 0
debounce_interval = 0.2

# Function to set angle for target motors
def set_angles(angles):
    global last_key_press_time
    current_time = time.time()
    if current_time - last_key_press_time > debounce_interval:
        last_key_press_time = current_time
        print(f"\nSetting angles {angles}")
        try:
            body.set_angles(angles)
        except Exception as e:
            print(f"Failed to set angle: {e}")

# Motor control state
motor_control_enabled = True

# Function to handle key presses
def on_press(key):
    global motor_control_enabled
    try:
        if key.char == 't':
            motor_control_enabled = not motor_control_enabled
            print(f"\nMotor control {'enabled' if motor_control_enabled else 'disabled'}")
            try:
                if motor_control_enabled:
                    body.reset_all()
                    time.sleep(0.5)
                    for motor_idx in TARGET_MOTORS:
                        body.set_control_status(motor=motor_idx, status=True)
                else:
                    body.reset_all()
                    time.sleep(0.5)
                    for motor_idx in TARGET_MOTORS:
                        body.set_control_status(motor=motor_idx, status=False)
            except Exception as e:
                print(f"Failed to toggle control status: {e}")
        elif motor_control_enabled:
            if key.char == 'a':
                angles = [0] * NUM_MOTORS
                
                # Front Left (Hip)
                angles[3] = -45
                # Front Left (Knee)
                angles[0] = 45  

                # Front Right (Hip)
                angles[1] = 45
                # Front Right (Knee)
                angles[2] = 45

                # Back Right (Hip)
                angles[5] = 45
                # Back Right (Knee)
                angles[4] = -45

                # Back Left (Hip)
                angles[7] = 45
                # Back Left (Knee)
                angles[6] = -45 #m

                set_angles(angles)
            elif key.char == 'd':
                angles = [0] * NUM_MOTORS
                set_angles(angles)

    except AttributeError:
        pass

# Start the keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Main loop
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    listener.stop()
    try:
        for motor_idx in TARGET_MOTORS:
            body.set_control_status(motor=motor_idx, status=False)
        time.sleep(0.3)
    except Exception as e:
        print(f"Failed to disable control: {e}")
    print("Control disabled for target motors")