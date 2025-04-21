import os
import sys
import time
import threading
from pynput import keyboard

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
TARGET_MOTOR = 0  # Back Left Turning Motor (Knee), controlled by 192.168.137.101

# Set initial PID parameters for both ESP32s
body.set_control_params(0.9, 0.001, 0.3, 5, 5)
# Note: If motor overshoots, try reducing kp (e.g., 0.5) or increasing kd (e.g., 0.5)
# body.set_control_params(0.5, 0.001, 0.5, 5, 5)
print("Set control parameters")

# Motor configurations (real config for motor 7, dummy configs for others)
MOTOR_CONFIGS = [
    (0, "Back Left Turning Motor (Knee)", 16, 15, 7, 6),  # Real motor
    (1, "Dummy Motor 1", 0, 0, 0, 0),  # Dummy
    (2, "Dummy Motor 2", 0, 0, 0, 0),  # Dummy
    (3, "Dummy Motor 3", 0, 0, 0, 0),  # Dummy
    (4, "Dummy Motor 4", 0, 0, 0, 0),  # Dummy
    (5, "Dummy Motor 5", 0, 0, 0, 0),  # Dummy
    (6, "Dummy Motor 6", 0, 0, 0, 0),  # Dummy
    (7, "Dummy Motor 7", 0, 0, 0, 0),  # Dummy
]

#MOTORS
'''
ID NAME             ENCA ENCB IN1 IN2
0 Front Left (Knee) 39, 40, 41, 42
1 Front Right (Hip) 37, 38, 1, 2

2 Front Right (Knee) 17, 18, 5, 4
3 Front Left (Hip) 16, 15, 7, 6

4 Back Left (Knee) 37, 38, 1, 2
5 Back Left (Hip)  40, 39, 42, 41
6 Back Right (Knee) 15, 16, 6, 7
7 Back Right (Hip)  18, 17, 4, 5

'''

pin_configs = [(enc_a, enc_b, in1, in2) for _, _, enc_a, enc_b, in1, in2 in MOTOR_CONFIGS]

print("Initializing pins for all motors (only motor 7 will be active)")
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

# Initialize only the target motor
motor_idx, description, enc_a, enc_b, in1, in2 = MOTOR_CONFIGS[TARGET_MOTOR]
print(f"Initializing {description} (Motor {motor_idx})")
try:
    body.set_control_status(motor=motor_idx, status=True)
    print("Control status set to enabled")
except Exception as e:
    print(f"Failed to set control status: {e}")
time.sleep(0.1)

print("Back Left Turning Motor initialized and control enabled!")

# Debounce mechanism variables
last_key_press_time = 0
debounce_interval = 0.2

# Function to set angle for only motor 7
def set_angle(angle):
    global last_key_press_time
    current_time = time.time()
    if current_time - last_key_press_time > debounce_interval:
        last_key_press_time = current_time
        print(f"\nSetting angle for Motor 7: {angle}")
        try:
            angles = [0] * NUM_MOTORS
            angles[TARGET_MOTOR] = angle
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
                    body.set_control_status(motor=TARGET_MOTOR, status=True)
                else:
                    body.reset_all()
                    time.sleep(0.5)
                    body.set_control_status(motor=TARGET_MOTOR, status=False)
            except Exception as e:
                print(f"Failed to toggle control status: {e}")
        elif motor_control_enabled:
            if key.char == 'a':
                set_angle(45)
            elif key.char == 'd':
                set_angle(0)
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
        body.set_control_status(motor=TARGET_MOTOR, status=False)
        time.sleep(0.3)
    except Exception as e:
        print(f"Failed to disable control: {e}")
    print("Control disabled for Back Left Turning Motor")