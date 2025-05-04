import os
import sys

# Add the code directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

import time
import threading
from pynput import keyboard
from quadpilot import QuadPilotBody


# Initialize QuadPilotBody
body = QuadPilotBody()

# Motor indices
NUM_MOTORS = 8
TARGET_MOTORS = [0, 1, 2, 3, 4, 5, 6, 7]  # All motors: Front/Back Left/Right Knee/Hip

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
                angles[6] = -45
                set_angles(angles)
            elif key.char == 'd':
                angles = [0] * NUM_MOTORS
                set_angles(angles)
            
            # move the knees 10 degree
            elif key.char == 'w':
                # Phase 1: Lift Front Left and Back Right legs, swing them forward
                angles = [0] * NUM_MOTORS
                # Front Left: Lift and swing forward
                angles[3] = -60  # Hip swings forward more
                angles[0] = 60   # Knee lifts higher
                # Front Right: Stays on ground (support)
                angles[1] = 45
                angles[2] = 45
                # Back Right: Lift and swing forward
                angles[5] = 60   # Hip swings forward
                angles[4] = -60  # Knee lifts
                # Back Left: Stays on ground (support)
                angles[7] = 45
                angles[6] = -45
                set_angles(angles)
            elif key.char == 's':
                # Phase 2: Lift Front Right and Back Left legs, swing them forward
                angles = [0] * NUM_MOTORS
                # Front Left: Returns to support position
                angles[3] = -45
                angles[0] = 45
                # Front Right: Lift and swing forward
                angles[1] = 60   # Hip swings forward
                angles[2] = 60   # Knee lifts
                # Back Right: Returns to support position
                angles[5] = 45
                angles[4] = -45
                # Back Left: Lift and swing forward
                angles[7] = 60   # Hip swings forward
                angles[6] = -60  # Knee lifts
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