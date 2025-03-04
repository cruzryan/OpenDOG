import os
import sys
import time
import json
import threading
from pynput import keyboard
import requests

# Add the code directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from quadpilot import QuadPilotBody  # Assuming quadpilot.py contains the updated QuadPilotBody class

# Initialize with ESP32's IP address
body = QuadPilotBody("192.168.0.198")

# Motor indices
MOTOR_0 = 0
MOTOR_1 = 1

# Set initial PID parameters (applies to both motors)
body.set_control_params(0.9, 0.001, 0.3, 10, 5)

# Set pins and enable control for both motors
# Motor 0
body.set_pins(motor=MOTOR_0, ENCODER_A=8, ENCODER_B=18, IN1=12, IN2=11)
body.set_control_status(motor=MOTOR_0, status=True)
# Motor 1
body.set_pins(motor=MOTOR_1, ENCODER_A=46, ENCODER_B=3, IN1=9, IN2=10)
body.set_control_status(motor=MOTOR_1, status=True)

# Track current target angles for each motor
target_angles = {MOTOR_0: 0, MOTOR_1: 0}  # Start at 0 degrees for both

# Debounce mechanism variables
last_key_press_time = 0
debounce_interval = 0.2  # seconds

# SSE handling variables
sse_active = threading.Event()
sse_active.set()  # Start as active
sse_thread = None
lock = threading.Lock()

# Function to set angle for a specific motor
def set_angle(motor, angle):
    global last_key_press_time
    current_time = time.time()
    if current_time - last_key_press_time > debounce_interval:
        last_key_press_time = current_time
        target_angles[motor] = angle  # Update the target angle
        print(f"Setting motor {motor} angle to {angle} degrees")
        # Pause SSE briefly to allow the set_angle request to go through
        with lock:
            sse_active.clear()  # Signal SSE thread to pause
            time.sleep(0.05)  # Brief pause to let SSE loop yield
            body.set_angle(motor=motor, a=angle)  # Updated API with motor param
            sse_active.set()  # Resume SSE

# SSE processing function (updated to show both motors)
def sse_listener():
    url = "http://192.168.0.198:82/events"
    while True:
        if sse_active.is_set():
            try:
                session = requests.Session()
                response = session.get(url, stream=True, timeout=5)
                last_time = time.time()
                for line in response.iter_lines():
                    if not sse_active.is_set():
                        response.close()
                        break
                    if line and line.startswith(b'data: '):
                        data = json.loads(line[6:].decode('utf-8'))
                        # Extract data for both motors
                        angle_0 = data["angles"][MOTOR_0]
                        encoder_pos_0 = data["encoderPos"][MOTOR_0]
                        angle_1 = data["angles"][MOTOR_1]
                        encoder_pos_1 = data["encoderPos"][MOTOR_1]
                        current_time = time.time()
                        print(f"Motor {MOTOR_0} - Angle: {angle_0:.2f}, Encoder Position: {encoder_pos_0}")
                        print(f"Motor {MOTOR_1} - Angle: {angle_1:.2f}, Encoder Position: {encoder_pos_1}")
                        print(f"Time since last update: {(current_time - last_time) * 1000:.2f} ms")
                        last_time = current_time
            except Exception as e:
                print(f"SSE error: {e}")
                time.sleep(1)  # Retry after a short delay
        else:
            time.sleep(0.01)  # Wait while paused

# Start SSE in a separate thread
sse_thread = threading.Thread(target=sse_listener, daemon=True)
sse_thread.start()

# Function to handle key presses
def on_press(key):
    try:
        if key.char == 'a':
            set_angle(MOTOR_0, target_angles[MOTOR_0] + 10)  # Motor 0: +10 degrees
        elif key.char == 'd':
            set_angle(MOTOR_0, target_angles[MOTOR_0] - 10)  # Motor 0: -10 degrees
        elif key.char == 'w':
            set_angle(MOTOR_1, target_angles[MOTOR_1] + 10)  # Motor 1: +10 degrees
        elif key.char == 's':
            set_angle(MOTOR_1, target_angles[MOTOR_1] - 10)  # Motor 1: -10 degrees
    except AttributeError:
        pass  # Ignore non-character keys

# Start the keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Main loop (just keep the script running)
try:
    while True:
        time.sleep(1)  # Keep main thread alive
except KeyboardInterrupt:
    print("Stopped by user")
finally:
    listener.stop()
    sse_active.clear()  # Stop SSE thread cleanly
    # Disable control for both motors on exit
    body.set_control_status(motor=MOTOR_0, status=False)
    body.set_control_status(motor=MOTOR_1, status=False)