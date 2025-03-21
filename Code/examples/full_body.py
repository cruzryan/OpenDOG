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

from quadpilot import QuadPilotBody  # Assuming quadpilot.py contains QuadPilotBody

IP = "192.168.137.170"  # Update if your ESP32's IP changes

# Initialize with ESP32's IP address
body = QuadPilotBody(IP)

# Define motor roles
knee_motors = [0, 2, 4, 6]  # Back Left, Front Left, Back Right, Front Right Knee
hip_motors = [1, 3, 5, 7]   # Back Left, Front Left, Back Right, Front Right Hip

# Set initial PID parameters
body.set_control_params(0.9, 0.001, 0.3, 10, 5)

# Set pins for all motors (using old assignments where possible, adjusted for safety)
body.set_pins(motor=0, ENCODER_A=19, ENCODER_B=21, IN1=22, IN2=23)  # Back Left Knee (adjusted from 47, 21, 39, 40)
time.sleep(0.1)
body.set_pins(motor=1, ENCODER_A=32, ENCODER_B=33, IN1=25, IN2=26)  # Back Left Hip (adjusted from 45, 48, 38, 37)
time.sleep(0.1)
body.set_pins(motor=2, ENCODER_A=34, ENCODER_B=35, IN1=27, IN2=14)  # Front Left Knee (adjusted from 36, 35, 42, 41)
time.sleep(0.1)
body.set_pins(motor=3, ENCODER_A=18, ENCODER_B=5, IN1=2, IN2=4)    # Front Left Hip (adjusted from 19, 20, 2, 1)
time.sleep(0.1)
body.set_pins(motor=4, ENCODER_A=12, ENCODER_B=13, IN1=17, IN2=18)  # Back Right Knee (worked before)
time.sleep(0.1)
body.set_pins(motor=5, ENCODER_A=15, ENCODER_B=9, IN1=16, IN2=8)    # Back Right Hip (adjusted from 46, 9, 16, 15)
time.sleep(0.1)
body.set_pins(motor=6, ENCODER_A=3, ENCODER_B=0, IN1=4, IN2=5)      # Front Right Knee (adjusted from 3, 8, 4, 5)
time.sleep(0.1)
body.set_pins(motor=7, ENCODER_A=10, ENCODER_B=11, IN1=6, IN2=7)    # Front Right Hip (worked before)
time.sleep(0.1)

# Initialize all motors
for motor in range(8):
    body.set_control_status(motor=motor, status=True)  # Enable all motors
    body.reset(motor=motor)
    time.sleep(0.1)

print("Control enabled for all motors!")

# Debounce mechanism
last_key_press_time = 0
debounce_interval = 0.2

# SSE handling
sse_active = threading.Event()
sse_active.set()
sse_thread = None
lock = threading.Lock()

# Rolling window averages
angle_windows = {i: [] for i in range(8)}
encoder_windows = {i: [] for i in range(8)}
WINDOW_SIZE = 10

# Batch set angles function
def set_angles(angles):
    global last_key_press_time
    current_time = time.time()
    if current_time - last_key_press_time > debounce_interval:
        last_key_press_time = current_time
        angle_str = ",".join(map(str, angles))
        print(f"\nSetting angles: {angle_str}")
        with lock:
            sse_active.clear()
            time.sleep(0.05)
            url = f"http://{IP}:82/set_angles?angles={angle_str}"
            try:
                response = requests.get(url, timeout=5)
                if response.status_code != 200:
                    print(f"Failed to set angles: {response.status_code}")
            except Exception as e:
                print(f"Error setting angles: {e}")
            sse_active.set()

# SSE listener
def sse_listener():
    url = f"http://{IP}:82/events"
    while True:
        if sse_active.is_set():
            try:
                session = requests.Session()
                response = session.get(url, stream=True, timeout=5)
                for line in response.iter_lines():
                    if not sse_active.is_set():
                        response.close()
                        break
                    if line and line.startswith(b'data: '):
                        data = json.loads(line[6:].decode('utf-8'))
                        output = ""
                        for i in range(8):
                            angle = data["angles"][i]
                            encoder_pos = data["encoderPos"][i]
                            angle_windows[i].append(angle)
                            encoder_windows[i].append(encoder_pos)
                            if len(angle_windows[i]) > WINDOW_SIZE:
                                angle_windows[i].pop(0)
                                encoder_windows[i].pop(0)
                            avg_angle = sum(angle_windows[i]) / len(angle_windows[i])
                            avg_encoder = sum(encoder_windows[i]) / len(encoder_windows[i])
                            output += f" | M{i}: {avg_angle:.2f}, {avg_encoder:.0f}"
                        print(f"\r{output}", end="")
            except Exception as e:
                print(f"SSE error: {e}")
                time.sleep(1)
        else:
            time.sleep(0.01)

# Start SSE thread
sse_thread = threading.Thread(target=sse_listener, daemon=True)
sse_thread.start()

# Handle key presses
def on_press(key):
    try:
        angles = [0] * 8
        if key.char == 'a':
            for motor in knee_motors:
                angles[motor] = 25
            set_angles(angles)
        elif key.char == 'd':
            for motor in knee_motors:
                angles[motor] = -25
            set_angles(angles)
    except AttributeError:
        pass

# Start keyboard listener
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
    sse_active.clear()
    for motor in range(8):
        body.set_control_status(motor=motor, status=False)
    print("Control disabled")