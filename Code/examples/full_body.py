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

IP = "192.168.137.36"

# Initialize with ESP32's IP address
body = QuadPilotBody(IP)

# Motor indices (0-7 for 8 motors)
NUM_MOTORS = 8

# Set initial PID parameters for all motors
body.set_control_params(0.9, 0.001, 0.3, 10, 5)  # Applies to all motors
print("Set control parameters for all motors")

# Motor configurations (index: description, pins)
MOTOR_CONFIGS = [
    (0, "Back Left Arrow Motor (Knee)", 47, 21, 39, 40),
    (1, "Back Left Turning Motor (Hip)", 45, 48, 38, 37),
    (2, "Front Left Arrow Motor (Knee)", 36, 35, 42, 41),
    (3, "Front Left Turning Motor (Hip)", 19, 20, 2, 1),
    (4, "Back Right Arrow Motor (Knee)", 12, 13, 17, 18),
    (5, "Back Right Turning Motor (Hip)", 46, 9, 16, 15),
    (6, "Front Right Arrow Motor (Knee)", 3, 8, 4, 5),
    (7, "Front Right Turning Motor (Hip)", 10, 11, 6, 7),
]


# pin_configs = [(enc_a, enc_b, in1, in2) for _, _, enc_a, enc_b, in1, in2 in MOTOR_CONFIGS]

# print("Initializing all motors with set_all_pins")
# body.set_all_pins(pin_configs)
# time.sleep(1)

# Initialize all motors
for motor_idx, description, enc_a, enc_b, in1, in2 in MOTOR_CONFIGS:
    print(f"Initializing {description} (Motor {motor_idx})")
    response = body.session.get(f"http://{IP}:82/set_pins?motor={motor_idx}&ENCODER_A={enc_a}&ENCODER_B={enc_b}&IN1={in1}&IN2={in2}")
    print(f"Set Pins Response: {response.text}")
    time.sleep(0.1)
    response = body.session.get(f"http://{IP}:82/set_control_status?motor={motor_idx}&status=1")
    print(f"Control Status Response: {response.text}")
    time.sleep(0.1)
    body.reset(motor=motor_idx)
    time.sleep(0.1)

print("All motors initialized and control enabled!")

# Debounce mechanism variables
last_key_press_time = 0
debounce_interval = 0.2  # seconds

# SSE handling variables
sse_active = threading.Event()
sse_active.set()  # Start as active
sse_thread = None
lock = threading.Lock()

# Variables for rolling window average
angle_window = [[] for _ in range(NUM_MOTORS)]  # List of lists for each motor
encoder_window = [[] for _ in range(NUM_MOTORS)]
WINDOW_SIZE = 10

# Function to set angles for all 8 motors
def set_angles(angles):
    global last_key_press_time
    current_time = time.time()
    if current_time - last_key_press_time > debounce_interval:
        last_key_press_time = current_time
        print(f"\nSetting angles: {angles}")
        with lock:
            sse_active.clear()  # Signal SSE thread to pause
            time.sleep(0.05)  # Brief pause to let SSE loop yield
            body.set_angles(angles)  # Set all 8 angles at once
            sse_active.set()  # Resume SSE

# SSE processing function (updated for multi-motor JSON format)
def sse_listener():
    url = f"http://{IP}:82/events"
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
                        current_time = time.time()
                        
                        # Process data for all motors
                        output = ""
                        for i in range(NUM_MOTORS):
                            angle = data["angles"][i]
                            encoder_pos = data["encoderPos"][i]
                            
                            # Update rolling windows for this motor
                            angle_window[i].append(angle)
                            encoder_window[i].append(encoder_pos)
                            if len(angle_window[i]) > WINDOW_SIZE:
                                angle_window[i].pop(0)
                            if len(encoder_window[i]) > WINDOW_SIZE:
                                encoder_window[i].pop(0)
                                
                            # Calculate averages
                            avg_angle = sum(angle_window[i]) / len(angle_window[i])
                            avg_encoder = sum(encoder_window[i]) / len(encoder_window[i])
                            
                            output += f" M{i}: {avg_angle:.2f}°/{avg_encoder:.2f}"
                        
                        # Print averages with carriage return
                        print(f"\r{output}", end="")
                        last_time = current_time
            except Exception as e:
                print(f"SSE error: {e}")
                time.sleep(1)  # Retry after a short delay
        else:
            time.sleep(0.01)  # Wait while paused

# Start SSE in a separate thread
sse_thread = threading.Thread(target=sse_listener, daemon=True)
sse_thread.start()

# Add global variable for motor control state
motor_control_enabled = True

# Function to handle key presses
def on_press(key):
    global motor_control_enabled
    try:
        # Knee motors: 0, 2, 4, 6 (Arrow motors)
        # Hip motors: 1, 3, 5, 7 (Turning motors)
        angles = [0] * NUM_MOTORS  # Default all to 0
        if key.char == 't':
            motor_control_enabled = not motor_control_enabled
            print(f"\nMotor control {'enabled' if motor_control_enabled else 'disabled'}")
            for i in range(NUM_MOTORS):
                if motor_control_enabled:
                    body.reset(motor=i)
                    time.sleep(0.5)
                    body.set_control_status(motor=i, status=motor_control_enabled)
                    time.sleep(0.1)
                else:
                    body.reset(motor=i)
                    time.sleep(0.5)
                    body.set_control_status(motor=i, status=motor_control_enabled)
                    time.sleep(0.5)
        elif key.char == 'a':
            # Set knee motors to 15, hip motors stay at 0
            angles[0] = 0  # Back Left Knee
            angles[2] = 0  # Front Left Knee
            angles[4] = 15  # Back Right Knee
            angles[7] = 15  # Front Right Knee
            angles = [15] * NUM_MOTORS  # Default all to 0

            set_angles(angles)
        elif key.char == 'd':
            # Set knee motors to -15, hip motors stay at 0
            angles[0] = 0  # Back Left Knee
            angles[2] = 0  # Front Left Knee
            angles[4] = -15  # Back Right Knee
            angles[6] = -15  # Front Right Knee
            angles = [-15] * NUM_MOTORS  # Default all to 0

            set_angles(angles)
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
    print("\nStopped by user")
finally:
    listener.stop()
    sse_active.clear()  # Stop SSE thread cleanly
    for i in range(NUM_MOTORS):
        body.set_control_status(motor=i, status=False)  # Disable all motors
        time.sleep(0.3)  # Give ESP32 time to process
    print("Control disabled for all motors")