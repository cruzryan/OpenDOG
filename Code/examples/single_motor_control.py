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

# Motor index (using motor 0)
MOTOR_INDEX = 0

# Set initial PID parameters and pins for motor 0
body.set_control_params(0.9, 0.001, 0.3, 10, 5)  # Applies to all motors, no motor param needed
body.set_pins(motor=MOTOR_INDEX, ENCODER_A=8, ENCODER_B=18, IN1=12, IN2=11)
body.set_control_status(motor=MOTOR_INDEX, status=True)  # Enable control for motor 0

# Debounce mechanism variables
last_key_press_time = 0
debounce_interval = 0.2  # seconds

# SSE handling variables
sse_active = threading.Event()
sse_active.set()  # Start as active
sse_thread = None
lock = threading.Lock()

# Variables for rolling window average
angle_window = []
encoder_window = []
WINDOW_SIZE = 10

# Function to set angle for motor 0
def set_angle(angle):
    global last_key_press_time
    current_time = time.time()
    if current_time - last_key_press_time > debounce_interval:
        last_key_press_time = current_time
        print(f"\nSetting motor {MOTOR_INDEX} angle to {angle} degrees")
        # Pause SSE briefly to allow the set_angle request to go through
        with lock:
            sse_active.clear()  # Signal SSE thread to pause
            time.sleep(0.05)  # Brief pause to let SSE loop yield
            body.set_angle(motor=MOTOR_INDEX, a=angle)  # Updated API with motor param
            sse_active.set()  # Resume SSE

# SSE processing function (updated for multi-motor JSON format)
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
                        # Extract data for motor 0 (assuming NUM_MOTORS >= 1)
                        angle = data["angles"][MOTOR_INDEX]
                        encoder_pos = data["encoderPos"][MOTOR_INDEX]
                        current_time = time.time()
                        
                        # Update rolling windows
                        angle_window.append(angle)
                        encoder_window.append(encoder_pos)
                        if len(angle_window) > WINDOW_SIZE:
                            angle_window.pop(0)
                        if len(encoder_window) > WINDOW_SIZE:
                            encoder_window.pop(0)
                            
                        # Calculate averages
                        avg_angle = sum(angle_window) / len(angle_window)
                        avg_encoder = sum(encoder_window) / len(encoder_window)
                        
                        # Print averages with carriage return to avoid screen clearing
                        print(f"\rMotor {MOTOR_INDEX} - Average Angle: {avg_angle:.2f}, Average Encoder Position: {avg_encoder:.2f}", end="")
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
            set_angle(45)
        elif key.char == 'd':
            set_angle(-45)
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
    body.set_control_status(motor=MOTOR_INDEX, status=False)  # Disable motor control on exit