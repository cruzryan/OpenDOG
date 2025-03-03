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

from quadpilot import QuadPilotBody

# Initialize with ESP32's IP address
body = QuadPilotBody("192.168.0.198")

# Set initial PID parameters and pins
body.set_control_params(0.9, 0.001, 0.3, 10, 5)
body.set_pins(ENCODER_A=8, ENCODER_B=18, IN1=12, IN2=11)

# Debounce mechanism variables
last_key_press_time = 0
debounce_interval = 0.2  # seconds

# SSE handling variables
sse_active = threading.Event()
sse_active.set()  # Start as active
sse_thread = None
lock = threading.Lock()

# Function to set angle
def set_angle(angle):
    global last_key_press_time
    current_time = time.time()
    if current_time - last_key_press_time > debounce_interval:
        last_key_press_time = current_time
        print(f"Setting angle to {angle} degrees")
        # Pause SSE briefly to allow the set_angle request to go through
        with lock:
            sse_active.clear()  # Signal SSE thread to pause
            time.sleep(0.05)  # Brief pause to let SSE loop yield
            body.set_angle(angle)  # Make the HTTP GET request
            sse_active.set()  # Resume SSE

# SSE processing function
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
                        angle = data["angle"]
                        encoder_pos = data["encoderPos"]
                        current_time = time.time()
                        print(f"Angle: {angle:.2f}, Encoder Position: {encoder_pos}")
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