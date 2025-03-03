import os
import sys
import time
from sseclient import SSEClient

script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from quadpilot import QuadPilotBody

# Initialize with ESP32's IP address
body = QuadPilotBody("192.168.0.198")

# Set PID parameters
body.set_control_params(0.9, 0.001, 0.3, 10, 5)

# Set target angle to 90 degrees (uncomment if needed)
# body.set_angle(90)

# Change pins if needed
body.set_pins(ENCODER_A=8, ENCODER_B=18, IN1=12, IN2=11)

# Connect to the SSE endpoint
url = "http://192.168.0.198:82/events"
messages = SSEClient(url)

# Process real-time angle updates
last_time = time.time()
for msg in messages:
    try:
        angle = float(msg.data)
        current_time = time.time()
        print(f"Encoder position: {angle}")
        print(f"Time since last update: {(current_time - last_time) * 1000:.2f} ms")
        last_time = current_time
    except ValueError:
        print("Invalid data received")