import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from quadpilot import QuadPilotBody
import time

# Initialize with ESP32's IP address
body = QuadPilotBody("192.168.0.198")

# Set PID parameters
body.set_control_params(0.9, 0.001, 0.3, 10, 5)

# Set target angle to 90 degrees
# body.set_angle(90)

# Change pins if needed
body.set_pins(ENCODER_A=8, ENCODER_B=18, IN1=12, IN2=11)


while True:
    # Get current angle
    start = time.time()
    angle = body.get_angle()
    print(f"Encoder position: {angle}")
    end = time.time()
    # Print time in milliseconds
    print(f"Time: {(end - start) * 1000:.2f} ms")
    # time.sleep(0.5)
    # body.set_angle(45)
    # time.sleep(0.5)
    # body.set_angle(-45)

