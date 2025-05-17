import os
import sys
import time

# Add the code directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from quadpilot import QuadPilotBody

# Initialize QuadPilotBody
body = QuadPilotBody()

# Motor indices
NUM_MOTORS = 8
TARGET_MOTORS = [0, 1, 2, 3, 4, 5, 6, 7]  # All motors: Front/Back Left/Right Knee/Hip

# Set initial PID parameters for both ESP32s
body.set_control_params(4.0, 0, 0.3, 5, 5)
print("Set control parameters")

# Motor configurations
MOTOR_CONFIGS = [
    (0, "Front Left (Knee)", 39, 40, 41, 42),  # IP1
    (1, "Front Right (Hip)", 37, 38, 1, 2),   # IP1
    (2, "Front Right (Knee)", 17, 18, 5, 4),  # IP1
    (3, "Front Left (Hip)", 16, 15, 7, 6),    # IP1
    (4, "Back Right (Knee)", 37, 38, 1, 2),   # IP2
    (5, "Back Right (Hip)", 40, 39, 42, 41),  # IP2
    (6, "Back Left (Knee)", 15, 16, 6, 7),    # IP2
    (7, "Back Left (Hip)", 18, 17, 4, 5),     # IP2
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