import os
import sys
import time
import threading
import socket
import json
from pynput import keyboard

# Add the code directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

# QuadPilotBody class (updated to use UDP)
class QuadPilotBody:
    def __init__(self, ip1="192.168.137.100", ip2="192.168.137.101"):
        """Initialize with two ESP32 IPs, each controlling 4 motors."""
        self.ips = [ip1, ip2]  # ip1 for motors 0-3, ip2 for motors 4-7
        self.UDP_PORT = 12345
        # Create a single UDP socket for sending commands and receiving responses
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(2.0)  # Timeout for responses
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
        # Bind the socket to port 12345 to receive responses
        try:
            self.sock.bind(('0.0.0.0', self.UDP_PORT))
        except Exception as e:
            print(f"Failed to bind socket to port {self.UDP_PORT}: {e}")
            self.sock.close()
            raise
        # Lock for thread-safe socket access
        self.sock_lock = threading.Lock()

    def _get_ip_for_motor(self, motor):
        """Return the IP for the ESP32 controlling the given motor."""
        if not 0 <= motor <= 7:
            raise ValueError("Motor index must be 0-7")
        return self.ips[0] if motor < 4 else self.ips[1]

    def _adjust_motor_index(self, motor):
        """Adjust motor index for the ESP32 (0-3 for each ESP32)."""
        return motor % 4

    def _send_udp_command(self, ip, command_data, retries=3):
        """Send a command over UDP and wait for a response, with thread safety and retries."""
        attempt = 0
        timeout = 2.0  # Timeout per attempt in seconds
        while attempt < retries:
            with self.sock_lock:
                try:
                    message = json.dumps(command_data).encode('utf-8')
                    print(f"Sending to {ip}: {command_data}")
                    self.sock.sendto(message, (ip, self.UDP_PORT))
                    
                    # Keep receiving packets until we get the correct response or timeout
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        try:
                            data, addr = self.sock.recvfrom(1024)
                            response = json.loads(data.decode('utf-8'))
                            print(f"Received from {addr}: {response}")
                            # Check if this is the command response
                            if "status" in response and response["status"] == "OK":
                                return True
                            # If it's an angle update, keep looping
                            else:
                                print(f"Ignoring angle update packet from {addr}")
                        except socket.timeout:
                            break  # Timeout occurred, break inner loop and retry
                        except json.JSONDecodeError:
                            print(f"Received invalid JSON from {addr}, ignoring")
                    
                    print(f"Timeout waiting for response from {ip} for command {command_data['command']} (attempt {attempt + 1}/{retries})")
                    attempt += 1
                    if attempt == retries:
                        return False
                    time.sleep(0.1)  # Short delay before retry
                except Exception as e:
                    print(f"Failed to send command to {ip}: {e}")
                    return False

    def set_control_params(self, P: float, I: float, D: float, dead_zone: int, pos_thresh: int):
        """Set PID parameters, dead zone, and position threshold for all motors on both ESP32s."""
        command = {
            "command": "set_control_params",
            "P": P,
            "I": I,
            "D": D,
            "dead_zone": dead_zone,
            "pos_thresh": pos_thresh
        }
        for ip in self.ips:
            if not self._send_udp_command(ip, command):
                raise Exception(f"Failed to set control parameters for {ip}")

    def set_angles(self, angles: list[int]):
        """Set target angles in degrees for all 8 motors concurrently using threads."""
        if len(angles) != 8:
            raise ValueError("Exactly 8 angles must be provided")
        
        # Prepare commands for each ESP32 (first 4 motors on ip1, next 4 on ip2)
        command1 = {"command": "set_angles", "angles": angles[:4]}
        command2 = {"command": "set_angles", "angles": angles[4:]}
        results = [False, False]  # Track success for each ESP32
        
        def send_request(ip: str, command: dict, index: int):
            """Helper function to send UDP command for a group of motors."""
            results[index] = self._send_udp_command(ip, command)
        
        # Create two threads, one for each group of 4 motors
        thread1 = threading.Thread(target=send_request, args=(self.ips[0], command1, 0))
        thread2 = threading.Thread(target=send_request, args=(self.ips[1], command2, 1))
        
        # Start both threads
        thread1.start()
        thread2.start()
        
        # Wait for both threads to complete
        thread1.join()
        thread2.join()
        
        if not all(results):
            raise Exception("Failed to set angles for one or more ESP32s")

    def set_all_pins(self, pins: list[tuple[int, int, int, int]]):
        """Set encoder and motor pins for all 8 motors at once."""
        if len(pins) != 8:
            raise ValueError("Exactly 8 motor pin configurations must be provided")
        
        # Prepare commands for each ESP32
        command1 = {"command": "set_all_pins"}
        command2 = {"command": "set_all_pins"}
        
        # Pins for first ESP32 (motors 0-3)
        for i, p in enumerate(pins[:4]):
            command1[f"ENCODER_A{i}"] = p[0]
            command1[f"ENCODER_B{i}"] = p[1]
            command1[f"IN1_{i}"] = p[2]
            command1[f"IN2_{i}"] = p[3]
        
        # Pins for second ESP32 (motors 4-7)
        for i, p in enumerate(pins[4:]):
            command2[f"ENCODER_A{i}"] = p[0]
            command2[f"ENCODER_B{i}"] = p[1]
            command2[f"IN1_{i}"] = p[2]
            command2[f"IN2_{i}"] = p[3]
        
        # Send commands to both ESP32s
        if not self._send_udp_command(self.ips[0], command1):
            raise Exception(f"Failed to set pins for {self.ips[0]}")
        if not self._send_udp_command(self.ips[1], command2):
            raise Exception(f"Failed to set pins for {self.ips[1]}")

    def set_control_status(self, motor: int, status: bool):
        """Enable or disable control for a specific motor."""
        ip = self._get_ip_for_motor(motor)
        adjusted_motor = self._adjust_motor_index(motor)
        command = {
            "command": "set_control_status",
            "motor": adjusted_motor,
            "status": 1 if status else 0
        }
        if not self._send_udp_command(ip, command):
            raise Exception(f"Failed to set control status for motor {motor} on {ip}")

    def set_all_control_status(self, status: bool):
        """Enable or disable control for all motors on both ESP32s."""
        command = {
            "command": "set_control_status",
            "motor": 0,
            "status": 1 if status else 0
        }
        for ip in self.ips:
            for i in range(4):
                command["motor"] = i
                if not self._send_udp_command(ip, command):
                    raise Exception(f"Failed to set control status for motor {i} on {ip}")

    def reset_all(self):
        """Reset the encoder positions to 0 for all motors on both ESP32s."""
        command = {"command": "reset_all"}
        for ip in self.ips:
            if not self._send_udp_command(ip, command):
                raise Exception(f"Failed to reset all for {ip}")

    def __del__(self):
        """Cleanup: close the UDP socket."""
        self.sock.close()

# Initialize QuadPilotBody
body = QuadPilotBody()

# Motor indices
NUM_MOTORS = 8
TARGET_MOTORS = [0, 1, 2, 3, 4, 5, 6, 7]  # All motors: Front/Back Left/Right Knee/Hip

# Set initial PID parameters for both ESP32s
print("Setting control parameters...")
try:
    body.set_control_params(0.9, 0, 0.3, 5, 5)  # Updated to match output values
    print("Set control parameters successful")
except Exception as e:
    print(f"Failed to set control parameters: {e}")

# Motor configurations
MOTOR_CONFIGS = [
    (0, "Front Left (Knee)", 39, 40, 41, 42),  # IP1
    (1, "Front Right (Hip)", 37, 38, 1, 2),    # IP1
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

# Initialize only the target motors
for motor_idx in TARGET_MOTORS:
    motor_config = MOTOR_CONFIGS[motor_idx]
    description = motor_config[1]
    print(f"Initializing {description} (Motor {motor_idx})")
    try:
        body.set_control_status(motor=motor_idx, status=True)
        print(f"Control status set to enabled for Motor {motor_idx}")
    except Exception as e:
        print(f"Failed to set control status for Motor {motor_idx}: {e}")
    time.sleep(0.3)

print("Target motors initialized and control enabled!")

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
                # Front Left (Hip)
                angles[3] = -45
                # Front Left (Knee)
                angles[0] = 25  
                # Front Right (Hip)
                angles[1] = 45
                # Front Right (Knee)
                angles[2] = 25
                # Back Right (Hip)
                angles[5] = 45
                # Back Right (Knee)
                angles[4] = -25
                # Back Left (Hip)
                angles[7] = 45
                # Back Left (Knee)
                angles[6] = -25
                set_angles(angles)
            
            # Move the knees 10 degrees
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
    print("\nPress 'a', 'd', 'w', 's' to control motors, 't' to toggle control, or Ctrl+C to exit")
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