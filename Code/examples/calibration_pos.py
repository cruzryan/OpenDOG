import os
import sys
import time
import threading
import socket
import json
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# Add the code directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

# Define the UDP-based QuadPilotBody class
class QuadPilotBody:
    def __init__(self, ip1="192.168.137.100", ip2="192.168.137.101"):
        """Initialize with two ESP32 IPs, each controlling 4 motors."""
        self.ips = [ip1, ip2]  # ip1 for motors 0-3, ip2 for motors 4-7
        self.UDP_PORT = 12345
        # Create a single UDP socket for sending commands and receiving responses/broadcasts
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Use a short timeout for receiving in the command sender (_send_udp_command)
        self.sock.settimeout(0.1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
        # Use SO_REUSEADDR to allow binding even if the port is in TIME_WAIT state
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind the socket to port 12345 to receive responses and broadcasts
        try:
            self.sock.bind(('0.0.0.0', self.UDP_PORT))
            print(f"UDP socket bound to port {self.UDP_PORT}")
        except Exception as e:
            print(f"FATAL: Failed to bind socket to port {self.UDP_PORT}: {e}")
            self.sock.close()
            raise

        # Lock for thread-safe socket access and data structure access
        self.sock_lock = threading.Lock()
        self._is_closed = False # Manual flag to track socket state
        self._stop_listener = threading.Event() # Event to signal listener thread to stop

        # Data structure to store latest received data (protected by lock)
        # Structure: {ip: {'angles': [...], 'encoderPos': [...], 'targetPos': [...]}}
        self._latest_data = {ip: {'angles': [0]*4, 'encoderPos': [0]*4, 'targetPos': [0]*4} for ip in self.ips}
        # Add a timestamp for when data was last updated for an IP
        self._last_data_time = {ip: 0.0 for ip in self.ips}


        # Start the UDP listener thread
        self._listener_thread = threading.Thread(target=self._udp_listener_task, daemon=True)
        self._listener_thread.start()
        print("UDP listener thread started.")


    def _udp_listener_task(self):
        """Thread task to listen for incoming UDP packets (broadcasts and responses)."""
        # print("Listener thread started.") # Verbose start
        while not self._stop_listener.is_set():
            with self.sock_lock:
                 if self._is_closed:
                     # print("Listener exiting: Socket is closed.") # Verbose exit
                     break

                 try:
                    data, addr = self.sock.recvfrom(1024)
                    # print(f"Listener received packet from {addr}: {len(data)} bytes") # Verbose receive

                    try:
                        response = json.loads(data.decode('utf-8'))

                        # Process data broadcasts from known IPs
                        if addr[0] in self.ips:
                            if "angles" in response and "encoderPos" in response and "targetPos" in response:
                                 # Ensure arrays are the correct size before storing
                                 if (len(response.get("angles", [])) == 4 and
                                     len(response.get("encoderPos", [])) == 4 and
                                     len(response.get("targetPos", [])) == 4):
                                     self._latest_data[addr[0]] = {
                                         'angles': response["angles"],
                                         'encoderPos': response["encoderPos"],
                                         'targetPos': response["targetPos"]
                                     }
                                     self._last_data_time[addr[0]] = time.time()
                                     # print(f"Listener stored data from {addr[0]}") # Verbose data storage
                                 # else:
                                     # print(f"Warning: Listener received unexpected array sizes from {addr[0]}") # Verbose warning

                            # Command responses ('OK', errors) are primarily handled by _send_udp_command's recvfrom
                            # If the listener picks one up first, we just ignore it here.

                        # else:
                           # print(f"Listener ignored packet from unknown IP {addr[0]}") # Verbose ignored

                    except json.JSONDecodeError:
                        print(f"Warning: Listener received invalid JSON from {addr}, ignoring")
                    except Exception as e:
                        print(f"Warning: Listener error processing packet from {addr}: {e}")

                 except socket.timeout:
                    # Timeout is expected as we use non-blocking receive
                    pass # Continue loop

                 except OSError as e:
                    # Handle socket errors like "Bad file descriptor" if socket is closed
                    if self._is_closed or self._stop_listener.is_set():
                         # print(f"Listener OS Error after stop signal: {e}") # Verbose error after stop
                         break # Exit loop if stop signal or closed flag is set
                    else:
                        print(f"Warning: Listener OS Error: {e}")
                        # Consider adding a small delay or retry logic on OS errors
                        time.sleep(0.1) # Prevent tight loop on repeated errors


                 except Exception as e:
                    print(f"Warning: Listener caught unexpected exception: {e}")
                    # Consider adding a small delay
                    time.sleep(0.1)

            # Small delay outside the lock to yield if no data was available
            time.sleep(0.001) # Sleep 1ms to prevent 100% CPU usage if no packets

        # print("Listener thread exiting.") # Final exit message


    def get_latest_motor_data(self, motor_index):
        """Retrieve the latest encoder, angle, and target data for a specific motor (0-7) along with a timestamp."""
        if not 0 <= motor_index <= 7:
            raise ValueError("Motor index must be 0-7")

        ip = self._get_ip_for_motor(motor_index)
        adjusted_motor_index = self._adjust_motor_index(motor_index)

        with self.sock_lock:
            esp_data = self._latest_data.get(ip)
            last_update_time = self._last_data_time.get(ip, 0.0)

            if esp_data and len(esp_data['encoderPos']) > adjusted_motor_index:
                 return {
                     'timestamp': last_update_time, # Return the timestamp of the data
                     'angle': esp_data['angles'][adjusted_motor_index],
                     'encoderPos': esp_data['encoderPos'][adjusted_motor_index],
                     'targetPos': esp_data['targetPos'][adjusted_motor_index]
                 }
            else:
                 # Data not yet available or incomplete for this ESP32
                 return None


    def _get_ip_for_motor(self, motor):
        """Return the IP for the ESP32 controlling the given motor."""
        if not 0 <= motor <= 7:
            raise ValueError("Motor index must be 0-7")
        return self.ips[0] if motor < 4 else self.ips[1]

    def _adjust_motor_index(self, motor):
        """Adjust motor index for the ESP32 (0-3 for each ESP32)."""
        return motor % 4

    def _send_udp_command(self, ip, command_data, retries=3, timeout_per_retry=0.5):
        """Send a command over UDP and wait for a specific status response ('OK')."""
        if self._is_closed:
             # print("Attempted to send command but socket is closed.") # Verbose closed
             return False

        attempt = 0
        message = json.dumps(command_data).encode('utf-8')

        while attempt < retries:
            with self.sock_lock:
                # Check closed status again inside lock
                if self._is_closed:
                     # print("Attempted to send command but socket is closed (inside lock).") # Verbose closed
                     return False
                try:
                    # print(f"Sending to {ip}: {command_data}") # Uncomment for verbose sending
                    self.sock.sendto(message, (ip, self.UDP_PORT))

                    # Wait for response using a blocking receive with the specified timeout
                    # The listener thread will NOT pick this up because recvfrom is blocking here.
                    # This is desired for a synchronous command/response model.
                    # The listener is only for non-addressed broadcasts.
                    # The socket timeout set in __init__ applies to this blocking recvfrom.
                    start_wait_time = time.time()
                    while time.time() - start_wait_time < timeout_per_retry:
                        try:
                            # Check closed status again inside inner loop
                            if self._is_closed:
                                 # print("Socket closed during receive in send command.") # Verbose closed
                                 return False

                            data, addr = self.sock.recvfrom(1024)
                            response = json.loads(data.decode('utf-8'))
                            # print(f"Received during send wait from {addr}: {response}") # Verbose receiving

                            # Accept any {'status': 'OK'} from the target IP as success for the last command sent.
                            if response.get("status") == "OK" and addr[0] == ip:
                                # print(f"Received OK from {addr}.") # Verbose success
                                return True
                            # Ignore other packets like angle updates - they are handled by the listener
                            # If we receive something unexpected from the *target IP*, it's still potentially a problem
                            # but for simplicity, we only consider 'OK' from the target IP as success for this specific command.
                            # If we receive data updates from the target IP here, the listener *won't* see them.
                            # This is a limitation of using a single socket for blocking recvfrom and a listener thread.
                            # However, command responses are typically small and quick, so this should usually work.

                        except socket.timeout:
                            # Timeout occurred for recvfrom, break inner loop and try sending again
                            # print(f"Timeout waiting for OK from {ip}") # Verbose timeout
                            break # Break the inner while loop to allow retry

                        except json.JSONDecodeError:
                            print(f"Warning: Received invalid JSON from {addr}, ignoring in send command wait.")
                            # Continue waiting if JSON is bad, might get a good packet next
                            continue # Keep waiting within the timeout

                        except OSError as e:
                             # Handle socket errors like "Bad file descriptor" if socket is closed
                             if self._is_closed:
                                 # print(f"Socket closed during receive in send command (OS Error): {e}") # Verbose OS closed
                                 return False
                             else:
                                 print(f"Warning: OS Error during recvfrom in send command wait on {ip}: {e}")
                                 break # Break retry loop on unexpected OS error

                        except Exception as e:
                            # Catch other potential errors during receive
                            print(f"Warning: Error during recvfrom in send command wait on {ip}: {e}")
                            break # Break retry loop on unexpected errors

                except Exception as e:
                    # Catch errors during sendto
                    print(f"Warning: Failed to send command to {ip} (attempt {attempt + 1}/{retries}): {e}")
                    # Allow retries for network glitches during send
                    pass

            attempt += 1
            # Short delay before next retry if we didn't get a response
            if attempt < retries:
                 time.sleep(0.05) # Give ESP32 a moment

        # Only print error if we exhausted retries and didn't succeed
        print(f"Error: Failed to receive OK status response from {ip} after {retries} attempts.")
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
        print("Setting control parameters...")
        # Increased retries for critical setup, longer timeout per retry
        success1 = self._send_udp_command(self.ips[0], command, retries=5, timeout_per_retry=1.0)
        success2 = self._send_udp_command(self.ips[1], command, retries=5, timeout_per_retry=1.0)
        if not success1 or not success2:
             raise Exception(f"Failed to set control parameters for one or both ESP32s.")
        print("Set control parameters successful.")
        time.sleep(0.1) # Give ESPs time to process


    def set_angles(self, angles: list[int]):
        """Set target angles in degrees for all 8 motors concurrently using threads."""
        if len(angles) != 8:
            raise ValueError("Exactly 8 angles must be provided")

        # Prepare commands for each ESP32 (first 4 motors on ip1, next 4 on ip2)
        # Ensure angles are integers
        angles_int = [int(round(a)) for a in angles]
        command1 = {"command": "set_angles", "angles": angles_int[:4]}
        command2 = {"command": "set_angles", "angles": angles_int[4:]}
        results = [False, False]  # Track success for each ESP32

        def send_request(ip: str, command: dict, index: int):
            """Helper function to send UDP command for a group of motors."""
            # Use minimal retries and timeout for set_angles as it's time-sensitive in sequences
            results[index] = self._send_udp_command(ip, command, retries=1, timeout_per_retry=0.1) # Keep minimal checks

        # Create two threads, one for each group of 4 motors
        thread1 = threading.Thread(target=send_request, args=(self.ips[0], command1, 0), daemon=True)
        thread2 = threading.Thread(target=send_request, args=(self.ips[1], command2, 1), daemon=True)

        # Start both threads
        thread1.start()
        thread2.start()

        # Wait for both threads to complete (with a small timeout)
        thread1.join(timeout=0.2)
        thread2.join(timeout=0.2)

        # We don't necessarily raise an exception here even if it fails, as set_angles might be part of a fast sequence
        # print warnings are handled in _send_udp_command
        # print(f"Set angles result: {results}") # Debug result
        return all(results) # Indicate if both were successful


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

        print("Initializing pins for all motors...")
        success1 = self._send_udp_command(self.ips[0], command1, retries=5, timeout_per_retry=1.0) # Increased retries/timeout
        success2 = self._send_udp_command(self.ips[1], command2, retries=5, timeout_per_retry=1.0) # Increased retries/timeout

        if not success1 or not success2:
            raise Exception(f"Failed to set pins for one or both ESP32s.")
        print("Set pins successful.")
        time.sleep(0.1) # Give ESPs time to process


    def set_control_status(self, motor: int, status: bool):
        """Enable or disable control for a specific motor."""
        ip = self._get_ip_for_motor(motor)
        adjusted_motor = self._adjust_motor_index(motor)
        command = {
            "command": "set_control_status",
            "motor": adjusted_motor,
            "status": 1 if status else 0
        }
        print(f"Setting control status to {status} for motor {motor} (ESP index {adjusted_motor})...")
        success = self._send_udp_command(ip, command, retries=3, timeout_per_retry=0.5) # Retries for status changes
        if not success:
             print(f"Warning: Failed to set control status for motor {motor} on {ip}.")
        time.sleep(0.05) # Small delay after status change
        return success

    def set_all_control_status(self, status: bool):
         """Enable or disable control for all motors on both ESP32s."""
         # This sends commands to each ESP32, telling it to update all 4 motors.
         # It's generally faster than setting each motor individually.
         command = {
             "command": "set_all_control_status",
             "status": 1 if status else 0
         }
         print(f"Setting control status to {status} for all motors...")
         success1 = self._send_udp_command(self.ips[0], command, retries=5, timeout_per_retry=1.0)
         success2 = self._send_udp_command(self.ips[1], command, retries=5, timeout_per_retry=1.0)

         if not success1 or not success2:
             print("Warning: Failed to set control status for all motors on one or both ESP32s.")
             return False
         print("Set all control status complete.")
         time.sleep(0.1) # Give ESPs time to settle
         return True


    def reset_all(self):
        """Reset the encoder positions to 0 for all motors on both ESP32s."""
        command = {"command": "reset_all"}
        print("Resetting all motors...")
        success1 = self._send_udp_command(self.ips[0], command, retries=5, timeout_per_retry=1.0) # Increased retries/timeout
        success2 = self._send_udp_command(self.ips[1], command, retries=5, timeout_per_retry=1.0) # Increased retries/timeout
        if not success1 or not success2:
            raise Exception(f"Failed to reset all for one or both ESP32s.")
        print("Reset all successful.")
        time.sleep(0.1) # Give ESPs time to reset
        return True

    def set_send_interval(self, interval_ms: int):
        """Set the UDP send interval in milliseconds on both ESP32s."""
        if interval_ms <= 0:
            print("Warning: Send interval must be positive. Setting to minimum 1ms.")
            interval_ms = 1

        command = {
            "command": "set_send_interval",
            "interval": interval_ms
        }
        print(f"Setting UDP send interval to {interval_ms} ms...")
        success1 = self._send_udp_command(self.ips[0], command, retries=3, timeout_per_retry=0.5)
        success2 = self._send_udp_command(self.ips[1], command, retries=3, timeout_per_retry=0.5)

        if not success1 or not success2:
            print("Warning: Failed to set send interval for one or both ESP32s.")
            # Continue anyway, but warn the user
            return False
        print("Set send interval successful.")
        time.sleep(0.05) # Small delay
        return True


    def close(self):
        """Explicitly close the UDP socket and mark it as closed."""
        print("Signaling listener thread to stop...")
        self._stop_listener.set() # Signal the listener to stop

        # Wait for the listener thread to finish
        if self._listener_thread.is_alive():
             print("Waiting for listener thread to join...")
             # Use a timeout in case the listener gets stuck
             self._listener_thread.join(timeout=1.0)
             if self._listener_thread.is_alive():
                  print("Warning: Listener thread did not join gracefully.")

        with self.sock_lock:
            if not self._is_closed:
                print("Explicitly closing UDP socket...")
                self.sock.close()
                self._is_closed = True
                print("Socket closed.")
            # else:
                # print("Socket already closed.") # Verbose check

    def __del__(self):
        """Cleanup: close the UDP socket if not already closed."""
        # Note: __del__ is not guaranteed to be called. Rely on explicit close or atexit.
        self.close()


# --- Motor Configurations (from your script) ---
# Ensure the index [0] matches the motor index in the 0-7 list order
MOTOR_CONFIGS = [
    (0, "Front Left (Knee)", 39, 40, 41, 42),  # IP1 (Motor 0)
    (1, "Front Right (Hip)", 37, 38, 1, 2),    # IP1 (Motor 1)
    (2, "Front Right (Knee)", 17, 18, 5, 4),  # IP1 (Motor 2)
    (3, "Front Left (Hip)", 16, 15, 7, 6),    # IP1 (Motor 3)
    (4, "Back Right (Knee)", 37, 38, 1, 2),   # IP2 (Motor 4)
    (5, "Back Right (Hip)", 40, 39, 42, 41),  # IP2 (Motor 5) <-- Target motor for calibration
    (6, "Back Left (Knee)", 15, 16, 6, 7),    # IP2 (Motor 6)
    (7, "Back Left (Hip)", 18, 17, 4, 5),     # IP2 (Motor 7)
]
pin_configs = [(enc_a, enc_b, in1, in2) for _, _, enc_a, enc_b, in1, in2 in MOTOR_CONFIGS]

# Encoder constant (from your Arduino code)
COUNTS_PER_REV = 1975.0 # Use float for division

# Target motor for calibration (BR Hip)
CALIBRATION_MOTOR_INDEX = 5 # Motor 5 is BR Hip according to your config
CALIBRATION_MOTOR_NAME = MOTOR_CONFIGS[CALIBRATION_MOTOR_INDEX][1]

# Calibration Angles
CALIBRATION_ANGLE_NEG_LIMIT = -90.0 # Target beyond expected limit to ensure mechanical stop
CALIBRATION_ANGLE_ZERO = 0.0

# PID parameters for calibration (Adjust if needed)
CAL_KP = 0.9
CAL_KI = 0.0 # Keep I low or zero for calibration to avoid integral windup at limits
CAL_KD = 0.3
CAL_DEAD_ZONE = 5 # Pixels/counts
CAL_POS_THRESHOLD = 5 # Pixels/counts (for PID)

# Parameters for detecting stability and movement from encoder data
STABILITY_WINDOW_SIZE = 20 # Number of recent readings to check for stability
STABILITY_THRESHOLD_COUNTS = 3 # Max allowed difference in counts over the window
MOVEMENT_START_THRESHOLD_COUNTS = 5 # Minimum counts change from the *stable* position to register start of movement
MOVEMENT_DETECTION_WINDOW = 5 # Check for consistent movement over this many readings

# Timeouts for calibration steps
WAIT_FOR_DATA_TIMEOUT = 5.0 # Max time to wait for first data packet
WAIT_FOR_STABILITY_TIMEOUT = 15.0 # Max time to wait for motor to stabilize at a target/limit (increased)


# --- Plotting Setup ---
PLOT_DURATION = 30 # Duration of the rolling plot window in seconds
PLOT_INTERVAL = 50 # Update interval for the plot in milliseconds (20 FPS)

# Deque to store plotting data: (timestamp, encoder_pos, angle, target_pos)
# Max size calculated to hold PLOT_DURATION seconds of data at 1ms interval + some buffer
MAX_PLOT_DATA_POINTS = int(PLOT_DURATION * 1000 / 1) + 100 # Assuming 1ms interval, plus buffer
plot_data = deque(maxlen=MAX_PLOT_DATA_POINTS)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx() # Create a second y-axis sharing the same x-axis

line_encoder, = ax1.plot([], [], label='Encoder Pos (counts)', color='tab:blue')
line_angle, = ax2.plot([], [], label='Current Angle (deg)', color='tab:orange')
line_target, = ax2.plot([], [], label='Target Angle (deg)', color='tab:red', linestyle='--')

ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Encoder Position (counts)", color='tab:blue')
ax2.set_ylabel("Angle (degrees)", color='tab:orange')
plt.title(f"Real-time Motor Data for {CALIBRATION_MOTOR_NAME} (Motor {CALIBRATION_MOTOR_INDEX})")

ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Add a legend that combines elements from both axes
lines = [line_encoder, line_angle, line_target]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

# Set initial y-limits (can be adjusted dynamically if needed)
ax1.set_ylim(-3000, 3000) # Reasonable range for encoder counts
ax2.set_ylim(-100, 100)  # Range for angles


def update_plot(frame):
    """Update function for the matplotlib animation."""
    # Get the latest data from the QuadPilotBody instance
    # The listener thread is continuously updating _latest_data
    # get_latest_motor_data provides a thread-safe snapshot
    data = body.get_latest_motor_data(CALIBRATION_MOTOR_INDEX)

    if data is not None:
        current_time = data['timestamp']
        # Append the data to the deque
        plot_data.append((current_time, data['encoderPos'], data['angle'], data['targetPos']))

        # Extract data for plotting
        times, encoder_pos, angles, target_pos = zip(*plot_data)

        # Plot the data
        # Shift times so the plot starts near 0 seconds
        start_time = times[0] if times else time.time() # Use the first data time or current time
        relative_times = [t - start_time for t in times]

        line_encoder.set_data(relative_times, encoder_pos)
        line_angle.set_data(relative_times, angles)
        line_target.set_data(relative_times, target_pos)

        # Update x-axis limits to create a rolling window
        current_relative_time = relative_times[-1] if relative_times else 0
        ax1.set_xlim(max(0, current_relative_time - PLOT_DURATION), current_relative_time + 1) # Show last X seconds + 1s buffer

        # Optional: Dynamically adjust Y-limits if needed
        # ax1.set_ylim(min(encoder_pos) - 10, max(encoder_pos) + 10)
        # ax2.set_ylim(min(angles + target_pos) - 5, max(angles + target_pos) + 5)


    # Return the lines that were updated for blitting
    return line_encoder, line_angle, line_target


# --- Calibration Helper Functions (using data from QuadPilotBody) ---

# Use a separate deque for stability checks to avoid interference with plot data
stability_history = deque(maxlen=STABILITY_WINDOW_SIZE)

def wait_for_stability(motor_idx, timeout=WAIT_FOR_STABILITY_TIMEOUT):
    """Waits until the motor's encoder position stabilizes within a window."""
    stability_history.clear() # Clear history for a new check
    start_time = time.time()
    print(f"Waiting for motor {motor_idx} to stabilize (max {timeout}s)...")

    last_pos = None
    last_time = time.time()

    while time.time() - start_time < timeout:
        data = body.get_latest_motor_data(motor_idx)
        current_time = time.time() # Use current system time for loop control

        if data is not None and data['timestamp'] > last_time:
             current_pos = data['encoderPos']
             stability_history.append(current_pos)
             last_pos = current_pos # Update last received position
             last_time = data['timestamp'] # Update last data time

             # Check for stability only if we have enough data points in the window
             if len(stability_history) == STABILITY_WINDOW_SIZE:
                 min_pos = min(stability_history)
                 max_pos = max(stability_history)
                 # Check if the range of recent readings is within the stability threshold
                 if (max_pos - min_pos) <= STABILITY_THRESHOLD_COUNTS:
                     print(f"Motor {motor_idx} stable at ~{stability_history[-1]} counts after {current_time - start_time:.2f}s.")
                     return stability_history[-1] # Return the latest stable position

        # Wait a short time to avoid busy-waiting, but frequent enough to catch updates
        time.sleep(0.005) # Wait 5ms

    print(f"Warning: Motor {motor_idx} did not stabilize within {timeout}s. Last known position: {last_pos}")
    # Return the last known position or None if none received at all
    return last_pos if last_pos is not None else None


def wait_for_movement_start(motor_idx, initial_stable_pos, expected_direction_sign, timeout=WAIT_FOR_STABILITY_TIMEOUT):
    """Waits until the motor's encoder position starts moving consistently in the expected direction."""
    start_time = time.time()
    print(f"Waiting for motor {motor_idx} to start moving ({'positive' if expected_direction_sign > 0 else 'negative'}) from ~{initial_stable_pos} (max {timeout}s)...")

    last_pos = initial_stable_pos
    last_time = time.time()
    movement_count = 0 # Counter for consecutive movements in the expected direction

    # Give the command a moment to propagate and motor to potentially reverse into backlash/start moving
    time.sleep(0.05)

    while time.time() - start_time < timeout:
         data = body.get_latest_motor_data(motor_idx)
         current_time = time.time() # Use current system time for loop control

         if data is not None and data['timestamp'] > last_time:
             current_pos = data['encoderPos']
             pos_change = current_pos - last_pos

             # Check for a significant change from the *initial stable position*
             # This handles cases where the motor might dither slightly before starting bulk movement
             if abs(current_pos - initial_stable_pos) >= MOVEMENT_START_THRESHOLD_COUNTS:
                 # Check if the most recent movement is in the expected direction
                 if (current_pos - last_pos) * expected_direction_sign > 0:
                     movement_count += 1 # Increment counter for consecutive movement
                 else:
                     movement_count = 0 # Reset if direction reverses or stops

                 # If we've seen consistent movement in the expected direction
                 if movement_count >= MOVEMENT_DETECTION_WINDOW:
                     print(f"Motor {motor_idx} started moving ({pos_change:+d} counts current change) from ~{last_pos} at ~{current_pos} (from stable {initial_stable_pos}) after {current_time - start_time:.2f}s.")
                     return current_pos # Return the position when consistent movement was detected

             last_pos = current_pos # Update last position for the next check
             last_time = data['timestamp'] # Update last data time

         time.sleep(0.005) # Wait 5ms

    print(f"Warning: Motor {motor_idx} did not detect clear movement start within {timeout}s. Last known position: {last_pos}")
    return last_pos # Return the last known position if movement wasn't detected


# --- Calibration Sequence ---
def calibrate_br_hip(body: QuadPilotBody):
    """Performs the backlash calibration procedure for the Back Right Hip motor."""

    motor_index = CALIBRATION_MOTOR_INDEX
    motor_name = CALIBRATION_MOTOR_NAME
    print(f"\n--- Starting Calibration for {motor_name} (Motor {motor_index}) ---")

    # Ensure the plotting starts before calibration commands are sent
    # The calibration thread will wait for data before proceeding anyway.
    # The main thread handles plt.show() and animation.

    # --- Initial Setup ---
    try:
        print("Step 1/?: Applying calibration control parameters...")
        body.set_control_params(CAL_KP, CAL_KI, CAL_KD, CAL_DEAD_ZONE, CAL_POS_THRESHOLD)
        print("Step 2/?: Setting all motor pins...")
        body.set_all_pins(pin_configs)
        print("Step 3/?: Resetting all encoder positions...")
        # Resetting before enabling control might be safer depending on ESP code
        body.reset_all()
        time.sleep(1) # Give ESP32s time to process reset and settle

        print("Step 4/?: Setting UDP send interval to 1ms...")
        body.set_send_interval(1) # Request 1ms update rate

        print(f"Step 5/?: Enabling control for {motor_name} (Motor {motor_index}) only...")
        # Best practice: Disable all first, then enable only the target motor for safety during calibration
        body.set_all_control_status(False) # Disable all
        # Give ESPs a moment to process disabling all
        time.sleep(0.5)
        body.set_control_status(motor=motor_index, status=True) # Enable target motor
        time.sleep(1) # Give time for control to be enabled and motor to home/settle near 0 or last pos

    except Exception as e:
        print(f"FATAL: Initial setup failed: {e}")
        # Calibration failed, signal plot or main thread to exit?
        # For now, just print error and return False
        return False # Indicate failure


    # --- Wait for initial data packet ---
    print("Waiting for initial encoder data packet from ESP32s...")
    start_wait_time = time.time()
    initial_data = None
    while time.time() - start_wait_time < WAIT_FOR_DATA_TIMEOUT:
        initial_data = body.get_latest_motor_data(motor_index)
        if initial_data is not None:
            print(f"Received initial data. Current position: {initial_data['encoderPos']} counts / {initial_data['angle']:.1f} deg")
            # Add initial data to plot deque manually if animation hasn't started or missed it
            # The animation update should pick it up shortly after this loop finishes
            break
        time.sleep(0.01) # Wait 10ms between checks

    if initial_data is None:
        print(f"FATAL: Timed out waiting for encoder data from motor {motor_index}. Is the ESP32 powered and connected?")
        # Attempt to disable control before returning
        try:
             body.set_control_status(motor=motor_index, status=False)
        except:
             pass
        return False

    # --- Calibration Movements ---

    pos_at_neg_limit_1 = None
    pos_start_move_to_zero = None
    pos_at_zero_limit = None
    pos_start_move_to_neg90 = None
    pos_at_neg_limit_2 = None

    # Movement 1: Go to negative limit (-90 deg)
    print(f"\nStep 6/?: Moving {motor_name} to {CALIBRATION_ANGLE_NEG_LIMIT} degrees (towards negative limit)...")
    # Set target, but expect to hit the mechanical limit before reaching the target encoder count
    target_angles = [0] * 8
    target_angles[motor_index] = int(round(CALIBRATION_ANGLE_NEG_LIMIT))
    body.set_angles(target_angles)

    # Wait until the motor stabilizes at the mechanical limit
    pos_at_neg_limit_1 = wait_for_stability(motor_index, timeout=WAIT_FOR_STABILITY_TIMEOUT + 5) # Give extra time to hit limit

    if pos_at_neg_limit_1 is None:
         print("FATAL: Failed to reach negative limit or motor did not stabilize. Calibration cannot continue.")
         # Attempt to disable control before returning
         try:
              body.set_control_status(motor=motor_index, status=False)
         except:
              pass
         return False
    print(f"Negative limit 1 reached (approx): {pos_at_neg_limit_1} counts")


    # Movement 2: Go to zero (0 deg) - Detecting backlash from the negative side
    print(f"\nStep 7/?: Moving {motor_name} to {CALIBRATION_ANGLE_ZERO} degrees (detecting backlash from negative side)...")
    # Set target to 0. The motor will first move positively within the backlash.
    target_angles = [0] * 8
    target_angles[motor_index] = int(round(CALIBRATION_ANGLE_ZERO))
    body.set_angles(target_angles)

    # Wait until movement *in the positive direction* is detected
    # Use pos_at_neg_limit_1 as the reference for detecting movement start
    pos_start_move_to_zero = wait_for_movement_start(motor_index, initial_stable_pos=pos_at_neg_limit_1, expected_direction_sign=1, timeout=WAIT_FOR_STABILITY_TIMEOUT)

    if pos_start_move_to_zero is None:
         print("Warning: Failed to detect movement start towards zero. Cannot calculate backlash from negative side.")
         # Allow it to continue to find the zero position, but backlash measurement will be incomplete.
         pass
    else:
         print(f"Movement towards zero detected at (approx): {pos_start_move_to_zero} counts")


    # Now wait for it to stabilize at the zero position (which is a controlled target)
    pos_at_zero_limit = wait_for_stability(motor_index)

    if pos_at_zero_limit is None:
         print("FATAL: Failed to reach zero position or motor did not stabilize. Calibration cannot continue.")
         # Attempt to disable control before returning
         try:
              body.set_control_status(motor=motor_index, status=False)
         except:
              pass
         return False
    print(f"Zero position reached (approx): {pos_at_zero_limit} counts")


    # Movement 3: Go back to negative limit (-90 deg) - Detecting backlash from the zero side
    print(f"\nStep 8/?: Moving {motor_name} back to {CALIBRATION_ANGLE_NEG_LIMIT} degrees (detecting backlash from zero side)...")
    # Set target back to -90. The motor will first move negatively within the backlash.
    target_angles = [0] * 8
    target_angles[motor_index] = int(round(CALIBRATION_ANGLE_NEG_LIMIT))
    body.set_angles(target_angles)

    # Wait until movement *in the negative direction* is detected
    # Use pos_at_zero_limit as the reference for detecting movement start
    pos_start_move_to_neg90 = wait_for_movement_start(motor_index, initial_stable_pos=pos_at_zero_limit, expected_direction_sign=-1, timeout=WAIT_FOR_STABILITY_TIMEOUT)

    if pos_start_move_to_neg90 is None:
        print("Warning: Failed to detect movement start back towards negative limit. Cannot calculate backlash from zero side.")
        pass # Allow it to continue

    # Now wait for it to stabilize back at the negative mechanical limit
    pos_at_neg_limit_2 = wait_for_stability(motor_index, timeout=WAIT_FOR_STABILITY_TIMEOUT + 5) # Give extra time to hit limit again

    if pos_at_neg_limit_2 is None:
        print("Warning: Failed to reach negative limit again. Backlash measurement might be inaccurate.")
        # Use pos_at_neg_limit_1 for calculation if pos_at_neg_limit_2 is None
        pos_at_neg_limit_2 = pos_at_neg_limit_1
    else:
        print(f"Negative limit 2 reached (approx): {pos_at_neg_limit_2} counts")


    # --- Calculate and Report Backlash ---
    print("\n--- Calibration Results ---")

    backlash_neg_to_zero = None
    # Backlash when moving from negative limit towards zero (positive direction movement)
    # It's the distance moved *into* the backlash zone from the limit until free movement starts.
    # The stable position at the limit (pos_at_neg_limit_1) is one edge of the backlash.
    # The position where movement starts (pos_start_move_to_zero) is the other edge.
    # Backlash is the difference between these points.
    if pos_at_neg_limit_1 is not None and pos_start_move_to_zero is not None:
        backlash_neg_to_zero = pos_start_move_to_zero - pos_at_neg_limit_1
        print(f"Backlash (Moving -90 -> 0): {backlash_neg_to_zero} counts")
        print(f"                          ({backlash_neg_to_zero / COUNTS_PER_REV * 360:.2f} degrees)")
    else:
        print("Backlash (Moving -90 -> 0): Could not measure (failed to detect movement start)")


    backlash_zero_to_neg = None
    # Backlash when moving from zero towards negative limit (negative direction movement)
    # It's the distance moved *into* the backlash zone from the zero position until free movement starts.
    # The stable position at zero (pos_at_zero_limit) is one edge of the backlash.
    # The position where movement starts (pos_start_move_to_neg90) is the other edge.
    # Backlash is the difference between these points (positive difference).
    if pos_at_zero_limit is not None and pos_start_move_to_neg90 is not None:
        backlash_zero_to_neg = pos_at_zero_limit - pos_start_move_to_neg90 # Assuming pos_start_move_to_neg90 < pos_at_zero_limit
        print(f"Backlash (Moving 0 -> -90): {backlash_zero_to_neg} counts")
        print(f"                         ({backlash_zero_to_neg / COUNTS_PER_REV * 360:.2f} degrees)")
    else:
        print("Backlash (Moving 0 -> -90): Could not measure (failed to detect movement start)")

    # Interpretation and Recommendation
    measured_backlash_counts = []
    if backlash_neg_to_zero is not None and backlash_neg_to_zero >= 0: # Ensure positive value
        measured_backlash_counts.append(backlash_neg_to_zero)
    if backlash_zero_to_neg is not None and backlash_zero_to_neg >= 0: # Ensure positive value
        measured_backlash_counts.append(backlash_zero_to_neg)

    if measured_backlash_counts:
        average_backlash_counts = sum(measured_backlash_counts) / len(measured_backlash_counts)
        recommended_dead_zone = int(math.ceil(average_backlash_counts / 2.0)) # Dead zone is typically half the backlash
        print(f"\nAverage measured backlash: {average_backlash_counts:.1f} counts")
        print(f"                         ({average_backlash_counts / COUNTS_PER_REV * 360:.2f} degrees)")
        print(f"\nRecommended Dead Zone setting: {recommended_dead_zone} counts")
        print("Consider using this value for the 'dead_zone' parameter in your PID control.")
    else:
        print("\nCould not measure backlash in either direction.")


    print("\n--- Calibration Finished ---")
    # Ensure the calibrated motor is disabled after calibration finishes
    try:
         print(f"Disabling control for {motor_name} (Motor {motor_index})...")
         body.set_control_status(motor=motor_index, status=False)
         time.sleep(0.1) # Give time for command to send
    except Exception as cleanup_e:
         print(f"Error disabling motor control after calibration: {cleanup_e}")

    # The calibration thread simply finishes here. The main thread running plt.show() continues until closed.
    # Cleanup happens when plt.show() returns.
    return True # Indicate calibration procedure completed


# --- Main Execution ---
if __name__ == "__main__":
    body = None # Initialize body to None
    cal_thread = None # Initialize calibration thread to None

    try:
        print("Initializing QuadPilotBody...")
        body = QuadPilotBody()
        print("Initialization complete.")

        # Start the Matplotlib animation *before* the calibration thread
        # This ensures the plot window is open and the update function is called
        # while calibration is running.
        ani = animation.FuncAnimation(fig, update_plot, interval=PLOT_INTERVAL, blit=True, cache_frame_data=False)

        # Start the calibration procedure in a separate thread
        # This allows the main thread to run plt.show() and handle the plotting loop
        print("Starting calibration procedure in background thread...")
        cal_thread = threading.Thread(target=calibrate_br_hip, args=(body,), daemon=True) # Use daemon=True so it exits with main
        cal_thread.start()

        print("\nStarting plot window. Calibration steps will be printed to console.")
        print(f"Monitor the plot for {CALIBRATION_MOTOR_NAME} (Motor {CALIBRATION_MOTOR_INDEX}) data.")
        print("Close the plot window to exit the program.")

        # Display the plot window (this call blocks the main thread)
        plt.show()

    except Exception as e:
        print(f"\nFATAL: An error occurred: {e}")

    finally:
        # This block runs when plt.show() returns (user closes the window) or if a fatal error occurred
        print("\nCleaning up...")
        if cal_thread and cal_thread.is_alive():
            print("Waiting briefly for calibration thread to finish...")
            cal_thread.join(timeout=1.0) # Wait a moment, but don't block exit indefinitely

        if body:
            try:
                 # The calibrate function disables the motor at the end if successful.
                 # This final close ensures the socket and listener thread are properly shut down.
                 print("Closing UDP socket and stopping listener...")
                 body.close() # Use the explicit close method
            except Exception as cleanup_e:
                 print(f"Error during body cleanup: {cleanup_e}")
        else:
            print("Body object not initialized, no cleanup needed.")

        print("Cleanup complete. Exiting.")