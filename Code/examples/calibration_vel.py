import os
import sys
import time
import threading
import socket
import json
import math # Import math for abs
import collections # Import collections for deque

# Add the code directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

# Define the UDP-based QuadPilotBody class (MODIFIED FOR LISTENING)
class QuadPilotBody:
    def __init__(self, ip1="192.168.137.100", ip2="192.168.137.101"):
        """Initialize with two ESP32 IPs, each controlling 4 motors."""
        self.ips = [ip1, ip2]  # ip1 for motors 0-3, ip2 for motors 4-7
        self.UDP_PORT = 12345
        # Create a single UDP socket for sending commands and receiving responses/broadcasts
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Use a non-blocking socket or a short timeout to avoid hanging recvfrom
        self.sock.settimeout(0.1) # Timeout for receiving in the command sender
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

        # Store a short history of (timestamp, encoder_position) for each motor for velocity calculation
        # Access this via get_latest_motor_data or a new dedicated velocity function
        self._encoder_history = {motor_idx: collections.deque(maxlen=20) for motor_idx in range(8)} # Store up to 20 recent points per motor


        # Start the UDP listener thread
        self._listener_thread = threading.Thread(target=self._udp_listener_task, daemon=True)
        self._listener_thread.start()
        print("UDP listener thread started.")


    def _udp_listener_task(self):
        """Thread task to listen for incoming UDP packets (broadcasts and responses)."""
        while not self._stop_listener.is_set():
            with self.sock_lock:
                 # Check if socket is closed before attempting recvfrom
                 if self._is_closed:
                     # print("Listener exiting: Socket is closed.") # Verbose listener exit
                     break # Exit loop if socket is closed

                 try:
                    data, addr = self.sock.recvfrom(1024)
                    # print(f"Listener received packet from {addr}: {len(data)} bytes") # Verbose listener receive

                    try:
                        response = json.loads(data.decode('utf-8'))
                        current_time = time.time() # Timestamp when packet was received

                        # Process different types of packets
                        if "angles" in response and "encoderPos" in response and "targetPos" in response:
                             # This looks like a periodic update from an ESP32
                             # Store the latest data, matching IP to the correct ESP32
                             if addr[0] in self.ips:
                                 esp_index = self.ips.index(addr[0])
                                 # Ensure arrays are the correct size before storing
                                 if (len(response["angles"]) == 4 and
                                     len(response["encoderPos"]) == 4 and
                                     len(response["targetPos"]) == 4):
                                     self._latest_data[addr[0]] = {
                                         'angles': response["angles"],
                                         'encoderPos': response["encoderPos"],
                                         'targetPos': response["targetPos"]
                                     }
                                     # Update encoder history for each motor on this ESP
                                     motor_offset = esp_index * 4
                                     for i in range(4):
                                         global_motor_index = motor_offset + i
                                         if global_motor_index < 8: # Just a safety check
                                              self._encoder_history[global_motor_index].append((current_time, response["encoderPos"][i]))

                                     # print(f"Listener stored data from {addr[0]}") # Verbose data storage
                                 else:
                                     print(f"Warning: Listener received unexpected array sizes from {addr[0]}")
                             # else:
                                # print(f"Listener ignored broadcast from unknown IP {addr[0]}") # Verbose ignored

                        elif "status" in response:
                             # This looks like a command response ('OK' or error).
                             # The _send_udp_command method will handle this via its recvfrom timeout loop.
                             # We don't need to do anything specific with it in the listener,
                             # but processing it here prevents the listener from seeing it first
                             # and the command sender timing out. However, the _send_udp_command
                             # uses the *same* socket, so recvfrom *there* will pick it up.
                             # It's safer to let _send_udp_command handle its own synchronous response.
                             # So, if it's not a data broadcast, we just ignore it in the listener.
                             pass # Let _send_udp_command handle command responses


                        else:
                             # Received something else
                             # print(f"Listener received unhandled packet from {addr}: {response}") # Verbose unhandled
                             pass # Ignore other packets

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
            time.sleep(0.0001) # Sleep 1ms to prevent 100% CPU usage if no packets

        # print("Listener thread exiting.") # Final exit message


    def get_latest_motor_data(self, motor_index):
        """Retrieve the latest encoder, angle, and target data for a specific motor (0-7)."""
        if not 0 <= motor_index <= 7:
            raise ValueError("Motor index must be 0-7")

        ip = self._get_ip_for_motor(motor_index)
        adjusted_motor_index = self._adjust_motor_index(motor_index)

        with self.sock_lock:
            # Return a copy of the data for the specific motor
            # Need to be careful if data for this IP hasn't arrived yet.
            # Return None or default values in that case.
            esp_data = self._latest_data.get(ip)
            if esp_data and len(esp_data['encoderPos']) > adjusted_motor_index:
                 return {
                     'angle': esp_data['angles'][adjusted_motor_index],
                     'encoderPos': esp_data['encoderPos'][adjusted_motor_index],
                     'targetPos': esp_data['targetPos'][adjusted_motor_index]
                 }
            else:
                 # Data not yet available or incomplete for this ESP32
                 return None

    def get_smoothed_velocity(self, motor_index, window_size):
         """Calculate smoothed encoder velocity (counts/sec) for a motor."""
         if not 0 <= motor_index <= 7:
             raise ValueError("Motor index must be 0-7")
         if window_size <= 1:
             return 0.0 # Cannot calculate velocity with less than 2 points

         with self.sock_lock:
             history = list(self._encoder_history[motor_index]) # Get a copy of the history

         if len(history) < window_size:
             return 0.0 # Not enough data points

         # Calculate velocities between consecutive points in the window
         velocities = []
         # Use the last 'window_size' points, so history[-window_size] is the oldest relevant point
         # We need window_size-1 velocity calculations from window_size points
         for i in range(len(history) - window_size, len(history) - 1):
             t1, p1 = history[i]
             t2, p2 = history[i+1]
             dt_sec = t2 - t1
             if dt_sec > 0: # Avoid division by zero
                 velocities.append((p2 - p1) / dt_sec)

         if not velocities:
             return 0.0 # Should not happen if history >= window_size > 1, but safety check

         # Return the average of these velocities
         return sum(velocities) / len(velocities)


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
             print("Attempted to send command but socket is closed.")
             return False

        attempt = 0
        message = json.dumps(command_data).encode('utf-8')
        # Note: This method relies on the synchronous recvfrom timeout for confirmation.
        # The listener thread is also running and might pick up broadcasts first.
        # We rely on the target IP check (`addr[0] == ip`) to filter responses.

        while attempt < retries:
            with self.sock_lock:
                # Check closed status again inside lock
                if self._is_closed:
                     print("Attempted to send command but socket is closed (inside lock).")
                     return False
                try:
                    # print(f"Sending to {ip}: {command_data}") # Uncomment for verbose sending
                    self.sock.sendto(message, (ip, self.UDP_PORT))

                    start_time = time.time()
                    while time.time() - start_time < timeout_per_retry:
                        try:
                            # Check closed status again inside inner loop
                            if self._is_closed:
                                 # print("Socket closed during receive in send command.") # Verbose closed
                                 return False

                            # Use the socket timeout set in __init__
                            data, addr = self.sock.recvfrom(1024)
                            response = json.loads(data.decode('utf-8'))
                            # print(f"Received during send wait from {addr}: {response}") # Verbose receiving

                            # --- FIXED RESPONSE CHECK ---
                            # Accept any {'status': 'OK'} from the target IP as success for the last command sent.
                            if response.get("status") == "OK" and addr[0] == ip:
                                # print(f"Received OK from {addr}.") # Verbose success
                                return True
                            # Ignore other packets like angle updates - they are handled by the listener

                        except socket.timeout:
                            # Timeout occurred for recvfrom, break inner loop and try sending again
                            # print(f"Timeout waiting for OK from {ip}") # Verbose timeout
                            break
                        except json.JSONDecodeError:
                            print(f"Warning: Received invalid JSON from {addr}, ignoring in send command wait.")
                        except OSError as e:
                             # Handle socket errors like "Bad file descriptor" if socket is closed
                             if self._is_closed:
                                 # print("Socket closed during receive in send command (OS Error).") # Verbose OS closed
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
            # send_udp_command waits for 'OK'. set_angles doesn't necessarily need confirmation per step.
            # However, for reliability, let's keep a minimal check.
            results[index] = self._send_udp_command(ip, command, retries=1, timeout_per_retry=0.1) # Keep minimal checks

        # Create two threads, one for each group of 4 motors
        thread1 = threading.Thread(target=send_request, args=(self.ips[0], command1, 0), daemon=True)
        thread2 = threading.Thread(target=send_request, args=(self.ips[1], command2, 1), daemon=True)

        # Start both threads
        thread1.start()
        thread2.start()

        # Wait for both threads to complete (with a small timeout)
        # The set_angles command should be quick.
        thread1.join(timeout=0.2)
        thread2.join(timeout=0.2)

        # We don't raise an exception here even if it fails, as set_angles might be part of a fast sequence
        # print warnings are handled in _send_udp_command
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
         # Use a list of threads for concurrent updates to all 8 motors
         threads = []
         results = [False] * 8 # Track results for each motor

         def send_single_status(motor_idx, status_val, result_idx):
             ip = self._get_ip_for_motor(motor_idx)
             adjusted_motor = self._adjust_motor_index(motor_idx)
             cmd = {"command": "set_control_status", "motor": adjusted_motor, "status": status_val}
             results[result_idx] = self._send_udp_command(ip, cmd, retries=3, timeout_per_retry=0.5)
             # print(f"Status for motor {motor_idx} ({results[result_idx]})") # Debug status setting

         status_val = 1 if status else 0
         print(f"Setting control status to {status} for motors 0-7 individually...")
         for i in range(8):
             thread = threading.Thread(target=send_single_status, args=(i, status_val, i), daemon=True)
             threads.append(thread)
             thread.start()

         # Wait for all threads to complete
         for thread in threads:
             thread.join(timeout=0.5) # Wait briefly for each status command to process

         if not all(results):
             print("Warning: Failed to set control status for one or more motors.")
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
# Keep these as they are from the new API snippet
# Ensure the index [0] matches the motor index in the 0-7 list order
MOTOR_CONFIGS = [
    (0, "Front Left (Knee)", 39, 40, 41, 42),  # IP1 (Motor 0)
    (1, "Front Right (Hip)", 37, 38, 1, 2),    # IP1 (Motor 1)
    (2, "Front Right (Knee)", 17, 18, 5, 4),  # IP1 (Motor 2)
    (3, "Front Left (Hip)", 16, 15, 7, 6),    # IP1 (Motor 3)
    (4, "Back Right (Knee)", 37, 38, 1, 2),   # IP2 (Motor 4)
    (5, "Back Right (Hip)", 40, 39, 42, 41),  # IP2 (Motor 5) <-- Target motor
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
CALIBRATION_ANGLE_NEG_LIMIT = -90.0 # Target beyond expected limit
CALIBRATION_ANGLE_ZERO = 0.0

# PID parameters for calibration (Adjust if needed)
CAL_KP = 0.9
CAL_KI = 0.0 # Keep I low or zero for calibration to avoid integral windup at limits
CAL_KD = 0.3
CAL_DEAD_ZONE = 5 # Pixels/counts - This will be *overridden* by backsplash measurement, but set reasonably
CAL_POS_THRESHOLD = 5 # Pixels/counts - This affects when controller considers 'on target'

# Parameters for detecting stability and movement from encoder data
STABILITY_WINDOW_SIZE = 10 # Number of recent readings to check for stability
STABILITY_THRESHOLD_COUNTS = 2 # Max allowed difference in counts over the window

# Parameters for velocity-based movement start detection
VELOCITY_SMOOTHING_WINDOW = 8 # Number of recent velocity samples to average
EXPECTED_VELOCITY_THRESHOLD = 75 # Minimum smoothed velocity (counts/sec) to register movement start (Tune this!)

# Timeouts for calibration steps
WAIT_FOR_DATA_TIMEOUT = 5.0 # Max time to wait for first data packet
WAIT_FOR_STABILITY_TIMEOUT = 10.0 # Max time to wait for motor to stabilize at a target/limit
WAIT_FOR_MOVEMENT_TIMEOUT = 10.0 # Max time to wait for movement to start


# --- Calibration Function ---
def calibrate_br_hip(body: QuadPilotBody):
    """Performs the backlash calibration procedure for the Back Right Hip motor."""

    motor_index = CALIBRATION_MOTOR_INDEX
    motor_name = CALIBRATION_MOTOR_NAME
    print(f"\n--- Starting Calibration for {motor_name} (Motor {motor_index}) ---")

    # --- Initial Setup ---
    try:
        print("Step 1/?: Applying calibration control parameters...")
        # Apply calibration PID, but DEAD_ZONE/POS_THRESHOLD might be less critical for backlash finding
        body.set_control_params(CAL_KP, CAL_KI, CAL_KD, CAL_DEAD_ZONE, CAL_POS_THRESHOLD)
        print("Step 2/?: Setting all motor pins...")
        body.set_all_pins(pin_configs)
        print("Step 3/?: Resetting all encoder positions...")
        body.reset_all()
        time.sleep(1) # Give ESP32s time to process reset and settle

        print("Step 4/?: Setting UDP send interval to 1ms...")
        body.set_send_interval(1) # Request 1ms update rate
        time.sleep(0.5) # Give ESP32s time to start sending faster

        print(f"Step 5/?: Enabling control for {motor_name} (Motor {motor_index})...")
        # Ensure only the target motor is enabled if possible, or enable all if necessary
        # For this calibration, enabling only the target motor is best practice
        # Disable all first is a robust approach if you're unsure of current state
        # body.set_all_control_status(False) # Optional: ensure all are off
        body.set_control_status(motor=motor_index, status=True)
        time.sleep(1.5) # Give time for control to be enabled and motor to potentially home/settle


    except Exception as e:
        print(f"FATAL: Initial setup failed: {e}")
        return False # Indicate failure

    # --- Wait for first data packet ---
    print("Waiting for initial encoder data packet from ESP32s...")
    start_wait_time = time.time()
    initial_data = None
    # Loop until data is NOT None AND the encoder history for the target motor has started filling
    while time.time() - start_wait_time < WAIT_FOR_DATA_TIMEOUT:
        initial_data = body.get_latest_motor_data(motor_index)
        # Check if we have some history points (at least enough for initial velocity calc later)
        if initial_data is not None and len(body._encoder_history[motor_index]) >= 2:
            print(f"Received initial data. Current position: {initial_data['encoderPos']} counts / {initial_data['angle']:.1f} deg")
            break
        time.sleep(0.01) # Wait 10ms between checks

    if initial_data is None or len(body._encoder_history[motor_index]) < 2:
        print(f"FATAL: Timed out waiting for sufficient encoder data from motor {motor_index}.")
        # Attempt to disable control before returning
        try:
             body.set_control_status(motor=motor_index, status=False)
        except:
             pass
        return False

    # --- Define helper functions for monitoring ---

    def wait_for_stability(motor_idx, timeout=WAIT_FOR_STABILITY_TIMEOUT):
        """Waits until the motor's encoder position stabilizes."""
        # Use the history collected by the listener for stability check
        start_time = time.time()
        print(f"Waiting for motor {motor_idx} to stabilize (max {timeout}s)...")

        while time.time() - start_time < timeout:
            with body.sock_lock: # Lock to safely access history
                 history = list(body._encoder_history[motor_idx]) # Get a copy of the history

            if len(history) >= STABILITY_WINDOW_SIZE:
                # Check the last STABILITY_WINDOW_SIZE points
                recent_history = history[-STABILITY_WINDOW_SIZE:]
                min_pos = min(p for t, p in recent_history)
                max_pos = max(p for t, p in recent_history)

                # Check if the range of recent readings is within the stability threshold
                if (max_pos - min_pos) <= STABILITY_THRESHOLD_COUNTS:
                    # Return the latest position once stable
                    latest_time, latest_pos = history[-1]
                    print(f"Motor {motor_idx} stable at ~{latest_pos} counts after {time.time() - start_time:.2f}s.")
                    return latest_pos

            # If not enough history or not stable yet, wait a moment and check again
            time.sleep(0.005) # Wait between checks

        print(f"Warning: Motor {motor_idx} did not stabilize within {timeout}s.")
        # Return the last known position or None if none received recently
        latest_data = body.get_latest_motor_data(motor_idx)
        return latest_data['encoderPos'] if latest_data else None


    def wait_for_movement_start_velocity(motor_idx, expected_direction_sign, timeout=WAIT_FOR_MOVEMENT_TIMEOUT):
        """Waits until the motor's encoder velocity exceeds a threshold in the expected direction."""
        start_time = time.time()
        print(f"Waiting for motor {motor_idx} to start moving ({'positive' if expected_direction_sign > 0 else 'negative'} velocity > {EXPECTED_VELOCITY_THRESHOLD} counts/sec) (max {timeout}s)...")

        # Clear encoder history briefly to ensure velocity calculation starts fresh from the point of command
        # This is a bit aggressive, maybe just note the start time and ignore history before that?
        # Let's just rely on the smoothing window filtering out old velocity.
        # with body.sock_lock:
        #      body._encoder_history[motor_idx].clear()
        #      print(f"Cleared history for motor {motor_idx} at start of movement wait.")


        # Give the command a moment to propagate and motor to potentially reverse into backlash
        time.sleep(0.05)


        while time.time() - start_time < timeout:
             smoothed_vel = body.get_smoothed_velocity(motor_idx, VELOCITY_SMOOTHING_WINDOW)

             # Check if smoothed velocity is in the expected direction and exceeds threshold
             # We check `smoothed_vel * expected_direction_sign` > threshold
             # If expected_direction_sign is 1, this is smoothed_vel > threshold
             # If expected_direction_sign is -1, this is -smoothed_vel > threshold, or smoothed_vel < -threshold
             if smoothed_vel * expected_direction_sign > EXPECTED_VELOCITY_THRESHOLD:
                 latest_data = body.get_latest_motor_data(motor_idx)
                 if latest_data:
                     print(f"Motor {motor_idx} detected movement start! Smoothed Velocity: {smoothed_vel:.2f} counts/sec. Current position: {latest_data['encoderPos']} counts after {time.time() - start_time:.2f}s.")
                     return latest_data['encoderPos'] # Return the position when velocity threshold is met
                 else:
                     print("Warning: Detected movement start but no latest data available.")
                     # Try to get data again or just continue
                     time.sleep(0.01) # Wait briefly

             # Wait between checks
             time.sleep(0.005)

        print(f"Warning: Motor {motor_idx} did not detect clear movement start within {timeout}s (Smoothed Velocity > {EXPECTED_VELOCITY_THRESHOLD} counts/sec).")
        # Return the last known position if movement wasn't detected
        latest_data = body.get_latest_motor_data(motor_idx)
        return latest_data['encoderPos'] if latest_data else None


    # --- Calibration Movements ---

    pos_at_neg_limit_1 = None
    pos_start_move_to_zero = None
    pos_at_zero_limit = None
    pos_start_move_to_neg90 = None
    pos_at_neg_limit_2 = None

    # Movement 1: Go to negative limit (-90 deg)
    print(f"\nStep 6/?: Moving {motor_name} to {CALIBRATION_ANGLE_NEG_LIMIT} degrees...")
    # Set target, but expect to hit the mechanical limit before reaching the target encoder count
    body.set_angles([0]*motor_index + [int(round(CALIBRATION_ANGLE_NEG_LIMIT))] + [0]*(7-motor_index))
    pos_at_neg_limit_1 = wait_for_stability(motor_index, timeout=15.0) # Allow more time to hit limit

    if pos_at_neg_limit_1 is None:
         print("FATAL: Failed to reach negative limit. Calibration cannot continue.")
         # Attempt to disable control before returning
         try:
              body.set_control_status(motor=motor_index, status=False)
         except:
              pass
         return False
    print(f"Negative limit reached (approx): {pos_at_neg_limit_1} counts")


    # Movement 2: Go to zero (0 deg) - Detecting backlash from the negative side
    print(f"\nStep 7/?: Moving {motor_name} to {CALIBRATION_ANGLE_ZERO} degrees (detecting backlash from negative side)...")
    # Set target to 0. The motor will first move positively within the backlash.
    body.set_angles([0]*motor_index + [int(round(CALIBRATION_ANGLE_ZERO))] + [0]*(7-motor_index))
    # Wait until movement *in the positive direction* is detected using velocity
    pos_start_move_to_zero = wait_for_movement_start_velocity(motor_index, expected_direction_sign=1, timeout=WAIT_FOR_MOVEMENT_TIMEOUT)

    if pos_start_move_to_zero is None:
         print("Warning: Failed to detect movement start towards zero (positive velocity). Cannot calculate backlash from negative side.")
         # Calibration can potentially continue to find the 0 position, but backlash measurement will be incomplete.
         pass # Allow it to continue

    # Now wait for it to stabilize at the zero position (which is a controlled target)
    print("\nStep 8/?: Waiting for stabilization at zero...")
    pos_at_zero_limit = wait_for_stability(motor_index)

    if pos_at_zero_limit is None:
         print("FATAL: Failed to reach zero position. Calibration cannot continue.")
         # Attempt to disable control before returning
         try:
              body.set_control_status(motor=motor_index, status=False)
         except:
              pass
         return False
    print(f"Zero position reached (approx): {pos_at_zero_limit} counts")


    # Movement 3: Go back to negative limit (-90 deg) - Detecting backlash from the zero side
    print(f"\nStep 9/?: Moving {motor_name} back to {CALIBRATION_ANGLE_NEG_LIMIT} degrees (detecting backlash from zero side)...")
    # Set target back to -90. The motor will first move negatively within the backlash.
    body.set_angles([0]*motor_index + [int(round(CALIBRATION_ANGLE_NEG_LIMIT))] + [0]*(7-motor_index))
    # Wait until movement *in the negative direction* is detected using velocity
    pos_start_move_to_neg90 = wait_for_movement_start_velocity(motor_index, expected_direction_sign=-1, timeout=WAIT_FOR_MOVEMENT_TIMEOUT)

    if pos_start_move_to_neg90 is None:
        print("Warning: Failed to detect movement start back towards negative limit (negative velocity). Cannot calculate backlash from zero side.")
        pass # Allow it to continue

    # Now wait for it to stabilize back at the negative mechanical limit
    print("\nStep 10/?: Waiting for stabilization at negative limit again...")
    pos_at_neg_limit_2 = wait_for_stability(motor_index, timeout=15.0) # Allow more time to hit limit again

    if pos_at_neg_limit_2 is None:
        print("Warning: Failed to reach negative limit again. Backlash measurement might be inaccurate.")
        # Use pos_at_neg_limit_1 for calculation if pos_at_neg_limit_2 is None
        pos_at_neg_limit_2 = pos_at_neg_limit_1
    else:
        print(f"Negative limit reached again (approx): {pos_at_neg_limit_2} counts")


    # --- Calculate and Report Backlash ---
    print("\n--- Calibration Results ---")

    backlash_neg_to_zero = None
    if pos_at_neg_limit_1 is not None and pos_start_move_to_zero is not None:
        # Backlash when moving from negative limit towards zero
        # It's the distance moved *into* the backlash zone from the limit until free movement starts
        backlash_neg_to_zero = pos_start_move_to_zero - pos_at_neg_limit_1
        print(f"Backlash (Moving -90 -> 0): {backlash_neg_to_zero} counts")
        print(f"                          ({backlash_neg_to_zero / COUNTS_PER_REV * 360:.2f} degrees)")
    else:
        print("Backlash (Moving -90 -> 0): Could not measure (failed to detect movement start)")


    backlash_zero_to_neg = None
    if pos_at_zero_limit is not None and pos_start_move_to_neg90 is not None:
        # Backlash when moving from zero towards negative limit
        # It's the distance moved *into* the backlash zone from the zero position until free movement starts
        backlash_zero_to_neg = pos_at_zero_limit - pos_start_move_to_neg90 # Difference should be positive if pos_start_move_to_neg90 is less than pos_at_zero_limit
        print(f"Backlash (Moving 0 -> -90): {backlash_zero_to_neg} counts")
        print(f"                         ({backlash_zero_to_neg / COUNTS_PER_REV * 360:.2f} degrees)")
    else:
        print("Backlash (Moving 0 -> -90): Could not measure (failed to detect movement start)")

    # You can interpret these results. If they are similar, the total play is roughly that value.
    # If they differ, the play might be asymmetrical or detection points were slightly off.
    # A simple average could be used, or take the max.
    measured_backlash_counts = []
    if backlash_neg_to_zero is not None:
        measured_backlash_counts.append(abs(backlash_neg_to_zero))
    if backlash_zero_to_neg is not None:
        measured_backlash_counts.append(abs(backlash_zero_to_neg))

    if measured_backlash_counts:
        average_backlash_counts = sum(measured_backlash_counts) / len(measured_backlash_counts)
        print(f"\nAverage measured backlash: {average_backlash_counts:.1f} counts")
        print(f"                         ({average_backlash_counts / COUNTS_PER_REV * 360:.2f} degrees)")
        # Store this value or use it to inform your DEAD_ZONE or other control logic
    else:
        print("\nCould not measure backlash in either direction.")


    print("\n--- Calibration Finished ---")
    return True # Indicate calibration procedure completed (even if measurements failed)


# --- Main Execution ---
if __name__ == "__main__":
    body = None # Initialize body to None
    try:
        print("Initializing QuadPilotBody...")
        body = QuadPilotBody()
        print("Initialization complete.")

        # Execute the calibration procedure
        calibration_success = calibrate_br_hip(body)

        if calibration_success:
             print("\nCalibration procedure finished successfully.")
        else:
             print("\nCalibration procedure encountered errors.")

    except Exception as e:
        print(f"FATAL: An error occurred during calibration: {e}")

    finally:
        # This block runs even if an exception occurs
        print("\nCleaning up...")
        if body:
            try:
                 # Attempt to disable the calibrated motor control specifically
                 print(f"Disabling control for {CALIBRATION_MOTOR_NAME} (Motor {CALIBRATION_MOTOR_INDEX})...")
                 body.set_control_status(motor=CALIBRATION_MOTOR_INDEX, status=False)
                 time.sleep(0.1) # Give time for command to send
            except Exception as cleanup_e:
                 print(f"Error disabling motor control during cleanup: {cleanup_e}")

            try:
                 print("Closing UDP socket...")
                 body.close() # Use the explicit close method
            except Exception as cleanup_e:
                 print(f"Error closing socket during cleanup: {cleanup_e}")
        else:
            print("Body object not initialized, no cleanup needed.")

        print("Cleanup complete. Exiting.")