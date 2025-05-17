import threading
import time
import socket
import json

class QuadPilotBody:
    def __init__(self, ip1="192.168.137.100", ip2="192.168.137.101"):
        """Initialize with two ESP32 IPs, each controlling 4 motors."""
        self.ips = [ip1, ip2]  # ip1 for motors 0-3, ip2 for motors 4-7
        self.UDP_PORT = 12345
        # Create a single UDP socket for sending commands and receiving responses
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)  # 1-second timeout for responses
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
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

    def _send_udp_command(self, ip, command_data):
        """Send a command over UDP and wait for a response, with thread safety."""
        with self.sock_lock:
            try:
                message = json.dumps(command_data).encode('utf-8')
                self.sock.sendto(message, (ip, self.UDP_PORT))
                # Wait for response
                data, _ = self.sock.recvfrom(1024)
                response = json.loads(data.decode('utf-8'))
                if response.get("status") == "OK":
                    return True
                else:
                    print(f"Command {command_data['command']} to {ip} failed: {response}")
                    return False
            except socket.timeout:
                print(f"Timeout waiting for response from {ip} for command {command_data['command']}")
                return False
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
            self._send_udp_command(ip, command)

    def set_angles(self, angles: list[int]):
        """Set target angles in degrees for all 8 motors concurrently using threads."""
        if len(angles) != 8:
            raise ValueError("Exactly 8 angles must be provided")
        
        # Prepare commands for each ESP32 (first 4 motors on ip1, next 4 on ip2)
        command1 = {"command": "set_angles", "angles": angles[:4]}
        command2 = {"command": "set_angles", "angles": angles[4:]}
        
        def send_request(ip: str, command: dict):
            """Helper function to send UDP command for a group of motors."""
            self._send_udp_command(ip, command)
        
        # Create two threads, one for each group of 4 motors
        thread1 = threading.Thread(target=send_request, args=(self.ips[0], command1))
        thread2 = threading.Thread(target=send_request, args=(self.ips[1], command2))
        
        # Start both threads
        thread1.start()
        thread2.start()
        
        # Wait for both threads to complete
        thread1.join()
        thread2.join()

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
        self._send_udp_command(self.ips[0], command1)
        self._send_udp_command(self.ips[1], command2)

    def set_control_status(self, motor: int, status: bool):
        """Enable or disable control for a specific motor."""
        ip = self._get_ip_for_motor(motor)
        adjusted_motor = self._adjust_motor_index(motor)
        command = {
            "command": "set_control_status",
            "motor": adjusted_motor,
            "status": 1 if status else 0
        }
        self._send_udp_command(ip, command)

    def set_all_control_status(self, status: bool):
        """Enable or disable control for all motors on both ESP32s."""
        command = {
            "command": "set_control_status",
            "motor": 0,
            "status": 1 if status else 0
        }
        # ESP32 code expects a single command to handle all motors, so send to both
        for ip in self.ips:
            # Update motor index for each ESP32 (0-3)
            for i in range(4):
                command["motor"] = i
                self._send_udp_command(ip, command)

    def reset_all(self):
        """Reset the encoder positions to 0 for all motors on both ESP32s."""
        command = {"command": "reset_all"}
        for ip in self.ips:
            self._send_udp_command(ip, command)

    def __del__(self):
        """Cleanup: close the UDP socket."""
        self.sock.close()