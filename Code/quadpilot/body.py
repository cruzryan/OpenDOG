import requests

class QuadPilotBody:
    def __init__(self, ip1="192.168.137.100", ip2="192.168.137.101"):
        """Initialize with two ESP32 IPs, each controlling 4 motors."""
        self.ips = [ip1, ip2]  # ip1 for motors 0-3, ip2 for motors 4-7
        self.session = requests.Session()

    def _get_ip_for_motor(self, motor):
        """Return the IP for the ESP32 controlling the given motor."""
        if not 0 <= motor <= 7:
            raise ValueError("Motor index must be 0-7")
        return self.ips[0] if motor < 4 else self.ips[1]

    def _adjust_motor_index(self, motor):
        """Adjust motor index for the ESP32 (0-3 for each ESP32)."""
        return motor % 4

    def set_control_params(self, P: float, I: float, D: float, dead_zone: int, pos_thresh: int):
        """Set PID parameters, dead zone, and position threshold for all motors on both ESP32s."""
        for ip in self.ips:
            url = f"http://{ip}:82/set_control_params?P={P}&I={I}&D={D}&dead_zone={dead_zone}&pos_thresh={pos_thresh}"
            self.session.get(url)

    def set_angles(self, angles: list[int]):
        """Set target angles in degrees for all 8 motors at once."""
        if len(angles) != 8:
            raise ValueError("Exactly 8 angles must be provided")
        params1 = "&".join(f"a{i}={angle}" for i, angle in enumerate(angles[:4]))
        params2 = "&".join(f"a{i}={angle}" for i, angle in enumerate(angles[4:]))
        self.session.get(f"http://{self.ips[0]}:82/set_angles?{params1}")
        self.session.get(f"http://{self.ips[1]}:82/set_angles?{params2}")

    def set_all_pins(self, pins: list[tuple[int, int, int, int]]):
        """Set encoder and motor pins for all 8 motors at once."""
        if len(pins) != 8:
            raise ValueError("Exactly 8 motor pin configurations must be provided")
        params1 = "&".join(f"ENCODER_A{i}={p[0]}&ENCODER_B{i}={p[1]}&IN1_{i}={p[2]}&IN2_{i}={p[3]}" for i, p in enumerate(pins[:4]))
        params2 = "&".join(f"ENCODER_A{i}={p[0]}&ENCODER_B{i}={p[1]}&IN1_{i}={p[2]}&IN2_{i}={p[3]}" for i, p in enumerate(pins[4:]))
        self.session.get(f"http://{self.ips[0]}:82/set_all_pins?{params1}")
        self.session.get(f"http://{self.ips[1]}:82/set_all_pins?{params2}")

    def set_control_status(self, motor: int, status: bool):
        """Enable or disable control for a specific motor."""
        ip = self._get_ip_for_motor(motor)
        adjusted_motor = self._adjust_motor_index(motor)
        url = f"http://{ip}:82/set_control_status?motor={adjusted_motor}&status={1 if status else 0}"
        self.session.get(url)

    def set_all_control_status(self, status: bool):
        """Enable or disable control for all motors on both ESP32s."""
        for ip in self.ips:
            url = f"http://{ip}:82/set_control_status?motor=0&status={1 if status else 0}"
            for i in range(1, 4):  # Apply to motors 0-3 on each ESP32
                url += f"&motor={i}&status={1 if status else 0}"
            self.session.get(url)

    def reset_all(self):
        """Reset the encoder positions to 0 for all motors on both ESP32s."""
        for ip in self.ips:
            url = f"http://{ip}:82/reset_all"
            self.session.get(url)