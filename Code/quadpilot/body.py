import requests

class QuadPilotBody:
    def __init__(self, ip):
        """Initialize with the ESP32's IP address."""
        self.ip = ip
        self.session = requests.Session()

    def set_control_params(self, P: float, I: float, D: float, dead_zone: int, pos_thresh: int):
        """Set PID parameters, dead zone, and position threshold for all motors."""
        url = f"http://{self.ip}:82/set_control_params?P={P}&I={I}&D={D}&dead_zone={dead_zone}&pos_thresh={pos_thresh}"
        self.session.get(url)

    def set_all_control_params(self, P: float, I: float, D: float, dead_zone: int, pos_thresh: int):
        """Set PID parameters, dead zone, and position threshold for all motors at once."""
        url = f"http://{self.ip}:82/set_all_control_params?P={P}&I={I}&D={D}&dead_zone={dead_zone}&pos_thresh={pos_thresh}"
        self.session.get(url)

    def set_angle(self, motor: int, a: int):
        """Set target angle in degrees for a specific motor."""
        url = f"http://{self.ip}:82/set_angle?motor={motor}&a={a}"
        self.session.get(url)

    def set_angles(self, angles: list[int]):
        """Set target angles in degrees for all 8 motors at once."""
        if len(angles) != 8:
            raise ValueError("Exactly 8 angles must be provided")
        params = "&".join(f"a{i}={angle}" for i, angle in enumerate(angles))
        url = f"http://{self.ip}:82/set_angles?{params}"
        self.session.get(url)

    def set_pins(self, motor: int, ENCODER_A: int, ENCODER_B: int, IN1: int, IN2: int):
        """Set encoder and motor pins for a specific motor on the ESP32."""
        url = f"http://{self.ip}:82/set_pins?motor={motor}&ENCODER_A={ENCODER_A}&ENCODER_B={ENCODER_B}&IN1={IN1}&IN2={IN2}"
        self.session.get(url)

    def set_all_pins(self, pins: list[tuple[int, int, int, int]]):
        """Set encoder and motor pins for all 8 motors at once."""
        if len(pins) != 8:
            raise ValueError("Exactly 8 motor pin configurations must be provided")
        params = "&".join(f"ENCODER_A{i}={p[0]}&ENCODER_B{i}={p[1]}&IN1_{i}={p[2]}&IN2_{i}={p[3]}" for i, p in enumerate(pins))
        url = f"http://{self.ip}:82/set_all_pins?{params}"
        self.session.get(url)

    def get_angle(self, motor: int) -> float:
        """Get current angle in degrees for a specific motor."""
        response = self.session.get(f"http://{self.ip}:82/get_angle?motor={motor}")
        return float(response.text)

    def get_encoderPos(self, motor: int) -> int:
        """Get current encoder position for a specific motor."""
        response = self.session.get(f"http://{self.ip}:82/get_encoderPos?motor={motor}")
        return int(response.text)

    def set_control_status(self, motor: int, status: bool):
        """Enable or disable control for a specific motor (True = on, False = off)."""
        url = f"http://{self.ip}:82/set_control_status?motor={motor}&status={1 if status else 0}"
        self.session.get(url)

    def set_all_control_status(self, status: bool):
        """Enable or disable control for all motors at once (True = on, False = off)."""
        url = f"http://{self.ip}:82/set_all_control_status?status={1 if status else 0}"
        self.session.get(url)

    def reset(self, motor: int):
        """Reset the encoder position to 0 for a specific motor."""
        url = f"http://{self.ip}:82/reset?motor={motor}"
        self.session.get(url)

    def reset_all(self):
        """Reset the encoder positions to 0 for all motors at once."""
        url = f"http://{self.ip}:82/reset_all"
        self.session.get(url)