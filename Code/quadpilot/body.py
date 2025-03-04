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

    def set_angle(self, motor: int, a: int):
        """Set target angle in degrees for a specific motor."""
        url = f"http://{self.ip}:82/set_angle?motor={motor}&a={a}"
        self.session.get(url)

    def set_pins(self, motor: int, ENCODER_A: int, ENCODER_B: int, IN1: int, IN2: int):
        """Set encoder and motor pins for a specific motor on the ESP32."""
        url = f"http://{self.ip}:82/set_pins?motor={motor}&ENCODER_A={ENCODER_A}&ENCODER_B={ENCODER_B}&IN1={IN1}&IN2={IN2}"
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

    def reset(self, motor: int):
        """Reset the encoder position to 0 for a specific motor."""
        url = f"http://{self.ip}:82/reset?motor={motor}"
        self.session.get(url)