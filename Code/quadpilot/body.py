import requests

class QuadPilotBody:
    def __init__(self, ip):
        """Initialize with the ESP32's IP address."""
        self.ip = ip
        self.session = requests.Session()

    def set_control_params(self, P: float, I: float, D: float, dead_zone: int, pos_thresh: int):
        """Set PID parameters, dead zone, and position threshold."""
        url = f"http://{self.ip}:82/set_control_params?P={P}&I={I}&D={D}&dead_zone={dead_zone}&pos_thresh={pos_thresh}"
        self.session.get(url)

    def set_angle(self, a: int):
        """Set target angle in degrees."""
        url = f"http://{self.ip}:82/set_angle?a={a}"
        self.session.get(url)

    def set_pins(self, ENCODER_A: int, ENCODER_B: int, IN1: int, IN2: int):
        """Set encoder and motor pins on the ESP32."""
        url = f"http://{self.ip}:82/set_pins?ENCODER_A={ENCODER_A}&ENCODER_B={ENCODER_B}&IN1={IN1}&IN2={IN2}"
        self.session.get(url)

    def get_angle(self) -> float:
        """Get current angle in degrees, calculated from encoder position."""
        response = self.session.get(f"http://{self.ip}:82/get_angle")
        return float(response.text)

    def get_encoderPos(self) -> int:
        """Get current encoder position."""
        response = self.session.get(f"http://{self.ip}:82/get_encoderPos")
        return int(response.text)

    def set_control_status(self, status: bool):
        """Enable or disable motor control (true = on, false = off)."""
        url = f"http://{self.ip}:82/set_control_status?status={1 if status else 0}"
        self.session.get(url)