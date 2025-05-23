# quadpilot.py
import threading
import time
import socket
import json
import atexit

class QuadPilotBody:
    def __init__(self, ip1="192.168.137.100", ip2="192.168.137.101", listen_for_broadcasts=False):
        self.ips = [ip1, ip2]
        self.UDP_COMMAND_PORT = 12345
        self.UDP_LISTEN_PORT = 12345

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        self.sock.settimeout(0.1) # Default timeout for recvfrom, like original

        self._is_closed = False
        self.sock_lock = threading.Lock()

        self.listen_for_broadcasts_flag = listen_for_broadcasts
        self._listener_thread = None
        self._stop_listener_event = threading.Event()

        self.data_access_lock = threading.Lock()
        self._imu_data_store = {ip: {} for ip in self.ips}
        self._motor_data_store = {ip: {'angles': [0]*4, 'encoderPos': [0]*4, 'targetPos': [0]*4,
                                       'debug': [0]*4, 'debugComputed': [0]*4, 'mpu_available': False}
                                  for ip in self.ips}
        self._data_reception_status = {ip: False for ip in self.ips}

        try:
            self.sock.bind(('0.0.0.0', self.UDP_LISTEN_PORT))
            if self.listen_for_broadcasts_flag:
                self._listener_thread = threading.Thread(target=self._data_listener_loop, daemon=True)
                self._listener_thread.start()
        except Exception as e:
            self.close()
            raise e
        
        atexit.register(self.close)

    def _get_ip_for_motor(self, motor_idx: int):
        if not 0 <= motor_idx <= 7:
            raise ValueError("Motor index must be 0-7")
        return self.ips[0] if motor_idx < 4 else self.ips[1]

    def _adjust_motor_index_for_esp(self, motor_idx: int):
        return motor_idx % 4

    def _send_command_and_wait_for_ok(self, ip: str, command_data: dict, retries: int = 3, timeout_per_retry: float = 0.5):
        if self._is_closed:
            return False

        attempt = 0
        message = json.dumps(command_data).encode('utf-8')
        
        while attempt < retries:
            with self.sock_lock:
                if self._is_closed:
                     return False
                try:
                    self.sock.sendto(message, (ip, self.UDP_COMMAND_PORT))
                    start_time = time.time()
                    while time.time() - start_time < timeout_per_retry:
                        try:
                            if self._is_closed:
                                 return False
                            data, addr = self.sock.recvfrom(1024)
                            response = json.loads(data.decode('utf-8'))
                            if response.get("status") == "OK" and addr[0] == ip:
                                return True
                            elif "angles" in response or "imu" in response : # Ignore telemetry
                                pass 
                        except socket.timeout: # From self.sock.settimeout(0.1)
                            break 
                        except json.JSONDecodeError:
                            break 
                        except OSError:
                             if self._is_closed: return False
                             break 
                        except Exception:
                            break 
                except Exception:
                    pass

            attempt += 1
            if self._is_closed: return False
            if attempt < retries:
                 time.sleep(0.05)
        return False

    def _data_listener_loop(self):
        while not self._stop_listener_event.is_set():
            if self._is_closed: break
            try:
                data, addr = self.sock.recvfrom(1024) 
                ip_address = addr[0]
                if ip_address not in self.ips: continue
                json_data = json.loads(data.decode('utf-8'))
                with self.data_access_lock:
                    self._data_reception_status[ip_address] = True
                    current_esp_motor_data = self._motor_data_store.get(ip_address)
                    if not current_esp_motor_data: continue
                    if "angles" in json_data and "encoderPos" in json_data:
                        if (len(json_data["angles"]) == 4 and len(json_data["encoderPos"]) == 4):
                            current_esp_motor_data['angles'] = json_data["angles"]
                            current_esp_motor_data['encoderPos'] = json_data["encoderPos"]
                            current_esp_motor_data['targetPos'] = json_data.get("targetPos", current_esp_motor_data['targetPos'])
                            current_esp_motor_data['debug'] = json_data.get("debug", current_esp_motor_data['debug'])
                            current_esp_motor_data['debugComputed'] = json_data.get("debugComputed", current_esp_motor_data['debugComputed'])
                            current_esp_motor_data['mpu_available'] = json_data.get("mpu_available", current_esp_motor_data['mpu_available'])
                        if "imu" in json_data: 
                            self._imu_data_store[ip_address] = json_data["imu"]
                    elif "imu" in json_data: 
                        self._imu_data_store[ip_address] = json_data["imu"]
                        current_esp_motor_data['mpu_available'] = True
                    elif "status" in json_data and json_data["status"] == "OK": pass 
                    elif "error" in json_data and json_data["error"] == "MPU6050 not initialized":
                        self._imu_data_store[ip_address] = {}
                        current_esp_motor_data['mpu_available'] = False
            except socket.timeout: continue 
            except json.JSONDecodeError: pass
            except OSError: 
                if self._is_closed or self._stop_listener_event.is_set(): break
            except Exception:
                if self._is_closed or self._stop_listener_event.is_set(): break
                time.sleep(0.01)

    def get_latest_motor_data_for_esp(self, ip_index: int):
        if not (0 <= ip_index < len(self.ips)): return None
        ip = self.ips[ip_index]
        with self.data_access_lock:
            return self._motor_data_store.get(ip, {}).copy() if self._data_reception_status.get(ip) else None
            
    def get_latest_imu_data_for_esp(self, ip_index: int):
        if not (0 <= ip_index < len(self.ips)): return {}
        ip = self.ips[ip_index]
        with self.data_access_lock:
            return self._imu_data_store.get(ip, {}).copy() if self._data_reception_status.get(ip) else {}

    def is_mpu_available_for_esp(self, ip_index: int):
        if not (0 <= ip_index < len(self.ips)): return False
        ip = self.ips[ip_index]
        with self.data_access_lock:
            return self._motor_data_store.get(ip, {}).get('mpu_available', False)

    def is_data_available_from_esp(self, ip_index: int):
        if not (0 <= ip_index < len(self.ips)): return False
        ip = self.ips[ip_index]
        with self.data_access_lock:
            return self._data_reception_status.get(ip, False)

    def set_control_params(self, P: float, I: float, D: float, dead_zone: int, pos_thresh: int) -> bool:
        command = {"command": "set_control_params", "P": P, "I": I, "D": D, "dead_zone": dead_zone, "pos_thresh": pos_thresh}
        res1 = self._send_command_and_wait_for_ok(self.ips[0], command.copy(), retries=5, timeout_per_retry=1.0)
        res2 = self._send_command_and_wait_for_ok(self.ips[1], command.copy(), retries=5, timeout_per_retry=1.0)
        return res1 and res2

    def set_angles(self, angles: list[float]) -> bool:
        if len(angles) != 8: raise ValueError("Exactly 8 angles must be provided")
        angles_int = [int(round(a)) for a in angles]
        cmd1_data = {"command": "set_angles", "angles": angles_int[:4]}
        cmd2_data = {"command": "set_angles", "angles": angles_int[4:]}
        results = [False, False]
        def _task(ip_addr, data, result_idx):
            results[result_idx] = self._send_command_and_wait_for_ok(ip_addr, data, retries=1, timeout_per_retry=0.1)
        thread1 = threading.Thread(target=_task, args=(self.ips[0], cmd1_data, 0))
        thread2 = threading.Thread(target=_task, args=(self.ips[1], cmd2_data, 1))
        thread1.start(); thread2.start()
        thread1.join(timeout=0.3); thread2.join(timeout=0.3)
        return all(results)

    def set_all_pins(self, pins_config: list[tuple[int, int, int, int]]) -> bool:
        if len(pins_config) != 8: raise ValueError("Exactly 8 pin configs must be provided")
        cmd1 = {"command": "set_all_pins"}; cmd2 = {"command": "set_all_pins"}
        for i, p_cfg in enumerate(pins_config[:4]):
            cmd1[f"ENCODER_A{i}"]=p_cfg[0]; cmd1[f"ENCODER_B{i}"]=p_cfg[1]
            cmd1[f"IN1_{i}"]=p_cfg[2]; cmd1[f"IN2_{i}"]=p_cfg[3]
        for i, p_cfg in enumerate(pins_config[4:]):
            cmd2[f"ENCODER_A{i}"]=p_cfg[0]; cmd2[f"ENCODER_B{i}"]=p_cfg[1]
            cmd2[f"IN1_{i}"]=p_cfg[2]; cmd2[f"IN2_{i}"]=p_cfg[3]
        res1 = self._send_command_and_wait_for_ok(self.ips[0], cmd1, retries=5, timeout_per_retry=1.0)
        res2 = self._send_command_and_wait_for_ok(self.ips[1], cmd2, retries=5, timeout_per_retry=1.0)
        return res1 and res2

    def set_control_status(self, motor_idx: int, status: bool) -> bool:
        ip = self._get_ip_for_motor(motor_idx)
        esp_motor_idx = self._adjust_motor_index_for_esp(motor_idx)
        command = {"command": "set_control_status", "motor": esp_motor_idx, "status": 1 if status else 0}
        return self._send_command_and_wait_for_ok(ip, command, retries=2, timeout_per_retry=0.5)

    def set_all_control_status(self, status: bool) -> bool:
        all_success = True
        status_val = 1 if status else 0
        for ip_addr in self.ips:
            for esp_local_motor_idx in range(4):
                command = {"command": "set_control_status", "motor": esp_local_motor_idx, "status": status_val}
                if not self._send_command_and_wait_for_ok(ip_addr, command.copy(), retries=2, timeout_per_retry=0.2):
                    all_success = False
        return all_success

    def reset_all(self) -> bool:
        command = {"command": "reset_all"}
        res1 = self._send_command_and_wait_for_ok(self.ips[0], command, retries=5, timeout_per_retry=1.0)
        res2 = self._send_command_and_wait_for_ok(self.ips[1], command, retries=5, timeout_per_retry=1.0)
        return res1 and res2
    
    def set_send_interval(self, interval_ms: int) -> bool:
        if interval_ms <= 0: interval_ms = 1 
        command = {"command": "set_send_interval", "interval": interval_ms}
        res1 = self._send_command_and_wait_for_ok(self.ips[0], command, retries=3, timeout_per_retry=0.5)
        res2 = self._send_command_and_wait_for_ok(self.ips[1], command, retries=3, timeout_per_retry=0.5)
        return res1 and res2

    def close(self):
        if not self._is_closed:
            self._is_closed = True 
            self._stop_listener_event.set()
            if self._listener_thread and self._listener_thread.is_alive():
                self._listener_thread.join(timeout=1.0)
            with self.sock_lock:
                try: self.sock.close()
                except Exception: pass
            try: atexit.unregister(self.close)
            except Exception: pass

    def __del__(self):
        self.close()