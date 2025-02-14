import requests
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
import json # Import json module

class QuadPilotCamera:
    def __init__(self, ip):
        self.ip = ip
        self.stream_url = f"http://{ip}:81/stream"
        self.control_url = f"http://{ip}:81/control"
        self.imu_data_url = f"http://{ip}:81/imu_data" # New IMU data URL
        self.current_stream_thread = None
        self.streaming = False

    def connect(self):
        """
        Connects to the ESP32 camera (currently just sets URLs).
        In a more robust implementation, this could include pinging the IP
        or checking for a response from the camera's web server.
        """
        print(f"Connecting to camera at IP: {self.ip}")
        # In a real application, you might add connection testing here

    def _frame_generator(self, framesize):
        """
        Internal generator to fetch frames from the stream.
        """
        response = None # Initialize response outside try block for finally clause scope
        try:
            response = requests.get(self.stream_url, stream=True, timeout=10)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            bytes_ = bytes()
            for chunk in response.iter_content(chunk_size=1024):
                if not self.streaming:  # Check if streaming should stop
                    break
                bytes_ += chunk
                a = bytes_.find(b'\xff\xd8')
                b = bytes_.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_[a:b+2]
                    bytes_ = bytes_[b+2:]
                    img_np = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img_np is not None: # Check if decoding was successful
                        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                        yield img_rgb
                    else:
                        print("Error decoding frame.")
        except requests.exceptions.RequestException as e:
            print(f"Stream error: {e}")
        finally:
            self.streaming = False # Ensure streaming flag is reset when stream ends
            if response:
                response.close() # Close response to release resources


    def stream(self, framesize="VGA"):
        """
        Starts streaming video frames with the specified framesize.
        Returns a frame generator.
        """
        print(f"Starting stream with framesize: {framesize}")
        self.change_framesize(framesize) # Change framesize before starting stream
        self.streaming = True
        return self._frame_generator(framesize)

    def stop_stream(self):
        """
        Stops the current stream.
        """
        print("Stopping stream.")
        self.streaming = False
        if self.current_stream_thread and self.current_stream_thread.is_alive():
            self.current_stream_thread.join(timeout=5) # Wait for thread to finish, with timeout

    def change_framesize(self, framesize):
        """
        Changes the framesize of the camera.
        """
        print(f"Changing framesize to: {framesize}")
        try:
            response = requests.post(self.control_url, data=f"framesize:{framesize}", timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            if response.status_code == 200:
                print("Framesize changed successfully.")
            else:
                print(f"Failed to change framesize. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error changing framesize: {e}")

    def get_imu_data(self):
        """
        Fetches IMU data from the ESP32.
        Returns a dictionary containing IMU data, or None on error.
        """
        try:
            response = requests.get(self.imu_data_url, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses
            return response.json() # Parse JSON response
        except requests.exceptions.RequestException as e:
            print(f"Error fetching IMU data: {e}")
            return None