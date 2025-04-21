import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
from quadpilot import QuadPilotCamera 

class CameraApp:
    def __init__(self, root, camera_ip):
        self.root = root
        self.root.title("ESP32 Camera Stream")

        self.camera = QuadPilotCamera(camera_ip) # Initialize QuadPilotCamera
        self.camera.connect() # Connect to the camera (currently just sets URLs)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.framesize_var = tk.StringVar(value="VGA")
        # Modified framesize options here:
        self.framesize_menu = ttk.OptionMenu(root, self.framesize_var, "VGA",
                                             "VGA", "SVGA", "XGA", "HD", "SXGA", "UXGA",
                                             command=self.change_framesize)
        self.framesize_menu.pack()

        self.stream_thread = None
        self.current_frame_generator = None
        self.start_streaming() # Start streaming with default framesize

    def start_streaming(self):
        """Starts the streaming thread."""
        if self.stream_thread and self.stream_thread.is_alive():
            self.stop_streaming() # Ensure any existing stream is stopped first
        framesize = self.framesize_var.get()
        self.current_frame_generator = self.camera.stream(framesize)
        self.stream_thread = threading.Thread(target=self.update_frame)
        self.stream_thread.start()

    def stop_streaming(self):
        """Stops the streaming thread and camera stream."""
        self.camera.stop_stream()
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join() # Wait for thread to finish

    def update_frame(self):
        """Updates the image label with frames from the stream."""
        try:
            for frame in self.current_frame_generator:
                if not self.camera.streaming: # Check if streaming should stop from inside generator
                    break # Exit loop and thread will terminate
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.image_label.imgtk = imgtk
                self.image_label.configure(image=imgtk)
        except Exception as e: # Catch any exceptions during streaming and display error
            print(f"Error in update_frame: {e}")
            self.image_label.config(text=f"Stream Error: {e}")
        finally:
            print("Stream thread finished.")


    def change_framesize(self, framesize):
        """Handles framesize changes."""
        print(f"Changing framesize requested to: {framesize}")
        self.stop_streaming() # Stop current stream
        self.start_streaming() # Restart stream with new framesize

    def on_closing(self):
        """Handles window closing."""
        self.stop_streaming()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    camera_ip = "192.168.137.52" # Set your ESP32 camera IP here
    app = CameraApp(root, camera_ip)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()