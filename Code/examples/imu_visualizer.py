import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg") # Use TkAgg backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import threading
import time
from quadpilot import QuadPilotCamera

class IMUApp:
    def __init__(self, root, camera_ip):
        self.root = root
        self.root.title("3D IMU Visualizer")
        self.camera = QuadPilotCamera(camera_ip)
        self.camera.connect()

        # Matplotlib setup for 3D plot
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax3d = self.fig.add_subplot(111, projection='3d')
        self.ax3d.set_xlabel('X Acceleration')
        self.ax3d.set_ylabel('Y Acceleration')
        self.ax3d.set_zlabel('Z Acceleration')
        self.ax3d.set_xlim([-10, 10]) # Adjust limits as needed based on your accel range
        self.ax3d.set_ylim([-10, 10])
        self.ax3d.set_zlim([-10, 10])
        self.ax3d.view_init(elev=20, azim=45) # Set initial 3D view

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Initialize the vector representation (quiver)
        self.vector_artist = None

        self.is_running = True
        self.update_visualization() # Start updating visualization

    def update_visualization(self):
        """Updates the 3D visualization with new IMU data."""
        if not self.is_running:
            return

        imu_data = self.camera.get_imu_data()
        if imu_data:
            try:
                accel_x = imu_data.get('accel_x', 0.0)
                accel_y = imu_data.get('accel_y', 0.0)
                accel_z = imu_data.get('accel_z', 0.0)

                vector = [accel_x, accel_y, accel_z]

                # Clear previous vector if it exists
                if self.vector_artist:
                    self.vector_artist.remove()

                # Create a new vector (quiver plot)
                self.vector_artist = self.ax3d.quiver(0, 0, 0, vector[0], vector[1], vector[2], length=1.0, normalize=True, color='r')

                self.canvas.draw() # Redraw the canvas
            except Exception as e:
                print(f"Error processing IMU data: {e}")

        self.root.after(100, self.update_visualization) # Update every 100ms

    def on_closing(self):
        """Handles window closing."""
        self.is_running = False # Stop the update loop
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    camera_ip = "192.168.0.131" # Set your ESP32 camera IP here
    app = IMUApp(root, camera_ip)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()