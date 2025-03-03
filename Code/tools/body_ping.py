import os
import sys
import time
from sseclient import SSEClient
import statistics

script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from quadpilot import QuadPilotBody

# Initialize with ESP32's IP address
body = QuadPilotBody("192.168.0.198")

# Connect to the SSE endpoint
url = "http://192.168.0.198:82/events"
messages = SSEClient(url)

# Initialize variables for average calculation
ping_times = []
WINDOW_SIZE = 10  # Number of measurements to average
last_time = time.time()

# Process real-time angle updates
try:
    for msg in messages:
        try:
            current_time = time.time()
            ping = (current_time - last_time) * 1000  # Convert to milliseconds
            last_time = current_time
            
            # Add new ping time and maintain window size
            ping_times.append(ping)
            if len(ping_times) > WINDOW_SIZE:
                ping_times.pop(0)
            
            # Calculate and display average
            avg_ping = statistics.mean(ping_times)
            # Use carriage return to overwrite previous line
            print(f"Average Ping (last {WINDOW_SIZE} measurements): {avg_ping:.2f} ms", end='\r')
            
        except ValueError:
            print("Invalid data received", end='\r')
            
except KeyboardInterrupt:
    print("\nMonitoring stopped by user")
