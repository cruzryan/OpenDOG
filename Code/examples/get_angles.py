import os
import sys
import time
import threading
import queue
import socket
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Assuming QuadPilotBody is in the parent directory relative to this script's dir
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

try:
    from quadpilot import QuadPilotBody
except ImportError as e:
    print(f"Error importing QuadPilotBody: {e}")
    print("Please ensure 'quadpilot.py' is in the correct directory (expected: parent of script directory).")
    sys.exit(1)

# --- Initialize QuadPilotBody ---
try:
    body = QuadPilotBody()
    print("QuadPilotBody initialized.")
except Exception as e:
    print(f"FATAL: Failed to initialize QuadPilotBody: {e}")
    sys.exit(1)

# --- Define IPs for Old (HTTP) and New (UDP) ESP32s ---
ips = {
    "192.168.137.100": "old",  # Old code (HTTP)
    "192.168.137.101": "new"    # New code (UDP)
}

# --- Shared Data Structures ---
fetch_times = {ip: [] for ip in ips.keys()}  # Store fetch times for each IP
fetch_times_lock = threading.Lock()  # Lock for thread-safe access to fetch_times
data_queue = queue.Queue()  # Queue for passing fetch data to main thread
running = threading.Event()  # Control flag for threads
running.set()
MAX_FETCH_HISTORY = 100  # Limit the number of fetch times to keep the plot responsive

# --- UDP Socket Setup for the New ESP32 (192.168.137.101) ---
UDP_PORT = 12345
udp_sockets = {}

# Register with the UDP ESP32 (192.168.137.101) and set up UDP socket
udp_ip = "192.168.137.101"
try:
    # Register with the ESP32 to receive UDP packets
    print(f"Attempting to register with {udp_ip}...")
    response = requests.get(f"http://{udp_ip}:82/register", timeout=5)
    print(f"Registration response from {udp_ip}: {response.status_code} - {response.text}")
    if response.status_code != 200:
        print(f"Warning: Registration with {udp_ip} failed with status {response.status_code}")

    # Create a UDP socket for this ESP32
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', UDP_PORT))
    sock.settimeout(1.0)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast reception
    udp_sockets[udp_ip] = sock
except Exception as e:
    print(f"Failed to register with {udp_ip}: {e}")

# --- Functions ---
def fetch_angles_http(ip):
    """Fetch angles from an ESP32 using HTTP event stream (old method)."""
    while running.is_set():
        start_time = time.time()
        angles = []
        try:
            response = requests.get(f"http://{ip}:82/events", stream=True, timeout=5)
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data:"):
                        data = json.loads(decoded_line[5:])
                        angles = data.get("angles", [])
                        break
            response.close()
            end_time = time.time()
            time_taken_ms = (end_time - start_time) * 1000
            with fetch_times_lock:
                fetch_times[ip].append(time_taken_ms)
                if len(fetch_times[ip]) > MAX_FETCH_HISTORY:
                    fetch_times[ip] = fetch_times[ip][-MAX_FETCH_HISTORY:]
                avg_ms = np.mean(fetch_times[ip]) if fetch_times[ip] else 0
            data_queue.put((ip, angles, time_taken_ms, avg_ms))
        except Exception as e:
            print(f"HTTP fetch error from {ip}: {e}")
            data_queue.put((ip, [], 0, 0, str(e)))
        time.sleep(0.1)  # Slower fetch rate for HTTP to balance with UDP

def fetch_angles_udp(ip, sock):
    """Fetch angles from an ESP32 using UDP (new method)."""
    while running.is_set():
        start_time = time.time()
        angles = []
        try:
            data, addr = sock.recvfrom(1024)
            decoded_data = data.decode('utf-8')
            parsed_data = json.loads(decoded_data)
            angles = parsed_data.get("angles", [])
            end_time = time.time()
            time_taken_ms = (end_time - start_time) * 1000
            with fetch_times_lock:
                fetch_times[ip].append(time_taken_ms)
                if len(fetch_times[ip]) > MAX_FETCH_HISTORY:
                    fetch_times[ip] = fetch_times[ip][-MAX_FETCH_HISTORY:]
                avg_ms = np.mean(fetch_times[ip]) if fetch_times[ip] else 0
            data_queue.put((ip, angles, time_taken_ms, avg_ms))
        except Exception as e:
            print(f"UDP fetch error from {ip}: {e}")
            # Add a placeholder fetch time to keep the graph moving
            with fetch_times_lock:
                fetch_times[ip].append(0)  # Placeholder for failed fetch
                if len(fetch_times[ip]) > MAX_FETCH_HISTORY:
                    fetch_times[ip] = fetch_times[ip][-MAX_FETCH_HISTORY:]
                avg_ms = np.mean(fetch_times[ip]) if fetch_times[ip] else 0
            data_queue.put((ip, [], 0, avg_ms, str(e)))
        time.sleep(0.01)  # Match ESP32's 10ms send interval

def setup_plot():
    """Set up the Matplotlib figure and axes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Fetch Number (Last 100 Fetches)")
    ax.set_ylabel("Fetch Time (ms)")
    ax.set_title("ESP32 Fetch Times: (100) vs (101)")
    ax.grid(True)
    
    # Initialize empty lines for each IP and their averages
    lines = {}
    for ip, method in ips.items():
        label = f"{ip} ({method}) Fetch Time"
        lines[f"{ip}_data"], = ax.plot([], [], label=label, marker='o')
        lines[f"{ip}_avg"], = ax.plot([], [], label=f"{ip} ({method}) Avg", linestyle='--')
    
    ax.legend()
    return fig, ax, lines

def update_plot(frame, ax, lines):
    """Update the plot with new data from the queue."""
    items_processed = 0
    while not data_queue.empty() and items_processed < 50:  # Limit processing to avoid lag
        try:
            item = data_queue.get_nowait()
            if len(item) == 5:  # Error case
                ip, _, _, avg_ms, error = item
                print(f"Error fetching from {ip}: {error} (Avg: {avg_ms:.2f} ms)")
            else:
                ip, angles, time_ms, avg_ms = item
                print(f"Got angles {angles} from {ip} ({ips[ip]}) ({time_ms:.2f} ms, Avg: {avg_ms:.2f} ms)")
            items_processed += 1
        except queue.Queue.Empty:
            break
    
    # Update plot data
    for ip in ips.keys():
        with fetch_times_lock:
            times = fetch_times[ip]
            if times:
                # Adjust x-axis to show the last MAX_FETCH_HISTORY points
                start_idx = max(0, len(times) - MAX_FETCH_HISTORY)
                x = np.arange(start_idx, len(times))
                y = times[-MAX_FETCH_HISTORY:]
                lines[f"{ip}_data"].set_data(x, y)
                avg = np.mean(times) if times else 0
                lines[f"{ip}_avg"].set_data(x, [avg] * len(x))
    
    # Adjust plot limits
    max_times = max([len(times) for times in fetch_times.values()], default=1)
    max_ms = max([max(times, default=0) for times in fetch_times.values()], default=0)
    ax.set_xlim(max(0, max_times - MAX_FETCH_HISTORY), max_times)
    ax.set_ylim(0, max(100, max_ms * 1.2))  # Default to 100ms if max_ms is 0
    
    return list(lines.values())

if __name__ == "__main__":
    print("\nBenchmarking ESP32 Fetch Times: HTTP (100) vs (101)...")
    print("--------------------------")
    print("Press Ctrl+C to exit")
    print("--------------------------")

    # Start fetch threads for each ESP32
    fetch_threads = []
    for ip, method in ips.items():
        if method == "old":
            thread = threading.Thread(target=fetch_angles_http, args=(ip,), daemon=True)
        else:  # UDP
            thread = threading.Thread(target=fetch_angles_udp, args=(ip, udp_sockets[ip]), daemon=True)
        thread.start()
        fetch_threads.append(thread)

    # Set up and start the plot
    fig, ax, lines = setup_plot()
    ani = FuncAnimation(fig, update_plot, fargs=(ax, lines), interval=50, blit=True)  # Faster update interval

    try:
        plt.show()  # Blocks here until the plot is closed
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Stopping...")
    finally:
        running.clear()  # Signal threads to stop
        for thread in fetch_threads:
            thread.join(timeout=1.0)
        for sock in udp_sockets.values():
            sock.close()
        plt.close(fig)
        print("Cleanup complete. Exiting.")