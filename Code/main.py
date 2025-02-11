import cv2
import numpy as np
import requests
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
from http.client import IncompleteRead  # Import IncompleteRead explicitly

class CameraApp:
    def __init__(self, root, stream_url, control_url):
        self.root = root
        self.stream_url = stream_url
        self.control_url = control_url
        self.root.title("ESP32 Camera Stream")

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.framesize_var = tk.StringVar(value="VGA")
        self.framesize_menu = ttk.OptionMenu(root, self.framesize_var, "VGA",
                                             "96X96", "QQVGA", "128X128", "QCIF", "HQVGA",
                                             "240X240", "QVGA", "320X320", "CIF", "HVGA",
                                             "VGA", "SVGA", "XGA", "HD", "SXGA", "UXGA",
                                             command=self.change_framesize)
        self.framesize_menu.pack()

        self.status_label = tk.Label(root, text="")
        self.status_label.pack()

        self.running = True
        self.thread = threading.Thread(target=self.update_frame)
        self.thread.start()

    def update_frame(self):
        while self.running:
            try:
                response = requests.get(self.stream_url, stream=True, timeout=10)
                response.raise_for_status()
                bytes_ = bytes()
                for chunk in response.iter_content(chunk_size=1024):
                    bytes_ += chunk
                    a = bytes_.find(b'\xff\xd8')
                    b = bytes_.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg = bytes_[a:b+2]
                        bytes_ = bytes_[b+2:]
                        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(img)
                            imgtk = ImageTk.PhotoImage(image=img)
                            self.image_label.imgtk = imgtk
                            self.image_label.configure(image=imgtk)
                        else:
                            print("Error decoding JPEG frame")
            except requests.exceptions.RequestException as e:
                self.status_label.config(text=f"Streaming Error: {e}")
                print(f"Streaming Error: {e}")
                time.sleep(1)
            except IncompleteRead as e: # Catch IncompleteRead specifically
                self.status_label.config(text=f"Streaming Error: IncompleteRead - Reconnecting...")
                print(f"Streaming Error: IncompleteRead - Reconnecting...")
                time.sleep(1) # Wait before reconnecting
                continue # Go to the next iteration of the loop to retry streaming
            except Exception as e:
                self.status_label.config(text=f"Unexpected Streaming Error: {e}")
                print(f"Unexpected Streaming Error: {e}")
                time.sleep(1)

    def change_framesize(self, event):
        framesize = self.framesize_var.get()
        try:
            response = requests.post(self.control_url, data=f"framesize:{framesize}", timeout=30) # Timeout remains at 30 seconds
            response.raise_for_status()
            if response.status_code == 200:
                self.status_label.config(text=f"Framesize changed to {framesize}")
                print(f"Framesize changed to {framesize}")
            else:
                self.status_label.config(text=f"Error changing framesize: HTTP {response.status_code}")
                print(f"Error changing framesize: HTTP {response.status_code}")

        except requests.exceptions.RequestException as e:
            self.status_label.config(text=f"Error changing framesize: {e}")
            print(f"Error changing framesize: {e}")
        except Exception as e:
            self.status_label.config(text=f"Unexpected Error changing framesize: {e}")
            print(f"Unexpected Error changing framesize: {e}")


    def on_closing(self):
        self.running = False
        self.thread.join()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root, "http://192.168.0.131:81/stream", "http://192.168.0.131:81/control")
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()