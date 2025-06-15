from transformers import pipeline
from PIL import Image
import cv2
import numpy as np
import torch

# Optimize torch for inference
try:
    torch.set_num_threads(1)
except Exception:
    pass

# Use GPU if available, and try half-precision
pipe_device = 0 if torch.cuda.is_available() else -1
model_id = "depth-anything/Depth-Anything-V2-Small-hf"  # Use small model for speed
pipe = pipeline(
    task="depth-estimation",
    model=model_id,
    device=pipe_device,
    torch_dtype=torch.float32,
    use_fast=True,
)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Convert frame (BGR) to RGB and then to PIL Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        # Inference
        depth = pipe(pil_image)["depth"]
        # Convert depth to numpy array and normalize for display
        depth_np = np.array(depth)
        depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        # Show depth map
        cv2.imshow('Depth Estimation', depth_uint8)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()