from pyray import *
import pyray
import math
import numpy as np
import pyrealsense2 as rs
import time
from collections import defaultdict

# --- NEW: Import GPU and the new clustering library ---
import torch
import scipy.ndimage # Standard, reliable library for blob detection

# Check for CUDA and set the device
if not torch.cuda.is_available():
    print("Error: CUDA is not available. This script requires a CUDA-enabled GPU and PyTorch.")
    exit(1)
device = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")


# Initialize RealSense with robust setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

try:
    pipeline.start(config)
    print("Pipeline started successfully!")
except RuntimeError as e:
    print(f"Failed to start pipeline: {e}")
    exit(1)

# Point cloud setup and filters
pc = rs.pointcloud()
spatial_filter = rs.spatial_filter()
spatial_filter.set_option(rs.option.filter_magnitude, 2)
spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

# PyRay Window
screenWidth = 1000
screenHeight = 1000
init_window(screenWidth, screenHeight, "GPU Accelerated Obstacle Detection (No torch-cluster)")

# Camera
camera = Camera3D()
camera.position = Vector3(0.0, 0.8, -1.5)
camera.target = Vector3(0.0, 0.2, 0.0)
camera.up = Vector3(0.0, 1.0, 0.0)
camera.fovy = 70.0
camera.projection = CAMERA_PERSPECTIVE
set_target_fps(120)

# --- CONFIGURABLE SETTINGS ---
MAX_RENDER_DEPTH = 1.5
GREEN = Color(0, 255, 0, 255)
# --- NEW: Clustering grid size for blobs ---
OBSTACLE_VOXEL_GRID_SIZE = 0.05 # 5cm grid for finding connected blobs

VOXEL_SIZE_NEAR = 0.02
VOXEL_SIZE_FAR = 0.015
FLOOR_COLOR = Color(20, 160, 20, 255)
DANGER_ZONE_WIDTH = 0.4
DANGER_ZONE_MIN_HEIGHT_ABOVE_FLOOR = 0.02
DANGER_ZONE_MAX_HEIGHT_ABOVE_FLOOR = 0.8
TINT_RED_GPU = torch.tensor([255, 50, 50], device=device, dtype=torch.float32)
TINT_BLUE_GPU = torch.tensor([50, 50, 255], device=device, dtype=torch.float32)

def get_point_cloud_numpy():
    try:
        frames = pipeline.wait_for_frames(1000)
    except RuntimeError:
        return None
    depth_frame = frames.get_depth_frame()
    if not depth_frame: return None
    
    depth_frame = spatial_filter.process(depth_frame)
    points = pc.calculate(depth_frame)
    
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    
    valid_depth_mask = (verts[:, 2] > 0.1) & (verts[:, 2] < MAX_RENDER_DEPTH)
    return verts[valid_depth_mask]


# --- NEW: Voxel Blob Clustering (Replaces DBSCAN) ---
def cluster_voxel_blobs_cpu(danger_points_gpu):
    if danger_points_gpu.shape[0] < 20:
        return []

    # 1. Create a coarse grid of voxels just for clustering
    voxel_coords = (danger_points_gpu / OBSTACLE_VOXEL_GRID_SIZE).floor().int()
    
    # 2. Get unique voxel coordinates and their original point indices on GPU
    unique_coords, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
    
    # Transfer small amount of data to CPU
    unique_coords_cpu = unique_coords.cpu().numpy()
    
    # 3. Create a sparse 3D numpy array to represent the occupied voxel space
    # We add a buffer of 1 to handle edges correctly with scipy
    min_coords = unique_coords_cpu.min(axis=0)
    max_coords = unique_coords_cpu.max(axis=0)
    
    grid_shape = max_coords - min_coords + 3
    grid = np.zeros(grid_shape, dtype=bool)
    
    grid[
        unique_coords_cpu[:, 0] - min_coords[0] + 1,
        unique_coords_cpu[:, 1] - min_coords[1] + 1,
        unique_coords_cpu[:, 2] - min_coords[2] + 1
    ] = True
    
    # 4. Use SciPy to find connected "blobs" of voxels. This is super fast.
    labeled_grid, num_blobs = scipy.ndimage.label(grid)
    if num_blobs == 0:
        return []

    # 5. For each blob, find its points and calculate the bounding box
    obstacle_boxes = []
    # Create a map from grid coordinates back to the original points
    labels = labeled_grid[
        unique_coords_cpu[:, 0] - min_coords[0] + 1,
        unique_coords_cpu[:, 1] - min_coords[1] + 1,
        unique_coords_cpu[:, 2] - min_coords[2] + 1
    ]
    labels_gpu = torch.from_numpy(labels).to(device)
    full_point_labels = labels_gpu[inverse_indices]

    for i in range(1, num_blobs + 1):
        blob_points = danger_points_gpu[full_point_labels == i]
        if blob_points.shape[0] > 10: # Min points per valid blob
            min_b, _ = torch.min(blob_points, dim=0)
            max_b, _ = torch.max(blob_points, dim=0)
            
            # Calculate color on GPU
            avg_depth = torch.mean(blob_points[:, 2])
            norm_depth = torch.clamp(avg_depth / MAX_RENDER_DEPTH, 0.0, 1.0)
            color_gpu = (1.0 - norm_depth) * TINT_RED_GPU + norm_depth * TINT_BLUE_GPU
            
            obstacle_boxes.append((
                min_b.cpu().numpy(),
                max_b.cpu().numpy(),
                color_gpu.cpu().numpy().astype(np.uint8)
            ))
    
    return obstacle_boxes


# --- Main GPU Processing Function (Updated) ---
def process_points_gpu(verts_gpu):
    # 1. Fast Floor Detection
    floor_candidates = verts_gpu[verts_gpu[:, 1] < 0]
    floor_y = torch.median(floor_candidates[:, 1]) if floor_candidates.shape[0] > 100 else 0.0
    
    is_floor_mask_gpu = verts_gpu[:, 1] < (floor_y + 0.02)
    floor_verts_gpu = verts_gpu[is_floor_mask_gpu]
    obstacle_verts_gpu = verts_gpu[~is_floor_mask_gpu]

    # 2. Find Danger Zone Points
    dist_to_floor = obstacle_verts_gpu[:, 1] - floor_y
    danger_mask = (
        (torch.abs(obstacle_verts_gpu[:, 0]) < DANGER_ZONE_WIDTH / 2) &
        (dist_to_floor > DANGER_ZONE_MIN_HEIGHT_ABOVE_FLOOR) &
        (dist_to_floor < DANGER_ZONE_MAX_HEIGHT_ABOVE_FLOOR)
    )
    danger_points_gpu = obstacle_verts_gpu[danger_mask]
    
    # 3. Perform Voxel Blob Clustering
    obstacle_boxes_and_colors = cluster_voxel_blobs_cpu(danger_points_gpu)

    # 4. Voxelize for visualization (as before)
    def voxelize_for_drawing(points, voxel_size):
        if points.shape[0] == 0: return torch.empty((0,3), device=device), torch.empty((0,3), device=device)
        depths = points[:, 2]
        norm_depths = torch.clamp(depths / MAX_RENDER_DEPTH, 0.0, 1.0).unsqueeze(1)
        tinted_colors = (1.0 - norm_depths) * TINT_RED_GPU + norm_depths * TINT_BLUE_GPU
        voxel_coords = (points / voxel_size).floor().int()
        unique_coords, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
        perm = torch.arange(inverse_indices.size(0), dtype=inverse_indices.dtype, device=device)
        inverse_perm = torch.empty_like(perm).scatter_(0, inverse_indices, perm)
        unique_indices = inverse_perm[torch.unique(inverse_indices)]
        return unique_coords.float() * voxel_size, tinted_colors[unique_indices]

    near_mask_obs = obstacle_verts_gpu[:, 2] < VOXEL_SIZE_NEAR
    near_obstacle_centers, near_obstacle_colors = voxelize_for_drawing(obstacle_verts_gpu[near_mask_obs], VOXEL_SIZE_NEAR)
    far_obstacle_centers, far_obstacle_colors = voxelize_for_drawing(obstacle_verts_gpu[~near_mask_obs], VOXEL_SIZE_FAR)

    near_mask_floor = floor_verts_gpu[:, 2] < VOXEL_SIZE_NEAR
    near_floor_centers, _ = voxelize_for_drawing(floor_verts_gpu[near_mask_floor], VOXEL_SIZE_NEAR)
    far_floor_centers, _ = voxelize_for_drawing(floor_verts_gpu[~near_mask_floor], VOXEL_SIZE_FAR)

    return (
        obstacle_boxes_and_colors,
        near_obstacle_centers.cpu().numpy(), near_obstacle_colors.cpu().numpy().astype(np.uint8),
        far_obstacle_centers.cpu().numpy(), far_obstacle_colors.cpu().numpy().astype(np.uint8),
        near_floor_centers.cpu().numpy(),
        far_floor_centers.cpu().numpy()
    )


# Main loop
gpu_time = 0.0
while not window_should_close():
    update_camera(camera, CameraMode.CAMERA_FREE)
    
    valid_verts_np = get_point_cloud_numpy()
    
    # Init lists
    obstacle_boxes_and_colors_cpu = []
    near_obs_centers_cpu, near_obs_colors_cpu = np.array([]), np.array([])
    far_obs_centers_cpu, far_obs_colors_cpu = np.array([]), np.array([])
    near_floor_centers_cpu, far_floor_centers_cpu = np.array([]), np.array([])

    if valid_verts_np is not None and valid_verts_np.shape[0] > 0:
        start_time = time.perf_counter()
        
        verts_gpu = torch.from_numpy(valid_verts_np).to(device, non_blocking=True)
        
        (obstacle_boxes_and_colors_cpu,
         near_obs_centers_cpu, near_obs_colors_cpu,
         far_obs_centers_cpu, far_obs_colors_cpu,
         near_floor_centers_cpu, far_floor_centers_cpu) = process_points_gpu(verts_gpu)
         
        end_time = time.perf_counter()
        gpu_time = (end_time - start_time) * 1000

    # --- DRAWING BLOCK ---
    begin_drawing()
    clear_background(RAYWHITE)
    begin_mode_3d(camera)
    
    draw_grid(10, 1.0)

    for pos in near_floor_centers_cpu:
        draw_cube(Vector3(pos[0], pos[1], pos[2]), VOXEL_SIZE_NEAR, VOXEL_SIZE_NEAR, VOXEL_SIZE_NEAR, FLOOR_COLOR)
    for pos in far_floor_centers_cpu:
        draw_cube(Vector3(pos[0], pos[1], pos[2]), VOXEL_SIZE_FAR, VOXEL_SIZE_FAR, VOXEL_SIZE_FAR, FLOOR_COLOR)
    
    for i, pos in enumerate(near_obs_centers_cpu):
        r,g,b = near_obs_colors_cpu[i]
        draw_cube(Vector3(pos[0], pos[1], pos[2]), VOXEL_SIZE_NEAR, VOXEL_SIZE_NEAR, VOXEL_SIZE_NEAR, Color(r,g,b,255))
    for i, pos in enumerate(far_obs_centers_cpu):
        r,g,b = far_obs_colors_cpu[i]
        draw_cube(Vector3(pos[0], pos[1], pos[2]), VOXEL_SIZE_FAR, VOXEL_SIZE_FAR, VOXEL_SIZE_FAR, Color(r,g,b,255))
    
    for min_b, max_b, color in obstacle_boxes_and_colors_cpu:
        box = BoundingBox(Vector3(min_b[0], min_b[1], min_b[2]), Vector3(max_b[0], max_b[1], max_b[2]))
        r,g,b = color
        draw_bounding_box(box, Color(r,g,b,255))
        
    end_mode_3d()

    draw_fps(10, 10)
    draw_text(f"GPU Time: {gpu_time:.2f} ms", 10, 40, 20, BLACK)
    
    num_obstacles = len(obstacle_boxes_and_colors_cpu)
    if num_obstacles > 0:
        draw_text(f"Obstacles Detected: {num_obstacles}", 10, 70, 20, RED)
    else:
        draw_text("Path Clear", 10, 70, 20, GREEN)

    end_drawing()

close_window()
pipeline.stop()