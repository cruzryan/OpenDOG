import pyray
from pyray import *
import numpy as np
import pyrealsense2 as rs

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable depth stream (1024x768 at 30 FPS, as per L515 default)
config.enable_stream(rs.stream.depth, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

# Get depth stream profile to extract intrinsics (for point cloud calculation)
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Point cloud object to convert depth to 3D points
pc = rs.pointcloud()

# Initialize Raylib window
screenWidth = 1000
screenHeight = 1000
init_window(screenWidth, screenHeight, "L515 Up-to-Date Voxel Rendering")

# Set up the 3D camera
camera = Camera3D()
camera.position = Vector3(0.0, 50.0, -40.0)  # Positioned to view the voxel grid
camera.target = Vector3(0.0, 5.0, 5.0)       # Looking at the origin-ish
camera.up = Vector3(0.0, 1.0, 0.0)           # Up direction
camera.fovy = 70.0
camera.projection = CAMERA_PERSPECTIVE

set_target_fps(60)

# Voxel settings
voxel_size = 1.0  # Size of each voxel cube (in scaled space)
max_render_depth = 5.0  # Max depth to consider (in meters)
min_render_depth = 0.5  # Min depth to consider (avoid near-plane noise)
voxel_grid = set()  # Set to store voxel coordinates

# Color definitions (in RGB, 0-255)
BLUE_NP = np.array([0, 0, 255])
RED_NP = np.array([255, 0, 0])

while not window_should_close():
    # Update camera (lets you move it with mouse/keyboard)
    update_camera(camera, CameraMode.CAMERA_FREE)

    # Get depth frame from L515
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()

    if not depth_frame:
        continue

    # Clear the voxel grid to remove old voxels
    voxel_grid.clear()

    # Convert depth frame to 3D points
    points = pc.calculate(depth_frame)
    vertices = points.get_vertices()
    verts = np.asanyarray(vertices).view(np.float32).reshape(-1, 3)  # Shape: (N, 3) for x, y, z

    # Filter out invalid points (depth = 0 or outside min/max depth)
    valid_depth_mask = (verts[:, 2] > min_render_depth) & (verts[:, 2] < max_render_depth)
    valid_verts = verts[valid_depth_mask]

    # Voxelize the points (only from the current frame)
    if valid_verts.size > 0:
        # Scale the points (20x for visibility)
        scaled_verts = valid_verts * 20
        # Convert to voxel coordinates by dividing by voxel size and rounding to integers
        voxel_coords = (scaled_verts / voxel_size).astype(int)

        # Add voxel coordinates to the grid (as tuples for the set)
        for i in range(len(voxel_coords)):
            voxel_index = tuple(voxel_coords[i])
            voxel_grid.add(voxel_index)

    # Start drawing
    begin_drawing()
    clear_background(RAYWHITE)

    # Enter 3D mode with the camera
    begin_mode_3d(camera)

    # Draw a grid for reference
    draw_grid(10, 5.0)

    # Render the voxel grid with depth-based color
    for voxel_index in voxel_grid:
        voxel_x, voxel_y, voxel_z = voxel_index
        # Convert voxel coordinates back to world space
        cube_pos = Vector3(
            float(voxel_x) * voxel_size,
            float(voxel_y) * voxel_size,
            float(voxel_z) * voxel_size
        )
        # Calculate depth (z-coordinate in scaled space, divide by 20 to get back to meters)
        depth_val = abs(float(voxel_z) * voxel_size / 20)  # Use absolute z for simplicity

        # Normalize depth to [0, 1] based on min/max render depth
        normalized_depth = np.clip((depth_val - min_render_depth) / (max_render_depth - min_render_depth), 0.0, 1.0)
        # Interpolate between blue and red
        interpolated_color_np = BLUE_NP + (RED_NP - BLUE_NP) * normalized_depth
        interpolated_color_np = np.clip(interpolated_color_np, 0, 255).astype(np.uint8)
        voxel_color = Color(
            interpolated_color_np[0],  # Red
            interpolated_color_np[1],  # Green
            interpolated_color_np[2],  # Blue
            255                       # Alpha
        )

        # Draw the voxel cube with the calculated color
        draw_cube(cube_pos, voxel_size, voxel_size, voxel_size, voxel_color)

    end_mode_3d()

    # Show FPS for performance
    draw_fps(10, 10)
    end_drawing()

# Clean up
close_window()
pipeline.stop()