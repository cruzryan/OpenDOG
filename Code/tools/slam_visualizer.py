from pyray import *
import pyray
import math
import numpy as np
import pyrealsense2 as rs
import time
from scipy.spatial.transform import Rotation as R
import threading  

pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)  # Accelerometer
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)   # Gyroscope

pipeline.start(config)
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** 1)
screenWidth = 1000
screenHeight = 1000

init_window(screenWidth, screenHeight, "SLAM TT-2025-A025")

# Initialize the camera
camera = Camera3D()
camera.position = Vector3(0.0, 50.0, -40.0)
camera.target = Vector3(0.0, 5.0, 5.0)
camera.up = Vector3(0.0, 2.0, 1.0)
camera.fovy = 70.0
camera.projection = CAMERA_PERSPECTIVE

set_target_fps(60)

render_voxels = True
voxel_size_base = 1.0
voxel_size_far_multiplier = 10.0
max_depth_for_color = 3.0
min_depth_for_color = 0.5
max_depth_for_size_change = 1.0
max_render_depth = 5.0
voxel_grid = set()
MANUAL_RED_NP = np.array([255, 0, 0])
MANUAL_BLUE_NP = np.array([0, 0, 255])

# IMU Rotation Data - Global variables to be updated by IMU thread
imu_yaw_global = 0.0
imu_pitch_global = 0.0
imu_roll_global = 0.0
imu_thread_running = True  # Flag to control IMU thread
current_rotation_zyx = np.array([0.0, 0.0, 0.0]) # [Yaw, Pitch, Roll] in degrees


def imu_reading_thread_function():
    global imu_yaw_global, imu_pitch_global, imu_roll_global, imu_thread_running, current_rotation_zyx
    last_imu_time = time.time()

    while imu_thread_running:
        frames = pipeline.wait_for_frames()  # Wait for next set of frames (including motion)
        gyro_frame = frames.first_or_default(rs.stream.gyro)

        current_imu_time = time.time()
        time_step = current_imu_time - last_imu_time
        last_imu_time = current_imu_time

        if gyro_frame:
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()

            gyro_x = gyro_data.x
            gyro_y = gyro_data.y
            gyro_z = gyro_data.z

            rotation_speed_factor = 0.01 # Reduced factor

            rotation_angle_increment_x_degrees = gyro_x * rotation_speed_factor * time_step * 180/math.pi # to degrees
            rotation_angle_increment_y_degrees = gyro_y * rotation_speed_factor * time_step * 180/math.pi # to degrees
            rotation_angle_increment_z_degrees = gyro_z * rotation_speed_factor * time_step * 180/math.pi # to degrees

            current_rotation_zyx[0] += rotation_angle_increment_z_degrees # Yaw (Z)
            current_rotation_zyx[1] += rotation_angle_increment_y_degrees # Pitch (Y)
            current_rotation_zyx[2] += rotation_angle_increment_x_degrees # Roll (X)

            imu_yaw_raw = current_rotation_zyx[0]
            imu_pitch_raw = current_rotation_zyx[1]
            imu_roll_raw = current_rotation_zyx[2]


            imu_yaw_global = -imu_yaw_raw * 90  # Inverted & Mapped angles - Global variable update
            imu_pitch_global = -imu_pitch_raw * 90 # Inverted & Mapped angles - Global variable update
            imu_roll_global = -imu_roll_raw * 90  # Inverted & Mapped angles - Global variable update


        time.sleep(0.001) # Small sleep to reduce CPU usage in thread if needed


# Start the IMU reading thread
imu_thread = threading.Thread(target=imu_reading_thread_function)
imu_thread.daemon = True # Set as daemon thread so it exits when main thread exits
imu_thread.start()


while not window_should_close():
    update_camera(camera, CameraMode.CAMERA_FREE)

    frames = pipeline.wait_for_frames() # Get frames for depth in main loop
    depth_frame = frames.get_depth_frame()


    # Access latest IMU data from global variables - NO IMU READING HERE IN MAIN LOOP
    imu_yaw = imu_yaw_global
    imu_pitch = imu_pitch_global
    imu_roll = imu_roll_global


    if not depth_frame:
        continue

    points = pc.calculate(depth_frame)
    v = points.get_vertices()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)


    # Voxel Grid Update - ALWAYS update the voxel grid regardless of render_voxels flag
    if verts.size > 0:
        valid_depth_mask = verts[:, 2] > 0
        valid_depth_mask = valid_depth_mask & (verts[:, 2] < max_render_depth)
        valid_verts = verts[valid_depth_mask].copy()

        if valid_verts.size > 0:
            # **NUMPY ROTATION APPLIED HERE - INVERSE ROTATION**

            # Convert Euler angles (imu_yaw, imu_pitch, imu_roll) to radians - INVERSE ANGLES NOW
            yaw_rad = np.radians(-imu_yaw)   # Negate angles for inverse rotation
            pitch_rad = np.radians(-imu_pitch) # Negate angles for inverse rotation
            roll_rad = np.radians(-imu_roll)  # Negate angles for inverse rotation


            # Rotation matrices around Z, Y, and X axes respectively - INVERSE ROTATION
            rotation_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                                   [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                                   [0, 0, 1]])

            rotation_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                                   [0, 1, 0],
                                   [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])

            rotation_x = np.array([[1, 0, 0],
                                   [0, np.cos(roll_rad), -np.sin(roll_rad)],
                                   [0, np.sin(roll_rad), np.cos(roll_rad)]])


            # Apply rotations in ZYX order (Yaw, Pitch, Roll - extrinsic rotations) - INVERSE ROTATION
            rotated_verts = valid_verts.copy() # Start with a fresh copy
            rotation_matrix = rotation_x @ rotation_y @ rotation_z # Combined rotation matrix (ZYX order)
            rotated_verts = rotated_verts @ rotation_matrix.T


            scaled_verts = rotated_verts * 20
            voxel_coords = (scaled_verts / voxel_size_base).astype(int)

            for i in range(len(voxel_coords)):
                voxel_index = tuple(voxel_coords[i])
                depth_val = valid_verts[i][2]

                voxel_grid.add(voxel_index)


    begin_drawing()
    clear_background(RAYWHITE)
    begin_mode_3d(camera)
    draw_grid(10, 5.0)


    if render_voxels: # Voxel Rendering - Only render if render_voxels is True
        for voxel_index in voxel_grid:
            voxel_x, voxel_y, voxel_z = voxel_index

            depth_val = np.sqrt(voxel_x**2 + voxel_y**2 + voxel_z**2) / 20

            normalized_depth_color = np.clip((depth_val - min_depth_for_color) / (max_depth_for_color - min_depth_for_color), 0.0, 1.0)
            interpolated_color_np = MANUAL_BLUE_NP + (MANUAL_RED_NP - MANUAL_BLUE_NP) * normalized_depth_color
            interpolated_color_np = np.clip(interpolated_color_np, 0, 255).astype(np.uint8)
            voxel_color = Color(interpolated_color_np[0], interpolated_color_np[1], interpolated_color_np[2], 255)

            current_voxel_size = voxel_size_base
            cubePos = Vector3(float(voxel_x) * voxel_size_base, float(voxel_y) * voxel_size_base, float(voxel_z) * voxel_size_base)
            draw_cube(cubePos, current_voxel_size, current_voxel_size, current_voxel_size, voxel_color)

    end_mode_3d()

    # Display Rotation Info - MAPPED for display - read from global variables
    imu_yaw = imu_yaw_global
    imu_pitch = imu_pitch_global
    imu_roll = imu_roll_global

    draw_text(f"IMU Rotation (Euler - Threaded):", 10, 40, 20, BLACK)
    draw_text(f"  Yaw: {imu_yaw:.2f}", 10, 60, 20, BLACK)
    draw_text(f"  Pitch: {imu_pitch:.2f}", 10, 80, 20, BLACK)
    draw_text(f"  Roll: {imu_roll:.2f}", 10, 100, 20, BLACK)

    draw_text(f"Voxel Rendering: {'ON' if render_voxels else 'OFF'}", 10, 120, 20, render_voxels and GREEN or RED)
    draw_text("Press V to toggle Voxel Rendering", 10, 140, 20, GRAY)
    draw_text("Press E to Export Voxel Map to .npy", 10, 160, 20, GRAY) # Export instruction


    draw_fps(10, 10)
    end_drawing()

    if is_key_pressed(KeyboardKey.KEY_V):
        render_voxels = not render_voxels

    if is_key_pressed(KeyboardKey.KEY_E): # Export voxel map on 'E' press
        voxel_points_for_export = [] # Clear export list before exporting current voxel_grid
        for voxel_index in voxel_grid: # Iterate through voxel_grid set
            voxel_x, voxel_y, voxel_z = voxel_index
            voxel_world_pos = [float(voxel_x) * voxel_size_base, float(voxel_y) * voxel_size_base, float(voxel_z) * voxel_size_base]
            voxel_points_for_export.append(voxel_world_pos)

        if voxel_points_for_export: # Check if there are any voxels to export
            voxel_points_np = np.array(voxel_points_for_export)
            np.save("voxel_map.npy", voxel_points_np)
            print("Voxel map exported to voxel_map.npy")
        else:
            print("No voxels to export yet.")


close_window()
imu_thread_running = False # Signal IMU thread to stop
imu_thread.join() # Wait for IMU thread to finish


pipeline.stop()