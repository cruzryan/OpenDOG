from pyray import *
import pyray
import math
import numpy as np
import pyrealsense2 as rs
import time
import threading
import open3d as o3d
import copy
from collections import defaultdict

# Initialize RealSense with robust setup
pipeline = rs.pipeline()
config = rs.config()

# Check for connected devices
ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print("Error: No RealSense device connected. Please plug in your device and try again.")
    exit(1)

# Print device info for debugging
device = devices[0]
print(f"Detected device: {device.get_info(rs.camera_info.name)}")

# Configure streams with explicit settings
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth at 640x480, 30 FPS
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)  # Color at 640x480, 30 FPS
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)   # Gyro at 200 Hz (optional)

# Start pipeline with error handling
try:
    pipeline.start(config)
    print("Pipeline started successfully!")
except RuntimeError as e:
    print(f"RuntimeError: {str(e)}")
    if "Couldn't resolve requests" in str(e):
        print("Trying to start pipeline without gyro stream (device may lack IMU)...")
        config.disable_stream(rs.stream.gyro)
        try:
            pipeline.start(config)
            print("Pipeline started without gyro stream.")
            has_gyro = False
        except RuntimeError as e2:
            print(f"Failed to start pipeline even without gyro: {str(e2)}")
            exit(1)
    else:
        print(f"Failed to start pipeline: {str(e)}")
        exit(1)
else:
    has_gyro = True  # Gyro stream is available

# Get stream profiles and intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Point cloud setup
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** 1)

# PyRay Window
screenWidth = 1000
screenHeight = 1000
init_window(screenWidth, screenHeight, "SLAM with Loop Closure and Color")

# Camera
camera = Camera3D()
camera.position = Vector3(0.0, 50.0, -40.0)
camera.target = Vector3(0.0, 5.0, 5.0)
camera.up = Vector3(0.0, 2.0, 1.0)
camera.fovy = 70.0
camera.projection = CAMERA_PERSPECTIVE
set_target_fps(60)

# Voxel Settings
render_voxels = True
voxel_size_base = 0.5
max_render_depth = 9.0
voxel_grid = {}  # Dictionary: key=voxel_index, value=(avg_color_float, count)
GREEN = Color(0, 255, 0, 255)

# IMU Data
imu_yaw_global = 0.0
imu_pitch_global = 0.0
imu_roll_global = 0.0
imu_thread_running = True
current_rotation_zyx = np.array([0.0, 0.0, 0.0])  # [Yaw, Pitch, Roll] in degrees

# Pose Tracking
current_pose = np.eye(4)
keyframes = []
keyframe_interval = 10
frame_count = 0

def imu_reading_thread_function():
    """Thread to update IMU rotation if gyro is available."""
    global imu_yaw_global, imu_pitch_global, imu_roll_global, imu_thread_running, current_rotation_zyx
    if not has_gyro:
        print("No gyro stream available. IMU thread will not update rotation.")
        return
    last_imu_time = time.time()
    while imu_thread_running:
        frames = pipeline.wait_for_frames()
        gyro_frame = frames.first_or_default(rs.stream.gyro)
        current_imu_time = time.time()
        time_step = current_imu_time - last_imu_time
        last_imu_time = current_imu_time
        if gyro_frame:
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            gyro_x = gyro_data.x  # Pitch
            gyro_y = gyro_data.y  # Yaw
            gyro_z = gyro_data.z  # Roll
            rotation_speed_factor = 0.01
            rotation_angle_increment_x_degrees = gyro_x * rotation_speed_factor * time_step * 180 / math.pi
            rotation_angle_increment_y_degrees = gyro_y * rotation_speed_factor * time_step * 180 / math.pi
            rotation_angle_increment_z_degrees = gyro_z * rotation_speed_factor * time_step * 180 / math.pi
            current_rotation_zyx[0] += rotation_angle_increment_z_degrees  # Yaw
            current_rotation_zyx[1] += rotation_angle_increment_y_degrees  # Pitch
            current_rotation_zyx[2] += rotation_angle_increment_x_degrees  # Roll
            imu_yaw_global = -current_rotation_zyx[0] * 90
            imu_pitch_global = -current_rotation_zyx[1] * 90
            imu_roll_global = -current_rotation_zyx[2] * 90
        time.sleep(0.001)

imu_thread = threading.Thread(target=imu_reading_thread_function)
imu_thread.daemon = True
imu_thread.start()

def get_point_cloud_with_color():
    """Capture point cloud with color from depth and color frames."""
    frames = pipeline.wait_for_frames()
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None

    color_image = np.asanyarray(color_frame.get_data())
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    v = points.get_vertices()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)

    uvs = np.asanyarray(points.get_texture_coordinates())
    if uvs.size == 0 or uvs.shape[0] != verts.shape[0]:
        return verts, np.zeros((verts.shape[0], 3), dtype=np.uint8)

    try:
        uvs = uvs.view(np.float32).reshape(-1, 2)
    except (ValueError, AttributeError):
        if uvs.dtype.names is not None and 'u' in uvs.dtype.names and 'v' in uvs.dtype.names:
            uvs = np.column_stack((uvs['u'], uvs['v']))
        else:
            return verts, np.zeros((verts.shape[0], 3), dtype=np.uint8)

    colors = np.zeros((verts.shape[0], 3), dtype=np.uint8)
    valid_uv_mask = (uvs[:, 0] >= 0) & (uvs[:, 0] < 1) & (uvs[:, 1] >= 0) & (uvs[:, 1] < 1)
    if np.any(valid_uv_mask):
        valid_uvs = uvs[valid_uv_mask]
        x = (valid_uvs[:, 0] * (color_image.shape[1] - 1)).astype(int)
        y = (valid_uvs[:, 1] * (color_image.shape[0] - 1)).astype(int)
        colors[valid_uv_mask] = color_image[y, x]

    valid_depth_mask = (verts[:, 2] > 0) & (verts[:, 2] < max_render_depth)
    valid_verts = verts[valid_depth_mask]
    valid_colors = colors[valid_depth_mask]

    return valid_verts, valid_colors

def apply_rotation(verts, yaw, pitch, roll):
    """Apply inverse rotation to align points to global frame."""
    yaw_rad = np.radians(-yaw)
    pitch_rad = np.radians(-pitch)
    roll_rad = np.radians(-roll)
    rotation_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                           [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                           [0, 0, 1]])
    rotation_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                           [0, 1, 0],
                           [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(roll_rad), -np.sin(roll_rad)],
                           [0, np.sin(roll_rad), np.cos(roll_rad)]])
    rotation_matrix = rotation_x @ rotation_y @ rotation_z
    rotated_verts = verts @ rotation_matrix.T
    return rotated_verts

def detect_loop(current_verts, current_pose):
    """Detect loop closure with ICP."""
    for idx, (kf_verts, kf_pose, _) in enumerate(keyframes[:-5]):
        current_pcd = o3d.geometry.PointCloud()
        current_pcd.points = o3d.utility.Vector3dVector(current_verts)
        kf_pcd = o3d.geometry.PointCloud()
        kf_pcd.points = o3d.utility.Vector3dVector(kf_verts)
        current_down = current_pcd.voxel_down_sample(voxel_size=0.1)
        kf_down = kf_pcd.voxel_down_sample(voxel_size=0.1)
        relative_pose = np.linalg.inv(kf_pose) @ current_pose
        result = o3d.pipelines.registration.registration_icp(
            current_down, kf_down, max_correspondence_distance=0.5,
            init=relative_pose,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
        )
        if result.inlier_rmse < 0.05 and result.fitness > 0.5:
            return idx, result.transformation
    return None, None

def optimize_pose_graph(poses, loop_idx, loop_transform):
    """Optimize poses with loop closure."""
    correction = np.linalg.inv(loop_transform)
    for i in range(loop_idx + 1, len(poses)):
        poses[i] = correction @ poses[i]
    return poses

def rebuild_voxel_grid(poses, keyframe_verts, keyframe_colors):
    """Rebuild voxel map with corrected poses and colors."""
    global voxel_grid
    temp_voxel_colors = defaultdict(list)
    for pose, verts, colors in zip(poses, keyframe_verts, keyframe_colors):
        homogeneous_verts = np.hstack((verts, np.ones((verts.shape[0], 1))))
        transformed_verts = (pose @ homogeneous_verts.T).T[:, :3]
        scaled_verts = transformed_verts * 20
        voxel_coords = (scaled_verts / voxel_size_base).astype(int)
        for coord, color in zip(voxel_coords, colors):
            voxel_index = tuple(coord)
            temp_voxel_colors[voxel_index].append(color)

    voxel_grid = {}
    for voxel_index, colors in temp_voxel_colors.items():
        if colors:
            mean_color = np.mean(colors, axis=0)
            count = len(colors)
            voxel_grid[voxel_index] = (mean_color, count)

while not window_should_close():
    update_camera(camera, CameraMode.CAMERA_FREE)
    frame_count += 1

    valid_verts, valid_colors = get_point_cloud_with_color()
    if valid_verts is None:
        continue

    if valid_verts.size > 0:
        imu_yaw = imu_yaw_global if has_gyro else 0.0
        imu_pitch = imu_pitch_global if has_gyro else 0.0
        imu_roll = imu_roll_global if has_gyro else 0.0
        rotated_verts = apply_rotation(valid_verts, imu_yaw, imu_pitch, imu_roll)

        yaw_rad = np.radians(-imu_yaw / 90)
        pitch_rad = np.radians(-imu_pitch / 90)
        roll_rad = np.radians(-imu_roll / 90)
        R_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                        [0, 0, 1]])
        R_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                        [0, 1, 0],
                        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll_rad), -np.sin(roll_rad)],
                        [0, np.sin(roll_rad), np.cos(roll_rad)]])
        R = R_x @ R_y @ R_z
        current_pose[:3, :3] = R

        temp_voxel_colors = defaultdict(list)
        scaled_verts = rotated_verts * 20
        voxel_coords = (scaled_verts / voxel_size_base).astype(int)
        for i, coord in enumerate(voxel_coords):
            voxel_index = tuple(coord)
            temp_voxel_colors[voxel_index].append(valid_colors[i])

        for voxel_index, colors in temp_voxel_colors.items():
            if colors:
                mean_new = np.mean(colors, axis=0)
                m = len(colors)
                if voxel_index not in voxel_grid:
                    voxel_grid[voxel_index] = (mean_new, m)
                else:
                    avg_color, count = voxel_grid[voxel_index]
                    new_avg = (avg_color * count + mean_new * m) / (count + m)
                    new_count = count + m
                    voxel_grid[voxel_index] = (new_avg, new_count)

        if frame_count % keyframe_interval == 0:
            keyframes.append((valid_verts.copy(), current_pose.copy(), valid_colors.copy()))
            if len(keyframes) > 5:
                loop_idx, loop_transform = detect_loop(valid_verts, current_pose)
                if loop_idx is not None:
                    poses = [kf[1] for kf in keyframes]
                    corrected_poses = optimize_pose_graph(poses, loop_idx, loop_transform)
                    rebuild_voxel_grid(corrected_poses, [kf[0] for kf in keyframes], [kf[2] for kf in keyframes])
                    keyframes = [(kf[0], pose, kf[2]) for kf, pose in zip(keyframes, corrected_poses)]
                    print(f"Loop closure detected at keyframe {loop_idx}")

    begin_drawing()
    clear_background(RAYWHITE)
    begin_mode_3d(camera)
    draw_grid(10, 5.0)

    if render_voxels:
        for voxel_index, (avg_color, _) in voxel_grid.items():
            voxel_x, voxel_y, voxel_z = voxel_index
            render_color = np.clip(np.round(avg_color), 0, 255).astype(np.uint8)
            voxel_color = Color(render_color[0], render_color[1], render_color[2], 255)
            cubePos = Vector3(float(voxel_x) * voxel_size_base,
                             float(voxel_y) * voxel_size_base,
                             float(voxel_z) * voxel_size_base)
            draw_cube(cubePos, voxel_size_base, voxel_size_base, voxel_size_base, voxel_color)

    end_mode_3d()

    draw_text(f"IMU Rotation: Yaw: {imu_yaw_global:.2f}, Pitch: {imu_pitch_global:.2f}, Roll: {imu_roll_global:.2f}",
             10, 40, 20, BLACK)
    draw_text(f"Voxel Rendering: {'ON' if render_voxels else 'OFF'}",
             10, 120, 20, render_voxels and GREEN or RED)
    draw_text("Press V to toggle Voxel Rendering", 10, 140, 20, GRAY)
    draw_text("Press E to Export Voxel Map to .npy", 10, 160, 20, GRAY)
    draw_fps(10, 10)
    end_drawing()

    if is_key_pressed(KeyboardKey.KEY_V):
        render_voxels = not render_voxels
    if is_key_pressed(KeyboardKey.KEY_E):
        voxel_points = [[float(vx) * voxel_size_base,
                         float(vy) * voxel_size_base,
                         float(vz) * voxel_size_base]
                        for vx, vy, vz in voxel_grid.keys()]
        if voxel_points:
            np.save("voxel_map.npy", np.array(voxel_points))
            print("Voxel map exported to voxel_map.npy")

# Cleanup
close_window()
imu_thread_running = False
imu_thread.join()
pipeline.stop()