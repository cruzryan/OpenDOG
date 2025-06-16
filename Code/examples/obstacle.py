# intelligent_quad_control_v5.py
# Adds an "engagement distance" so the robot only reacts to immediate threats (<60cm).

# --- Core Python and System Imports ---
import os
import sys
import time
import threading
import math
from collections import defaultdict

# --- PyRay for Visualization ---
from pyray import *
import pyray

# --- Perception and Processing ---
import numpy as np
import pyrealsense2 as rs
import torch
import scipy.ndimage

# --- Robot Control ---
from pynput import keyboard
try:
    # This assumes your 'quadpilot' folder is a sibling to the directory
    # where you save this script. Adjust the path if necessary.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.join(os.path.dirname(script_dir), "quadpilot")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from body import QuadPilotBody
except ImportError:
    print(f"FATAL ERROR: Could not import QuadPilotBody.")
    print(f"Please ensure the 'quadpilot' directory is a sibling to this script's directory.")
    sys.exit(1)


# ==============================================================================
# --- 1. CONFIGURATION AND INITIALIZATION ---
# ==============================================================================

# --- Threading and State Management ---
class SharedState:
    """A thread-safe class to hold shared data between threads."""
    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = True
        self.user_command = None  # "start_walk", "stop_walk", "stance", "zero"

        # Robot State
        self.robot_mode = "IDLE"  # "IDLE", "WALKING", "AVOIDING"
        self.current_yaw = 0.0
        self.original_target_yaw = 0.0
        self.current_target_yaw = 0.0
        self.estimated_position = np.array([0.0, 0.0, 0.0]) # X, Y, Z world frame
        self.estimated_path_history = []

        # Perception State - Obstacles are stored in the CAMERA's local frame
        self.local_obstacles = [] # List of (min_b, max_b) tuples
        self.nearest_obstacle_index = -1 # Index of the nearest obstacle for rendering

# --- GPU and PyRay Setup ---
if not torch.cuda.is_available():
    print("Error: CUDA is not available. This script requires a CUDA-enabled GPU and PyTorch.")
    exit(1)
device = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

screenWidth = 1280
screenHeight = 720

# --- Robot Control Configuration ---
NUM_MOTORS = 8
TARGET_MOTORS = list(range(NUM_MOTORS))
ESP_WITH_IMU = 1

CORRECTION_GAIN_KP = 1.5
NEUTRAL_LIFT_ANGLE = 30.0
MIN_LIFT_ANGLE = 20.0
MAX_LIFT_ANGLE = 45.0
WALK_STEP_DURATION = 0.15
WALK_SPEED_MPS = 0.15

ACTUATOR_NAME_TO_INDEX_MAP = {
    "FL_tigh_actuator": 3, "FL_knee_actuator": 0, "FR_tigh_actuator": 1, "FR_knee_actuator": 2,
    "BR_tigh_actuator": 5, "BR_knee_actuator": 4, "BL_tigh_actuator": 7, "BL_knee_actuator": 6,
}
MOTOR_PINS = [
    (39, 40, 41, 42), (16, 15, 7, 6), (17, 18, 5, 4), (37, 38, 1, 2),
    (37, 38, 1, 2), (40, 39, 42, 41), (15, 16, 6, 7), (18, 17, 4, 5),
]

# --- Perception Configuration ---
MAX_RENDER_DEPTH = 2.5
OBSTACLE_VOXEL_GRID_SIZE = 0.05
DANGER_ZONE_WIDTH = 0.6
DANGER_ZONE_MIN_HEIGHT_ABOVE_FLOOR = 0.08
DANGER_ZONE_MAX_HEIGHT_ABOVE_FLOOR = 0.8
PATH_CORRECTION_INCREMENT_DEG = 2.0
AVOIDANCE_PATH_ANGLE_DEG = 25.0
PROBE_DEPTH = 1.5
PROBE_WIDTH = 0.35

### --- CHANGE: New constant for threat engagement distance --- ###
# The robot will only start an avoidance maneuver if an obstacle is closer than this distance (in meters).
ENGAGEMENT_DISTANCE = 0.6


# ==============================================================================
# --- 2. PERCEPTION AND VISUALIZATION FUNCTIONS (Unchanged from v4) ---
# ==============================================================================

def initialize_hardware():
    # --- RealSense ---
    print("Initializing RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    try:
        pipeline.start(config)
        print("RealSense Pipeline started successfully!")
    except RuntimeError as e:
        print(f"Failed to start RealSense pipeline: {e}")
        return None, None
    pc = rs.pointcloud()
    spatial_filter = rs.spatial_filter()
    spatial_filter.set_option(rs.option.filter_magnitude, 2)
    spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

    # --- Robot Body ---
    print("Initializing QuadPilotBody...")
    body_controller = None
    try:
        body_controller = QuadPilotBody(listen_for_broadcasts=True)
        time.sleep(2.0)
        if not body_controller.set_control_params(P=1.5, I=0.0, D=0.3, dead_zone=5, pos_thresh=5):
            raise Exception("Failed to set control parameters.")
        if not body_controller.set_all_pins(MOTOR_PINS):
            raise Exception("Failed to set motor pins.")
        time.sleep(0.5)
        if not body_controller.reset_all(): raise Exception("Failed to reset all motors.")
        time.sleep(0.2)
        if not body_controller.set_all_control_status(True):
            raise Exception("Failed to enable motor control.")
        print("QuadPilotBody initialized successfully.")
    except Exception as e:
        print(f"FATAL: Failed during QuadPilotBody initialization: {e}")
        if pipeline: pipeline.stop()
        if body_controller: body_controller.close()
        return None, None

    return (pipeline, pc, spatial_filter), body_controller

def get_point_cloud_numpy(rs_components):
    pipeline, pc, spatial_filter = rs_components
    try:
        frames = pipeline.wait_for_frames(1000)
    except RuntimeError: return None
    depth_frame = frames.get_depth_frame()
    if not depth_frame: return None
    
    depth_frame = spatial_filter.process(depth_frame)
    points = pc.calculate(depth_frame)
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    
    valid_depth_mask = (verts[:, 2] > 0.1) & (verts[:, 2] < MAX_RENDER_DEPTH)
    return verts[valid_depth_mask]

def cluster_voxel_blobs_cpu(danger_points_gpu):
    if danger_points_gpu.shape[0] < 20: return []
    voxel_coords = (danger_points_gpu / OBSTACLE_VOXEL_GRID_SIZE).floor().int()
    unique_coords, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
    unique_coords_cpu = unique_coords.cpu().numpy()
    min_coords = unique_coords_cpu.min(axis=0)
    grid_shape = unique_coords_cpu.max(axis=0) - min_coords + 3
    grid = np.zeros(grid_shape, dtype=bool)
    grid[unique_coords_cpu[:, 0] - min_coords[0] + 1, unique_coords_cpu[:, 1] - min_coords[1] + 1, unique_coords_cpu[:, 2] - min_coords[2] + 1] = True
    labeled_grid, num_blobs = scipy.ndimage.label(grid, structure=np.ones((3,3,3)))
    if num_blobs == 0: return []
    obstacle_boxes = []
    labels = labeled_grid[unique_coords_cpu[:, 0] - min_coords[0] + 1, unique_coords_cpu[:, 1] - min_coords[1] + 1, unique_coords_cpu[:, 2] - min_coords[2] + 1]
    labels_gpu = torch.from_numpy(labels).to(device)
    full_point_labels = labels_gpu[inverse_indices]
    for i in range(1, num_blobs + 1):
        blob_points = danger_points_gpu[full_point_labels == i]
        if blob_points.shape[0] > 20:
            min_b, _ = torch.min(blob_points, dim=0)
            max_b, _ = torch.max(blob_points, dim=0)
            obstacle_boxes.append((min_b.cpu().numpy(), max_b.cpu().numpy()))
    return obstacle_boxes

def process_points_gpu(verts_gpu):
    floor_candidates = verts_gpu[verts_gpu[:, 1] < 0]
    floor_y = torch.median(floor_candidates[:, 1]) if floor_candidates.shape[0] > 100 else 0.0
    is_floor_mask_gpu = verts_gpu[:, 1] < (floor_y + 0.02)
    obstacle_verts_gpu = verts_gpu[~is_floor_mask_gpu]
    dist_to_floor = obstacle_verts_gpu[:, 1] - floor_y
    danger_mask = ((torch.abs(obstacle_verts_gpu[:, 0]) < DANGER_ZONE_WIDTH / 2) & (dist_to_floor > DANGER_ZONE_MIN_HEIGHT_ABOVE_FLOOR) & (dist_to_floor < DANGER_ZONE_MAX_HEIGHT_ABOVE_FLOOR) & (obstacle_verts_gpu[:, 2] < MAX_RENDER_DEPTH))
    danger_points_gpu = obstacle_verts_gpu[danger_mask]
    return cluster_voxel_blobs_cpu(danger_points_gpu)

def transform_obstacles_to_world_frame(local_obstacles, robot_pos, robot_yaw_deg):
    world_obstacles = []
    yaw_rad = math.radians(robot_yaw_deg)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    def transform_point(p):
        lx, ly, lz = p[0], p[1], p[2]
        world_x_rot = lx * cos_yaw + lz * sin_yaw
        world_z_rot = -lx * sin_yaw + lz * cos_yaw
        world_x = world_x_rot + robot_pos[0]
        world_y = ly
        world_z = world_z_rot + robot_pos[2]
        return np.array([world_x, world_y, world_z])
    for min_b_local, max_b_local in local_obstacles:
        corners_local = [np.array([x, y, z]) for x in [min_b_local[0], max_b_local[0]] for y in [min_b_local[1], max_b_local[1]] for z in [min_b_local[2], max_b_local[2]]]
        corners_world = [transform_point(c) for c in corners_local]
        final_min = np.min(corners_world, axis=0)
        final_max = np.max(corners_world, axis=0)
        world_obstacles.append((final_min, final_max))
    return world_obstacles

def draw_scene(camera, state, fps, gpu_time):
    with state.lock:
        local_obstacles = list(state.local_obstacles)
        robot_pos = state.estimated_position.copy()
        path_history = list(state.estimated_path_history)
        current_target_yaw = state.current_target_yaw
        original_target_yaw = state.original_target_yaw
        current_yaw = state.current_yaw
        robot_mode = state.robot_mode
        nearest_obstacle_index = state.nearest_obstacle_index
    
    world_obstacles = transform_obstacles_to_world_frame(local_obstacles, robot_pos, current_yaw)
    
    begin_mode_3d(camera)
    draw_grid(20, 0.5)
    draw_cube(Vector3(robot_pos[0], 0.05, robot_pos[2]), 0.2, 0.1, 0.3, BLUE)

    if len(path_history) > 1:
        for i in range(len(path_history) - 1):
            p1, p2 = path_history[i], path_history[i+1]
            draw_line_3d(Vector3(p1[0], 0.01, p1[2]), Vector3(p2[0], 0.01, p2[2]), MAGENTA)
    
    yaw_orig_rad = math.radians(original_target_yaw)
    path_len = 8.0
    planned_path_start = Vector3(robot_pos[0], 0.01, robot_pos[2])
    planned_path_end = Vector3(robot_pos[0] + path_len * math.sin(yaw_orig_rad), 0.01, robot_pos[2] + path_len * math.cos(yaw_orig_rad))
    draw_line_3d(planned_path_start, planned_path_end, Color(0, 228, 48, 180))

    yaw_curr_rad = math.radians(current_target_yaw)
    line_len = 0.7
    curr_end_pos = Vector3(robot_pos[0] + line_len * math.sin(yaw_curr_rad), 0.05, robot_pos[2] + line_len * math.cos(yaw_curr_rad))
    draw_line_3d(Vector3(robot_pos[0], 0.05, robot_pos[2]), curr_end_pos, RED)
    
    for i, (min_b, max_b) in enumerate(world_obstacles):
        box_color = RED if i == nearest_obstacle_index else LIME
        box = BoundingBox(Vector3(min_b[0], min_b[1], min_b[2]), Vector3(max_b[0], max_b[1], max_b[2]))
        draw_bounding_box(box, box_color)

    end_mode_3d()

    draw_fps(10, 10)
    draw_text(f"GPU Time: {gpu_time:.2f} ms", 10, 40, 20, BLACK)
    draw_text(f"Robot Mode: {robot_mode}", 10, 70, 20, BLUE)
    draw_text(f"Original Yaw: {original_target_yaw:.1f}", 10, 100, 20, Color(0, 117, 44, 255))
    draw_text(f"Target Yaw:   {current_target_yaw:.1f}", 10, 130, 20, RED)
    draw_text(f"Current Yaw:  {current_yaw:.1f}", 10, 160, 20, BLACK)
    draw_text("Green Line: Planned Path", screenWidth - 250, 10, 20, Color(0, 117, 44, 255))
    draw_text("Magenta Line: Actual Path", screenWidth - 250, 40, 20, MAGENTA)
    draw_text("Red Line: Immediate Target", screenWidth - 250, 70, 20, RED)


# ==============================================================================
# --- 3. ROBOT CONTROL LOGIC ---
# ==============================================================================

def get_stance_angles():
    stance = [0.0] * NUM_MOTORS
    stance[ACTUATOR_NAME_TO_INDEX_MAP["FL_tigh_actuator"]] = 45.0; stance[ACTUATOR_NAME_TO_INDEX_MAP["FL_knee_actuator"]] = 45.0
    stance[ACTUATOR_NAME_TO_INDEX_MAP["FR_tigh_actuator"]] = -45.0; stance[ACTUATOR_NAME_TO_INDEX_MAP["FR_knee_actuator"]] = 45.0
    stance[ACTUATOR_NAME_TO_INDEX_MAP["BR_tigh_actuator"]] = 45.0; stance[ACTUATOR_NAME_TO_INDEX_MAP["BR_knee_actuator"]] = -45.0
    stance[ACTUATOR_NAME_TO_INDEX_MAP["BL_tigh_actuator"]] = 45.0; stance[ACTUATOR_NAME_TO_INDEX_MAP["BL_knee_actuator"]] = -45.0
    return stance

def clamp(value, min_val, max_val): return max(min_val, min(value, max_val))

def execute_walk_step(body_controller, state):
    idx_fr_knee = ACTUATOR_NAME_TO_INDEX_MAP["FR_knee_actuator"]; idx_bl_knee = ACTUATOR_NAME_TO_INDEX_MAP["BL_knee_actuator"]
    idx_fl_knee = ACTUATOR_NAME_TO_INDEX_MAP["FL_knee_actuator"]; idx_br_knee = ACTUATOR_NAME_TO_INDEX_MAP["BR_knee_actuator"]
    stance_pose = get_stance_angles()
    with state.lock:
        current_yaw = state.current_yaw
        target_yaw = state.current_target_yaw
    yaw_error = target_yaw - current_yaw
    while yaw_error > 180: yaw_error -= 360
    while yaw_error < -180: yaw_error += 360
    correction = CORRECTION_GAIN_KP * yaw_error
    N = clamp(NEUTRAL_LIFT_ANGLE - correction, MIN_LIFT_ANGLE, MAX_LIFT_ANGLE)
    Y = clamp(NEUTRAL_LIFT_ANGLE + correction, MIN_LIFT_ANGLE, MAX_LIFT_ANGLE)
    step_pose = stance_pose[:]; step_pose[idx_fr_knee] = Y; step_pose[idx_bl_knee] = -N
    body_controller.set_angles(step_pose); time.sleep(WALK_STEP_DURATION)
    body_controller.set_angles(stance_pose); time.sleep(WALK_STEP_DURATION)
    step_pose = stance_pose[:]; step_pose[idx_fl_knee] = N; step_pose[idx_br_knee] = -Y
    body_controller.set_angles(step_pose); time.sleep(WALK_STEP_DURATION)
    body_controller.set_angles(stance_pose); time.sleep(WALK_STEP_DURATION)

def find_nearest_obstacle(local_obstacles):
    if not local_obstacles: return -1, float('inf')
    nearest_dist, nearest_idx = float('inf'), -1
    for i, (min_b, _) in enumerate(local_obstacles):
        dist = min_b[2]
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_idx = i
    return nearest_idx, nearest_dist

def score_paths(local_obstacles):
    scores = {'left': 0.0, 'forward': 0.0, 'right': 0.0}
    if not local_obstacles: return scores
    
    forward_min_x = -PROBE_WIDTH / 2
    forward_max_x = PROBE_WIDTH / 2
    right_min_x = forward_max_x
    right_max_x = forward_max_x + PROBE_WIDTH
    left_min_x = -PROBE_WIDTH - forward_min_x
    left_max_x = forward_min_x

    for min_b, max_b in local_obstacles:
        if min_b[2] > PROBE_DEPTH: continue
        threat_score = 1.0 / (min_b[2] * min_b[2] + 0.1)
        obs_min_x, obs_max_x = min_b[0], max_b[0]
        if max(forward_min_x, obs_min_x) < min(forward_max_x, obs_max_x):
            scores['forward'] += threat_score
        if max(right_min_x, obs_min_x) < min(right_max_x, obs_max_x):
            scores['right'] += threat_score
        if max(left_min_x, obs_min_x) < min(left_max_x, obs_max_x):
            scores['left'] += threat_score
    return scores

def robot_control_thread_func(body_controller, state):
    print("Robot control thread started.")
    last_update_time = time.perf_counter()

    while state.is_running:
        command = None
        with state.lock:
            if state.user_command:
                command = state.user_command
                state.user_command = None
        
        if command == "start_walk":
            dmp_data = body_controller.get_latest_dmp_data_for_esp(ESP_WITH_IMU)
            if dmp_data and dmp_data.get('ypr_deg'):
                current_heading = dmp_data['ypr_deg'].get('yaw', 0.0)
                with state.lock:
                    state.robot_mode = "WALKING"
                    state.original_target_yaw = current_heading
                    state.current_target_yaw = current_heading
                    state.estimated_position = np.array([0.0, 0.0, 0.0])
                    state.estimated_path_history = [state.estimated_position.copy()]
                print(f"Starting walk. Original Yaw captured: {current_heading:.1f} deg.")
            else: print("Warning: Cannot start walk, no IMU data available.")
        elif command == "stop_walk":
            with state.lock: state.robot_mode = "IDLE"
            print("Stopping walk. Returning to Stance.")
            body_controller.set_angles(get_stance_angles())
        elif command in ["stance", "zero"]:
            with state.lock: state.robot_mode = "IDLE"
            angles = get_stance_angles() if command == "stance" else [0.0] * NUM_MOTORS
            body_controller.set_angles(angles)

        dmp_data = body_controller.get_latest_dmp_data_for_esp(ESP_WITH_IMU)
        if dmp_data and dmp_data.get('ypr_deg'):
            with state.lock:
                state.current_yaw = dmp_data['ypr_deg'].get('yaw', state.current_yaw)

        current_mode = ""
        with state.lock: current_mode = state.robot_mode

        if current_mode in ["WALKING", "AVOIDING"]:
            ### --- CHANGE: Main logic now hinges on the ENGAGEMENT_DISTANCE --- ###
            local_obs = []
            current_yaw = 0
            original_target_yaw = 0
            with state.lock:
                local_obs = list(state.local_obstacles)
                current_yaw = state.current_yaw
                original_target_yaw = state.original_target_yaw

            # 1. PRIMARY ASSESSMENT: Is there an immediate threat?
            nearest_idx, nearest_dist = find_nearest_obstacle(local_obs)
            is_immediate_threat = nearest_dist < ENGAGEMENT_DISTANCE

            with state.lock:
                state.nearest_obstacle_index = nearest_idx

                # 2. DECIDE AND UPDATE STATE based on immediate threat
                if is_immediate_threat:
                    # THREAT DETECTED -> Engage avoidance logic
                    if state.robot_mode == "WALKING":
                        print(f"IMMEDIATE THREAT at {nearest_dist:.2f}m! Engaging avoidance...")
                    
                    state.robot_mode = "AVOIDING"
                    
                    # Score paths to find the best escape route
                    path_scores = score_paths(local_obs)
                    best_path = 'forward'
                    if path_scores['left'] < path_scores[best_path]: best_path = 'left'
                    if path_scores['right'] < path_scores[best_path] - 0.1: best_path = 'right'

                    # Set target relative to current heading to find clear space
                    if best_path == 'left':
                        state.current_target_yaw = current_yaw + AVOIDANCE_PATH_ANGLE_DEG
                    elif best_path == 'right':
                        state.current_target_yaw = current_yaw - AVOIDANCE_PATH_ANGLE_DEG
                    else: # If forward is still best, turn away from nearest obstacle
                        state.current_target_yaw = current_yaw + AVOIDANCE_PATH_ANGLE_DEG
                else:
                    # NO IMMEDIATE THREAT -> Continue to global target
                    if state.robot_mode == "AVOIDING":
                        print("Threat cleared. Resuming original path.")
                    
                    state.robot_mode = "WALKING"

                    # Gradually steer back towards the original global path
                    yaw_diff = original_target_yaw - state.current_target_yaw
                    while yaw_diff > 180: yaw_diff -= 360
                    while yaw_diff < -180: yaw_diff += 360
                    
                    correction = clamp(yaw_diff, -PATH_CORRECTION_INCREMENT_DEG, PATH_CORRECTION_INCREMENT_DEG)
                    state.current_target_yaw += correction
                    if abs(yaw_diff) < 1.0: # Snap to target when very close
                        state.current_target_yaw = original_target_yaw

                # Normalize yaw
                while state.current_target_yaw > 180: state.current_target_yaw -= 360
                while state.current_target_yaw < -180: state.current_target_yaw += 360
            
            # 3. ACT
            execute_walk_step(body_controller, state)

            # 4. UPDATE POSITION (Dead Reckoning)
            dt = time.perf_counter() - last_update_time
            last_update_time = time.perf_counter()
            with state.lock:
                yaw_rad = math.radians(state.current_yaw)
                delta_z = WALK_SPEED_MPS * dt * math.cos(yaw_rad)
                delta_x = WALK_SPEED_MPS * dt * math.sin(yaw_rad)
                state.estimated_position[0] += delta_x
                state.estimated_position[2] += delta_z
                state.estimated_path_history.append(state.estimated_position.copy())
                if len(state.estimated_path_history) > 300:
                    state.estimated_path_history.pop(0)
        else:
            time.sleep(0.1)
            last_update_time = time.perf_counter()

# ==============================================================================
# --- 4. MAIN EXECUTION AND THREAD MANAGEMENT ---
# ==============================================================================

def on_key_press(key, state):
    try:
        if hasattr(key, 'char'):
            char = key.char.lower()
            with state.lock:
                if char == 'y': state.user_command = "start_walk"
                elif char == 's': state.user_command = "stop_walk"
                elif char == 'a': state.user_command = "stance"
                elif char == 'd': state.user_command = "zero"
    except Exception as e: print(f"Error in on_key_press: {e}")

def main():
    rs_components, body_controller = initialize_hardware()
    if not rs_components or not body_controller: return
    shared_state = SharedState()
    control_thread = threading.Thread(target=robot_control_thread_func, args=(body_controller, shared_state), daemon=True)
    control_thread.start()
    key_listener = keyboard.Listener(on_press=lambda key: on_key_press(key, shared_state))
    key_listener.start()
    
    init_window(screenWidth, screenHeight, "Intelligent Quadruped Control v5")
    camera = Camera3D()
    camera.position = Vector3(0.0, 4.0, -4.0); camera.target = Vector3(0.0, 0.0, 0.0)
    camera.up = Vector3(0.0, 1.0, 0.0); camera.fovy = 60.0; camera.projection = CAMERA_PERSPECTIVE
    set_target_fps(60)
    gpu_time = 0.0
    body_controller.set_angles(get_stance_angles())

    print("\n" + "="*50 + "\nSystem Ready. Visualization and Control are LIVE.\n" +
          "  'y': Start autonomous walk | 's': Stop walk\n" +
          "  'a': Go to Stance | 'd': Go to Zero position\n" +
          "  'ESC' in window to quit.\n" + "="*50 + "\n")

    try:
        while not window_should_close() and shared_state.is_running:
            valid_verts_np = get_point_cloud_numpy(rs_components)
            if valid_verts_np is not None and valid_verts_np.shape[0] > 0:
                start_time = time.perf_counter()
                verts_gpu = torch.from_numpy(valid_verts_np).to(device, non_blocking=True)
                local_boxes = process_points_gpu(verts_gpu)
                with shared_state.lock:
                    shared_state.local_obstacles = local_boxes
                gpu_time = (time.perf_counter() - start_time) * 1000

            update_camera(camera, CameraMode.CAMERA_FREE)
            begin_drawing()
            clear_background(RAYWHITE)
            draw_scene(camera, shared_state, get_fps(), gpu_time)
            end_drawing()
    finally:
        print("\nShutdown sequence initiated...")
        shared_state.is_running = False
        if key_listener.is_alive(): key_listener.stop()
        print("Waiting for control thread to finish..."); control_thread.join(timeout=2.0)
        if body_controller:
            print("Disabling motors and closing connection...")
            body_controller.set_all_control_status(False); time.sleep(0.1)
            body_controller.reset_all(); body_controller.close()
        if rs_components:
            print("Stopping RealSense pipeline..."); rs_components[0].stop()
        close_window()
        print("Shutdown complete.")

if __name__ == '__main__':
    main()