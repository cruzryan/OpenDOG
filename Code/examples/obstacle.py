# intelligent_quad_control_v8.3_reshape_fixed.py
# Fixes the NumPy reshape error from the previous version.

# --- Core Python and System Imports ---
import os
import sys
import time
import threading
import math

# --- PyRay for Visualization ---
from pyray import *

# --- Perception and Processing ---
import numpy as np
import pyrealsense2 as rs
import torch
import scipy.ndimage

# --- Robot Control ---
from pynput import keyboard
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.join(os.path.dirname(script_dir), "quadpilot")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from body import QuadPilotBody
except ImportError:
    print(f"FATAL ERROR: Could not import QuadPilotBody."); sys.exit(1)

# ==============================================================================
# --- 1. CONFIGURATION AND INITIALIZATION ---
# ==============================================================================

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = True
        self.user_command = None
        self.robot_mode = "IDLE"
        self.current_yaw, self.original_target_yaw, self.current_target_yaw = 0.0, 0.0, 0.0
        self.estimated_position = np.array([0.0, 0.0, 0.0])
        self.estimated_path_history = []
        self.path_start_point = np.array([0.0, 0.0, 0.0])
        self.local_obstacles = []
        self.avoidance_timer = 0.0
        self.avoidance_direction = 1
        self.avoidance_turn_angle = 0.0

if not torch.cuda.is_available(): print("Error: CUDA is not available."); exit(1)
device = torch.device("cuda"); screenWidth, screenHeight = 1280, 720

NUM_MOTORS, ESP_WITH_IMU = 8, 1
CORRECTION_GAIN_KP = 1.5; NEUTRAL_LIFT_ANGLE = 30.0
MIN_LIFT_ANGLE, MAX_LIFT_ANGLE = 20.0, 45.0
WALK_STEP_DURATION, WALK_SPEED_MPS = 0.15, 0.15
PATH_CORRECTION_GAIN_KD = 45.0; MAX_PATH_CORRECTION_ANGLE_DEG = 35.0

DANGER_ZONE_DISTANCE = 0.7
WARNING_ZONE_DISTANCE = 1.5
AVOIDANCE_ANGLE_NORMAL = 20.0
AVOIDANCE_ANGLE_EMERGENCY = 45.0
AVOIDANCE_DURATION_S = 2.5

MAX_RENDER_DEPTH = 2.5; OBSTACLE_VOXEL_GRID_SIZE = 0.05
DANGER_ZONE_WIDTH = 0.4; DANGER_ZONE_MIN_HEIGHT_ABOVE_FLOOR = 0.08
DANGER_ZONE_MAX_HEIGHT_ABOVE_FLOOR = 0.8

ACTUATOR_NAME_TO_INDEX_MAP = {
    "FL_tigh_actuator": 3, "FL_knee_actuator": 0, "FR_tigh_actuator": 1, "FR_knee_actuator": 2,
    "BR_tigh_actuator": 5, "BR_knee_actuator": 4, "BL_tigh_actuator": 7, "BL_knee_actuator": 6,
}
MOTOR_PINS = [
    (39, 40, 41, 42), (16, 15, 7, 6), (17, 18, 5, 4), (37, 38, 1, 2),
    (37, 38, 1, 2), (40, 39, 42, 41), (15, 16, 6, 7), (18, 17, 4, 5),
]

# ==============================================================================
# --- 2. CORE FUNCTIONS ---
# ==============================================================================
def get_stance_angles():
    stance = [0.0] * NUM_MOTORS; stance[3]=45.0; stance[0]=45.0; stance[1]=-45.0; stance[2]=45.0;
    stance[5]=45.0; stance[4]=-45.0; stance[7]=45.0; stance[6]=-45.0; return stance

def initialize_hardware():
    print("Initializing RealSense..."); pipeline = rs.pipeline(); config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    try: pipeline.start(config); print("RealSense Pipeline started successfully!")
    except RuntimeError as e: print(f"Failed to start RealSense pipeline: {e}"); return None, None
    pc, s_filter = rs.pointcloud(), rs.spatial_filter(); s_filter.set_option(rs.option.filter_magnitude, 2); s_filter.set_option(rs.option.filter_smooth_alpha, 0.5); s_filter.set_option(rs.option.filter_smooth_delta, 20)
    print("Initializing QuadPilotBody..."); body_controller = None
    try:
        body_controller = QuadPilotBody(listen_for_broadcasts=True); time.sleep(2.0)
        body_controller.set_control_params(P=1.5, I=0.0, D=0.3, dead_zone=5, pos_thresh=5); body_controller.set_all_pins(MOTOR_PINS); time.sleep(0.5)
        body_controller.reset_all(); time.sleep(0.2); body_controller.set_all_control_status(True); print("QuadPilotBody initialized successfully.")
    except Exception as e:
        if pipeline: pipeline.stop();
        if body_controller: body_controller.close();
        print(f"FATAL: Failed during QuadPilotBody initialization: {e}"); return None, None
    return (pipeline, pc, s_filter), body_controller

# *** THE BUG FIX IS HERE ***
def get_point_cloud_numpy(rs_components):
    pipeline, pc, s_filter = rs_components
    try:
        frames = pipeline.wait_for_frames(1000)
    except RuntimeError:
        return None
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return None
    depth_frame = s_filter.process(depth_frame)
    points = pc.calculate(depth_frame)
    # The error was `--1` which should have been `-1`
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    valid_depth_mask = (verts[:, 2] > 0.1) & (verts[:, 2] < MAX_RENDER_DEPTH)
    return verts[valid_depth_mask]


def process_points_gpu(verts_gpu):
    if verts_gpu.shape[0] < 100: return []
    floor_y = torch.median(verts_gpu[verts_gpu[:, 1] < 0][:, 1]) if torch.any(verts_gpu[:, 1] < 0) else 0.0
    obstacle_verts_gpu = verts_gpu[verts_gpu[:, 1] >= (floor_y + DANGER_ZONE_MIN_HEIGHT_ABOVE_FLOOR)]
    if obstacle_verts_gpu.shape[0] < 20: return []
    danger_mask = (torch.abs(obstacle_verts_gpu[:, 0]) < DANGER_ZONE_WIDTH / 2)
    danger_points_gpu = obstacle_verts_gpu[danger_mask]
    if danger_points_gpu.shape[0] < 20: return []
    voxel_coords = (danger_points_gpu / OBSTACLE_VOXEL_GRID_SIZE).floor().int(); unique_coords, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
    unique_coords_cpu = unique_coords.cpu().numpy(); min_coords = unique_coords_cpu.min(axis=0)
    grid_shape = unique_coords_cpu.max(axis=0) - min_coords + 3; grid = np.zeros(grid_shape, dtype=bool)
    grid[unique_coords_cpu[:, 0]-min_coords[0]+1, unique_coords_cpu[:, 1]-min_coords[1]+1, unique_coords_cpu[:, 2]-min_coords[2]+1] = True
    labeled_grid, num_blobs = scipy.ndimage.label(grid, structure=np.ones((3,3,3)))
    if num_blobs == 0: return []
    obstacle_boxes = []; labels = labeled_grid[unique_coords_cpu[:, 0]-min_coords[0]+1, unique_coords_cpu[:, 1]-min_coords[1]+1, unique_coords_cpu[:, 2]-min_coords[2]+1]
    labels_gpu = torch.from_numpy(labels).to(device); full_point_labels = labels_gpu[inverse_indices]
    for i in range(1, num_blobs + 1):
        blob_points = danger_points_gpu[full_point_labels == i]
        if blob_points.shape[0] > 20:
            min_b, _ = torch.min(blob_points, dim=0); max_b, _ = torch.max(blob_points, dim=0)
            obstacle_boxes.append((min_b.cpu().numpy(), max_b.cpu().numpy()))
    return obstacle_boxes

def transform_obstacles_to_world_frame(local_obstacles, robot_pos, robot_yaw_deg):
    world_obstacles = []; yaw_rad = math.radians(robot_yaw_deg)
    cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
    def transform_point(p):
        lx, ly, lz = p[0], p[1], p[2]
        world_x_rel = lx * cos_yaw + lz * sin_yaw
        world_z_rel = -lx * sin_yaw + lz * cos_yaw
        return np.array([robot_pos[0] + world_x_rel, ly, robot_pos[2] + world_z_rel])
    for min_b_local, max_b_local in local_obstacles:
        corners = [ np.array([x,y,z]) for x in (min_b_local[0], max_b_local[0]) for y in (min_b_local[1], max_b_local[1]) for z in (min_b_local[2], max_b_local[2]) ]
        world_corners = [transform_point(c) for c in corners]; final_min = np.min(world_corners, axis=0); final_max = np.max(world_corners, axis=0)
        world_obstacles.append((final_min, final_max))
    return world_obstacles

def draw_scene(camera, state):
    with state.lock:
        local_obstacles, robot_pos = list(state.local_obstacles), state.estimated_position.copy()
        path_history, path_start = list(state.estimated_path_history), state.path_start_point.copy()
        current_target_yaw, original_target_yaw, current_yaw = state.current_target_yaw, state.original_target_yaw, state.current_yaw
        robot_mode = state.robot_mode; avoidance_timer = state.avoidance_timer
    world_obstacles = transform_obstacles_to_world_frame(local_obstacles, robot_pos, current_yaw)
    begin_mode_3d(camera)
    draw_grid(20, 0.5); draw_cube(Vector3(robot_pos[0], 0.05, robot_pos[2]), 0.2, 0.1, 0.3, BLUE)
    for min_b, max_b in world_obstacles:
        draw_bounding_box(BoundingBox(Vector3(min_b[0], min_b[1], min_b[2]), Vector3(max_b[0], max_b[1], max_b[2])), ORANGE)
    if len(path_history) > 1:
        for i in range(len(path_history) - 1): draw_line_3d(Vector3(path_history[i][0], 0.01, path_history[i][2]), Vector3(path_history[i+1][0], 0.01, path_history[i+1][2]), MAGENTA)
    yaw_orig_rad = math.radians(original_target_yaw); path_len = 8.0
    planned_path_start = Vector3(path_start[0], 0.01, path_start[2])
    planned_path_end = Vector3(path_start[0] - path_len * math.sin(yaw_orig_rad), 0.01, path_start[2] + path_len * math.cos(yaw_orig_rad))
    draw_line_3d(planned_path_start, planned_path_end, Color(0, 228, 48, 180))
    yaw_curr_rad = math.radians(current_target_yaw); line_len = 0.7
    curr_end_pos = Vector3(robot_pos[0] - line_len * math.sin(yaw_curr_rad), 0.05, robot_pos[2] + line_len * math.cos(yaw_curr_rad))
    draw_line_3d(Vector3(robot_pos[0], 0.05, robot_pos[2]), curr_end_pos, RED)
    end_mode_3d()
    draw_fps(10, 10); draw_text(f"Robot Mode: {robot_mode}", 10, 40, 20, BLUE if robot_mode == "WALKING" else ORANGE)
    if robot_mode == "AVOIDING": draw_text(f"  (Time Left: {avoidance_timer:.1f}s)", 180, 40, 20, ORANGE)
    draw_text(f"Original Yaw: {original_target_yaw:.1f}", 10, 70, 20, Color(0, 117, 44, 255))
    draw_text(f"Target Yaw:   {current_target_yaw:.1f}", 10, 100, 20, RED); draw_text(f"Current Yaw:  {current_yaw:.1f}", 10, 130, 20, BLACK)

def clamp(value, min_val, max_val): return max(min_val, min(value, max_val))

def execute_walk_step(body_controller, state):
    idx_fr_k, idx_bl_k = ACTUATOR_NAME_TO_INDEX_MAP["FR_knee_actuator"], ACTUATOR_NAME_TO_INDEX_MAP["BL_knee_actuator"]
    idx_fl_k, idx_br_k = ACTUATOR_NAME_TO_INDEX_MAP["FL_knee_actuator"], ACTUATOR_NAME_TO_INDEX_MAP["BR_knee_actuator"]
    stance = get_stance_angles()
    with state.lock: yaw_err = state.current_target_yaw - state.current_yaw
    while yaw_err > 180: yaw_err -= 360
    while yaw_err < -180: yaw_err += 360
    correction = CORRECTION_GAIN_KP * yaw_err
    N = clamp(NEUTRAL_LIFT_ANGLE - correction, MIN_LIFT_ANGLE, MAX_LIFT_ANGLE); Y = clamp(NEUTRAL_LIFT_ANGLE + correction, MIN_LIFT_ANGLE, MAX_LIFT_ANGLE)
    step = stance[:]; step[idx_fr_k] = Y; step[idx_bl_k] = -N; body_controller.set_angles(step); time.sleep(WALK_STEP_DURATION)
    body_controller.set_angles(stance); time.sleep(WALK_STEP_DURATION)
    step = stance[:]; step[idx_fl_k] = N; step[idx_br_k] = -Y; body_controller.set_angles(step); time.sleep(WALK_STEP_DURATION)
    body_controller.set_angles(stance); time.sleep(WALK_STEP_DURATION)

def robot_control_thread_func(body_controller, state):
    print("Robot control thread started.")
    last_update_time = time.perf_counter()
    while state.is_running:
        dt = time.perf_counter() - last_update_time; last_update_time = time.perf_counter()
        with state.lock: command = state.user_command; state.user_command = None
        if command:
            if command == "start_walk":
                dmp_data = body_controller.get_latest_dmp_data_for_esp(ESP_WITH_IMU)
                if dmp_data and dmp_data.get('ypr_deg'):
                    heading = dmp_data['ypr_deg'].get('yaw', 0.0)
                    with state.lock:
                        state.robot_mode = "WALKING"; state.original_target_yaw = heading
                        state.estimated_position = np.array([0.0, 0.0, 0.0]); state.path_start_point = state.estimated_position.copy()
                        state.estimated_path_history = [state.estimated_position.copy()]
                    print(f"Starting walk. Original Yaw captured: {heading:.1f} deg.")
                else: print("Warning: Cannot start walk, no IMU data available.")
            elif command in ["stop_walk", "stance", "zero"]:
                with state.lock: state.robot_mode = "IDLE"; state.avoidance_timer = 0.0
                if command == "stop_walk": print("Stopping walk. Returning to Stance.")
                angles = get_stance_angles() if command != "zero" else [0.0] * NUM_MOTORS
                body_controller.set_angles(angles)

        dmp_data = body_controller.get_latest_dmp_data_for_esp(ESP_WITH_IMU)
        if dmp_data and dmp_data.get('ypr_deg'):
            with state.lock: state.current_yaw = dmp_data['ypr_deg'].get('yaw', state.current_yaw)

        with state.lock: current_mode = state.robot_mode
        if current_mode == "IDLE": time.sleep(0.1); continue

        if current_mode == "WALKING":
            closest_obstacle_dist, closest_obstacle_center_x = float('inf'), 0
            with state.lock:
                if state.local_obstacles:
                    for min_b, max_b in state.local_obstacles:
                        if min_b[2] < closest_obstacle_dist:
                            closest_obstacle_dist = min_b[2]
                            closest_obstacle_center_x = (min_b[0] + max_b[0]) / 2
            if closest_obstacle_dist < WARNING_ZONE_DISTANCE:
                with state.lock:
                    state.avoidance_direction = -1 if closest_obstacle_center_x > 0 else 1
                    state.avoidance_timer = AVOIDANCE_DURATION_S
                    state.robot_mode = "AVOIDING"
                    if closest_obstacle_dist < DANGER_ZONE_DISTANCE:
                        print(f"!!! DANGER ZONE at {closest_obstacle_dist:.2f}m. Emergency Turn! !!!")
                        state.avoidance_turn_angle = AVOIDANCE_ANGLE_EMERGENCY
                    else:
                        print(f"Warning Zone at {closest_obstacle_dist:.2f}m. Normal Turn.")
                        state.avoidance_turn_angle = AVOIDANCE_ANGLE_NORMAL
            else:
                with state.lock:
                    orig_yaw, robot_pos, path_start = state.original_target_yaw, state.estimated_position, state.path_start_point
                    orig_yaw_rad = math.radians(orig_yaw)
                    line_dir_x, line_dir_z = -math.sin(orig_yaw_rad), math.cos(orig_yaw_rad)
                    vec_to_robot_x, vec_to_robot_z = robot_pos[0] - path_start[0], robot_pos[2] - path_start[2]
                    signed_dist = line_dir_x * vec_to_robot_z - line_dir_z * vec_to_robot_x
                    path_correction_angle = -signed_dist * PATH_CORRECTION_GAIN_KD
                    state.current_target_yaw = orig_yaw + clamp(path_correction_angle, -MAX_PATH_CORRECTION_ANGLE_DEG, MAX_PATH_CORRECTION_ANGLE_DEG)
        elif current_mode == "AVOIDING":
            with state.lock:
                state.avoidance_timer -= dt
                if state.avoidance_timer <= 0:
                    print("Avoidance maneuver complete. -> WALKING (Re-acquiring path)")
                    state.robot_mode = "WALKING"; state.avoidance_timer = 0
                else:
                    turn = state.avoidance_turn_angle * state.avoidance_direction
                    state.current_target_yaw = state.original_target_yaw + turn

        execute_walk_step(body_controller, state)
        with state.lock:
            yaw_rad = math.radians(state.current_yaw)
            state.estimated_position[0] -= WALK_SPEED_MPS * dt * math.sin(yaw_rad)
            state.estimated_position[2] += WALK_SPEED_MPS * dt * math.cos(yaw_rad)
            state.estimated_path_history.append(state.estimated_position.copy())
            if len(state.estimated_path_history) > 200: state.estimated_path_history.pop(0)

def on_key_press(key, state):
    try:
        if hasattr(key, 'char'):
            with state.lock:
                char_map = {'y': "start_walk", 's': "stop_walk", 'a': "stance", 'd': "zero"}
                if key.char.lower() in char_map: state.user_command = char_map[key.char.lower()]
    except Exception as e: print(f"Error in on_key_press: {e}")

def main():
    rs_components, body_controller = initialize_hardware()
    if not rs_components or not body_controller: return
    shared_state = SharedState()
    control_thread = threading.Thread(target=robot_control_thread_func, args=(body_controller, shared_state), daemon=True); control_thread.start()
    key_listener = keyboard.Listener(on_press=lambda key: on_key_press(key, shared_state)); key_listener.start()
    init_window(screenWidth, screenHeight, "Intelligent Quadruped Control v8.3 (Reshape Fixed)")
    camera = Camera3D(); camera.position = Vector3(0.0, 4.0, -4.0); camera.target = Vector3(0.0, 0.0, 0.0)
    camera.up = Vector3(0.0, 1.0, 0.0); camera.fovy = 60.0; camera.projection = CAMERA_PERSPECTIVE
    set_target_fps(60); body_controller.set_angles(get_stance_angles())
    print("\n" + "="*50 + "\nSystem Ready. This version uses timed, tiered avoidance.\n" + "="*50 + "\n")
    try:
        while not window_should_close() and shared_state.is_running:
            verts_np = get_point_cloud_numpy(rs_components)
            if verts_np is not None:
                verts_gpu = torch.from_numpy(verts_np).to(device, non_blocking=True)
                with shared_state.lock: shared_state.local_obstacles = process_points_gpu(verts_gpu)
            update_camera(camera, CameraMode.CAMERA_FREE); begin_drawing(); clear_background(RAYWHITE)
            draw_scene(camera, shared_state); end_drawing()
    finally:
        print("\nShutdown sequence initiated..."); shared_state.is_running = False
        if key_listener.is_alive(): key_listener.stop()
        print("Waiting for control thread..."); control_thread.join(timeout=2.0)
        if body_controller:
            print("Disabling motors..."); body_controller.set_all_control_status(False); time.sleep(0.1)
            body_controller.reset_all(); body_controller.close()
        if rs_components: print("Stopping RealSense..."); rs_components[0].stop()
        close_window(); print("Shutdown complete.")

if __name__ == '__main__':
    main()