# intelligent_quad_control.py
# Combines GPU obstacle detection, robot control, and state-driven avoidance.

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
        self.estimated_position = np.array([0.0, 0.0, 0.0]) # X, Y, Z
        self.estimated_path_history = []

        # Perception State
        self.detected_obstacles = [] # List of (min_b, max_b, color) tuples

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
ESP_WITH_IMU = 1 # The ESP32 that sends the primary IMU/DMP data

# Yaw Auto-Correction Configuration
CORRECTION_GAIN_KP = 1.5
NEUTRAL_LIFT_ANGLE = 30.0
MIN_LIFT_ANGLE = 20.0
MAX_LIFT_ANGLE = 45.0
WALK_STEP_DURATION = 0.15
WALK_SPEED_MPS = 0.15 # Estimated speed in meters/sec for dead reckoning

ACTUATOR_NAME_TO_INDEX_MAP = {
    "FL_tigh_actuator": 3, "FL_knee_actuator": 0, "FR_tigh_actuator": 1, "FR_knee_actuator": 2,
    "BR_tigh_actuator": 5, "BR_knee_actuator": 4, "BL_tigh_actuator": 7, "BL_knee_actuator": 6,
}
MOTOR_PINS = [
    (39, 40, 41, 42), (16, 15, 7, 6), (17, 18, 5, 4), (37, 38, 1, 2),
    (37, 38, 1, 2), (40, 39, 42, 41), (15, 16, 6, 7), (18, 17, 4, 5),
]

# --- Perception Configuration ---
MAX_RENDER_DEPTH = 2.0
OBSTACLE_VOXEL_GRID_SIZE = 0.05
VOXEL_SIZE_NEAR = 0.02
VOXEL_SIZE_FAR = 0.015
FLOOR_COLOR = Color(20, 160, 20, 255)
TINT_RED_GPU = torch.tensor([255, 50, 50], device=device, dtype=torch.float32)
TINT_BLUE_GPU = torch.tensor([50, 50, 255], device=device, dtype=torch.float32)

# Danger zone for obstacle avoidance, in robot's local frame (X-right, Z-forward)
DANGER_ZONE_WIDTH = 0.4
DANGER_ZONE_MIN_HEIGHT_ABOVE_FLOOR = 0.02
DANGER_ZONE_MAX_HEIGHT_ABOVE_FLOOR = 0.8
AVOIDANCE_TURN_ANGLE_DEG = 35.0 # Angle to turn when avoiding
AVOIDANCE_CLEARANCE_STEPS = 5 # Number of walk steps to take before returning to path

# ==============================================================================
# --- 2. PERCEPTION AND VISUALIZATION FUNCTIONS ---
# ==============================================================================

def initialize_hardware():
    """Initializes RealSense camera and QuadPilotBody controller."""
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
        time.sleep(2.0) # Wait for ESPs
        if not body_controller.set_control_params(P=1.5, I=0.0, D=0.3, dead_zone=5, pos_thresh=5):
            raise Exception("Failed to set control parameters.")
        if not body_controller.set_all_pins(MOTOR_PINS):
            raise Exception("Failed to set motor pins.")
        time.sleep(0.5)
        if not body_controller.reset_all():
            raise Exception("Failed to reset all motors.")
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
    except RuntimeError:
        return None
    depth_frame = frames.get_depth_frame()
    if not depth_frame: return None
    
    depth_frame = spatial_filter.process(depth_frame)
    points = pc.calculate(depth_frame)
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    
    valid_depth_mask = (verts[:, 2] > 0.1) & (verts[:, 2] < MAX_RENDER_DEPTH)
    return verts[valid_depth_mask]

def cluster_voxel_blobs_cpu(danger_points_gpu):
    if danger_points_gpu.shape[0] < 20:
        return []

    voxel_coords = (danger_points_gpu / OBSTACLE_VOXEL_GRID_SIZE).floor().int()
    unique_coords, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
    unique_coords_cpu = unique_coords.cpu().numpy()
    
    min_coords = unique_coords_cpu.min(axis=0)
    grid_shape = unique_coords_cpu.max(axis=0) - min_coords + 3
    grid = np.zeros(grid_shape, dtype=bool)
    
    grid[
        unique_coords_cpu[:, 0] - min_coords[0] + 1,
        unique_coords_cpu[:, 1] - min_coords[1] + 1,
        unique_coords_cpu[:, 2] - min_coords[2] + 1
    ] = True
    
    labeled_grid, num_blobs = scipy.ndimage.label(grid, structure=np.ones((3,3,3)))
    if num_blobs == 0:
        return []

    obstacle_boxes = []
    labels = labeled_grid[
        unique_coords_cpu[:, 0] - min_coords[0] + 1,
        unique_coords_cpu[:, 1] - min_coords[1] + 1,
        unique_coords_cpu[:, 2] - min_coords[2] + 1
    ]
    labels_gpu = torch.from_numpy(labels).to(device)
    full_point_labels = labels_gpu[inverse_indices]

    for i in range(1, num_blobs + 1):
        blob_points = danger_points_gpu[full_point_labels == i]
        if blob_points.shape[0] > 20:
            min_b, _ = torch.min(blob_points, dim=0)
            max_b, _ = torch.max(blob_points, dim=0)
            obstacle_boxes.append((
                min_b.cpu().numpy(),
                max_b.cpu().numpy()
            ))
    return obstacle_boxes

def process_points_gpu(verts_gpu):
    # Floor Detection
    floor_candidates = verts_gpu[verts_gpu[:, 1] < 0]
    floor_y = torch.median(floor_candidates[:, 1]) if floor_candidates.shape[0] > 100 else 0.0
    is_floor_mask_gpu = verts_gpu[:, 1] < (floor_y + 0.02)
    obstacle_verts_gpu = verts_gpu[~is_floor_mask_gpu]

    # Danger Zone Points for Clustering
    dist_to_floor = obstacle_verts_gpu[:, 1] - floor_y
    danger_mask = (
        (torch.abs(obstacle_verts_gpu[:, 0]) < DANGER_ZONE_WIDTH / 2) &
        (dist_to_floor > DANGER_ZONE_MIN_HEIGHT_ABOVE_FLOOR) &
        (dist_to_floor < DANGER_ZONE_MAX_HEIGHT_ABOVE_FLOOR) &
        (obstacle_verts_gpu[:, 2] < MAX_RENDER_DEPTH) # Only cluster things in front
    )
    danger_points_gpu = obstacle_verts_gpu[danger_mask]
    
    obstacle_boxes = cluster_voxel_blobs_cpu(danger_points_gpu)
    
    # Voxelize for drawing (this is separate from clustering)
    def voxelize_for_drawing(points, voxel_size):
        if points.shape[0] == 0: return torch.empty((0,3), device=device)
        voxel_coords = (points / voxel_size).floor().int()
        unique_coords = torch.unique(voxel_coords, dim=0)
        return unique_coords.float() * voxel_size
    
    floor_verts_gpu = verts_gpu[is_floor_mask_gpu]
    obstacle_centers = voxelize_for_drawing(obstacle_verts_gpu, VOXEL_SIZE_FAR)
    floor_centers = voxelize_for_drawing(floor_verts_gpu, VOXEL_SIZE_FAR)
    
    return obstacle_boxes, obstacle_centers.cpu().numpy(), floor_centers.cpu().numpy()

def draw_scene(camera, state, fps, gpu_time):
    # Read shared state safely
    with state.lock:
        obstacles = list(state.detected_obstacles)
        robot_pos = state.estimated_position.copy()
        path_history = list(state.estimated_path_history)
        current_target_yaw = state.current_target_yaw
        original_target_yaw = state.original_target_yaw
        current_yaw = state.current_yaw
        robot_mode = state.robot_mode
    
    # --- 3D Drawing ---
    begin_mode_3d(camera)
    draw_grid(20, 0.5)

    # Draw the robot's estimated position and path history
    draw_cube(Vector3(robot_pos[0], 0, robot_pos[2]), 0.2, 0.1, 0.3, BLUE)
    if len(path_history) > 1:
        for i in range(len(path_history) - 1):
            p1 = path_history[i]
            p2 = path_history[i+1]
            draw_line_3d(Vector3(p1[0], 0.01, p1[2]), Vector3(p2[0], 0.01, p2[2]), BLUE)
    
    # Draw Orientation Lines (from robot center)
    # Convert yaws from degrees to radians for math functions
    yaw_orig_rad = math.radians(original_target_yaw)
    yaw_curr_rad = math.radians(current_target_yaw)
    line_len = 0.5
    
    # Original Target Yaw (Green Line)
    orig_end_pos = Vector3(
        robot_pos[0] + line_len * math.sin(yaw_orig_rad),
        0.05,
        robot_pos[2] + line_len * math.cos(yaw_orig_rad)
    )
    draw_line_3d(Vector3(robot_pos[0], 0.05, robot_pos[2]), orig_end_pos, GREEN)

    # Current Target Yaw (Red Line)
    curr_end_pos = Vector3(
        robot_pos[0] + line_len * math.sin(yaw_curr_rad),
        0.05,
        robot_pos[2] + line_len * math.cos(yaw_curr_rad)
    )
    draw_line_3d(Vector3(robot_pos[0], 0.05, robot_pos[2]), curr_end_pos, RED)
    
    # Draw the intended straight-line path
    path_len = 5.0
    path_width = 0.5
    path_center_x = robot_pos[0] + (path_len/2) * math.sin(yaw_orig_rad)
    path_center_z = robot_pos[2] + (path_len/2) * math.cos(yaw_orig_rad)
    # NOTE: PyRay's DrawCubeV is not friendly to rotation. We draw a BoundingBox instead.
    # This box won't rotate with yaw, it's just a visual guide of the corridor.
    # For a simple line, we can just draw_line_3d. Let's do that.
    path_end_pos = Vector3(
        robot_pos[0] + path_len * math.sin(yaw_orig_rad),
        0.01,
        robot_pos[2] + path_len * math.cos(yaw_orig_rad)
    )
    draw_line_3d(Vector3(robot_pos[0], 0.01, robot_pos[2]), path_end_pos, Color(0, 255, 0, 100))
    
    # Draw detected obstacles
    for min_b, max_b in obstacles:
        box = BoundingBox(Vector3(min_b[0], min_b[1], min_b[2]), Vector3(max_b[0], max_b[1], max_b[2]))
        draw_bounding_box(box, ORANGE)

    end_mode_3d()

    # --- 2D UI Drawing ---
    draw_fps(10, 10)
    draw_text(f"GPU Time: {gpu_time:.2f} ms", 10, 40, 20, BLACK)
    draw_text(f"Robot Mode: {robot_mode}", 10, 70, 20, BLUE)
    
    yaw_err = current_target_yaw - current_yaw
    draw_text(f"Original Yaw: {original_target_yaw:.1f}", 10, 100, 20, GREEN)
    draw_text(f"Target Yaw:   {current_target_yaw:.1f}", 10, 130, 20, RED)
    draw_text(f"Current Yaw:  {current_yaw:.1f}", 10, 160, 20, BLACK)
    draw_text(f"Yaw Error:    {yaw_err:.1f}", 10, 190, 20, BLACK)

    if robot_mode == "AVOIDING":
        draw_text("AVOIDING OBSTACLE", 300, 10, 30, RED)
    elif robot_mode == "WALKING":
        draw_text("WALKING STRAIGHT", 300, 10, 30, GREEN)
    else:
        draw_text("IDLE", 300, 10, 30, GRAY)


# ==============================================================================
# --- 3. ROBOT CONTROL LOGIC ---
# ==============================================================================

def get_stance_angles():
    stance = [0.0] * NUM_MOTORS
    stance[ACTUATOR_NAME_TO_INDEX_MAP["FL_tigh_actuator"]] = 45.0
    stance[ACTUATOR_NAME_TO_INDEX_MAP["FL_knee_actuator"]] = 45.0
    stance[ACTUATOR_NAME_TO_INDEX_MAP["FR_tigh_actuator"]] = -45.0
    stance[ACTUATOR_NAME_TO_INDEX_MAP["FR_knee_actuator"]] = 45.0
    stance[ACTUATOR_NAME_TO_INDEX_MAP["BR_tigh_actuator"]] = 45.0
    stance[ACTUATOR_NAME_TO_INDEX_MAP["BR_knee_actuator"]] = -45.0
    stance[ACTUATOR_NAME_TO_INDEX_MAP["BL_tigh_actuator"]] = 45.0
    stance[ACTUATOR_NAME_TO_INDEX_MAP["BL_knee_actuator"]] = -45.0
    return stance

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def execute_walk_step(body_controller, state):
    """Executes one full diagonal-pair walk cycle with yaw correction."""
    idx_fr_knee = ACTUATOR_NAME_TO_INDEX_MAP["FR_knee_actuator"]
    idx_bl_knee = ACTUATOR_NAME_TO_INDEX_MAP["BL_knee_actuator"]
    idx_fl_knee = ACTUATOR_NAME_TO_INDEX_MAP["FL_knee_actuator"]
    idx_br_knee = ACTUATOR_NAME_TO_INDEX_MAP["BR_knee_actuator"]
    stance_pose = get_stance_angles()
    
    # Read shared state safely
    with state.lock:
        current_yaw = state.current_yaw
        target_yaw = state.current_target_yaw

    # Yaw correction logic
    yaw_error = target_yaw - current_yaw
    # Normalize error to [-180, 180]
    while yaw_error > 180: yaw_error -= 360
    while yaw_error < -180: yaw_error += 360

    correction = CORRECTION_GAIN_KP * yaw_error
    
    # To steer LEFT (positive yaw error), we need to push more with the right legs.
    # To steer RIGHT (negative yaw error), we need to push more with the left legs.
    N = clamp(NEUTRAL_LIFT_ANGLE - correction, MIN_LIFT_ANGLE, MAX_LIFT_ANGLE) # Left side lift
    Y = clamp(NEUTRAL_LIFT_ANGLE + correction, MIN_LIFT_ANGLE, MAX_LIFT_ANGLE) # Right side lift
    
    # Step 1: Lift FR/BL
    step_pose = stance_pose[:]
    step_pose[idx_fr_knee] = Y 
    step_pose[idx_bl_knee] = -N
    body_controller.set_angles(step_pose)
    time.sleep(WALK_STEP_DURATION)

    # Step 2: Plant All
    body_controller.set_angles(stance_pose)
    time.sleep(WALK_STEP_DURATION)
    
    # Step 3: Lift FL/BR
    step_pose = stance_pose[:]
    step_pose[idx_fl_knee] = N
    step_pose[idx_br_knee] = -Y
    body_controller.set_angles(step_pose)
    time.sleep(WALK_STEP_DURATION)
    
    # Step 4: Plant All
    body_controller.set_angles(stance_pose)
    time.sleep(WALK_STEP_DURATION)

def robot_control_thread_func(body_controller, state):
    """The main loop for the robot's brain and state machine."""
    print("Robot control thread started.")
    avoidance_step_counter = 0
    last_update_time = time.perf_counter()

    while state.is_running:
        # --- Handle User Commands First ---
        command = None
        with state.lock:
            if state.user_command:
                command = state.user_command
                state.user_command = None # Consume command
        
        if command == "start_walk":
            dmp_data = body_controller.get_latest_dmp_data_for_esp(ESP_WITH_IMU)
            if dmp_data and dmp_data.get('ypr_deg'):
                current_heading = dmp_data['ypr_deg'].get('yaw', 0.0)
                with state.lock:
                    state.robot_mode = "WALKING"
                    state.original_target_yaw = current_heading
                    state.current_target_yaw = current_heading
                    state.estimated_position = np.array([0.0, 0.0, 0.0]) # Reset position on start
                    state.estimated_path_history = [state.estimated_position.copy()]
                print(f"Starting walk. Original Yaw captured: {current_heading:.1f} deg.")
            else:
                print("Warning: Cannot start walk, no IMU data available.")

        elif command == "stop_walk":
            with state.lock:
                state.robot_mode = "IDLE"
            print("Stopping walk. Returning to Stance.")
            body_controller.set_angles(get_stance_angles())
        
        elif command == "stance":
            with state.lock:
                state.robot_mode = "IDLE"
            body_controller.set_angles(get_stance_angles())
        
        elif command == "zero":
            with state.lock:
                state.robot_mode = "IDLE"
            body_controller.set_angles([0.0] * NUM_MOTORS)

        # --- Continuous State Logic ---
        dmp_data = body_controller.get_latest_dmp_data_for_esp(ESP_WITH_IMU)
        if dmp_data and dmp_data.get('ypr_deg'):
            with state.lock:
                state.current_yaw = dmp_data['ypr_deg'].get('yaw', state.current_yaw)

        # --- State Machine: WALKING / AVOIDING ---
        current_mode = ""
        with state.lock:
            current_mode = state.robot_mode

        if current_mode in ["WALKING", "AVOIDING"]:
            # Obstacle check
            obstacles_in_path = False
            with state.lock:
                # Check if any obstacle bounding box is in our direct path
                for min_b, max_b in state.detected_obstacles:
                    # Z is forward, X is side-to-side
                    obstacle_center_x = (min_b[0] + max_b[0]) / 2
                    if abs(obstacle_center_x) < DANGER_ZONE_WIDTH / 2 and max_b[2] < (MAX_RENDER_DEPTH-0.2):
                        obstacles_in_path = True
                        break

            if current_mode == "WALKING" and obstacles_in_path:
                print("OBSTACLE DETECTED! Switching to AVOIDANCE mode.")
                with state.lock:
                    state.robot_mode = "AVOIDING"
                    # Decide turn direction based on obstacle position
                    # Simple logic: if obstacle is right of center (positive X), turn left (positive angle).
                    # This assumes a right-handed coord system. Let's make it simpler:
                    # Just turn left by default.
                    state.current_target_yaw = state.original_target_yaw + AVOIDANCE_TURN_ANGLE_DEG
                avoidance_step_counter = 0

            elif current_mode == "AVOIDING":
                if not obstacles_in_path:
                    avoidance_step_counter += 1
                    if avoidance_step_counter >= AVOIDANCE_CLEARANCE_STEPS:
                        print("Path clear. Returning to original trajectory.")
                        with state.lock:
                            state.robot_mode = "WALKING"
                            state.current_target_yaw = state.original_target_yaw
                else:
                    # If we still see an obstacle, reset the clearance counter
                    avoidance_step_counter = 0

            # Execute a walk step based on current target yaw
            execute_walk_step(body_controller, state)

            # Update position (Dead Reckoning)
            dt = time.perf_counter() - last_update_time
            last_update_time = time.perf_counter()
            with state.lock:
                yaw_rad = math.radians(state.current_yaw)
                delta_z = WALK_SPEED_MPS * dt * math.cos(yaw_rad)
                delta_x = WALK_SPEED_MPS * dt * math.sin(yaw_rad)
                state.estimated_position[0] += delta_x
                state.estimated_position[2] += delta_z
                state.estimated_path_history.append(state.estimated_position.copy())
                if len(state.estimated_path_history) > 200:
                    state.estimated_path_history.pop(0)
        
        else: # IDLE
            time.sleep(0.1) # Don't spin CPU when idle
            last_update_time = time.perf_counter()


# ==============================================================================
# --- 4. MAIN EXECUTION AND THREAD MANAGEMENT ---
# ==============================================================================

def on_key_press(key, state):
    """Keyboard handler that sets flags in the shared state."""
    try:
        if hasattr(key, 'char'):
            char = key.char.lower()
            if char == 'y': state.user_command = "start_walk"
            elif char == 's': state.user_command = "stop_walk"
            elif char == 'a': state.user_command = "stance"
            elif char == 'd': state.user_command = "zero"
    except Exception as e:
        print(f"Error in on_key_press: {e}")

def main():
    # --- Initialization ---
    rs_components, body_controller = initialize_hardware()
    if not rs_components or not body_controller:
        print("Hardware initialization failed. Exiting.")
        return

    shared_state = SharedState()
    
    # Start the robot control thread
    control_thread = threading.Thread(
        target=robot_control_thread_func,
        args=(body_controller, shared_state),
        daemon=True
    )
    control_thread.start()

    # Start the keyboard listener
    # The listener needs a way to modify the shared state. We use a lambda.
    key_listener = keyboard.Listener(on_press=lambda key: on_key_press(key, shared_state))
    key_listener.start()
    
    # --- PyRay Window and Main Loop ---
    init_window(screenWidth, screenHeight, "Intelligent Quadruped Control")
    camera = Camera3D()
    camera.position = Vector3(0.0, 3.0, -3.0)
    camera.target = Vector3(0.0, 0.0, 0.0)
    camera.up = Vector3(0.0, 1.0, 0.0)
    camera.fovy = 60.0
    camera.projection = CAMERA_PERSPECTIVE
    set_target_fps(60)

    gpu_time = 0.0
    
    # Set initial stance
    body_controller.set_angles(get_stance_angles())

    print("\n" + "="*50)
    print("System Ready. Visualization and Control are LIVE.")
    print("  'y': Start autonomous walk")
    print("  's': Stop walk and hold stance")
    print("  'a': Go to Stance")
    print("  'd': Go to Zero position")
    print("  'ESC' in window to quit.")
    print("="*50 + "\n")


    try:
        while not window_should_close() and shared_state.is_running:
            # --- Perception Step ---
            valid_verts_np = get_point_cloud_numpy(rs_components)
            
            obstacles_cpu = []
            if valid_verts_np is not None and valid_verts_np.shape[0] > 0:
                start_time = time.perf_counter()
                verts_gpu = torch.from_numpy(valid_verts_np).to(device, non_blocking=True)
                
                # Get obstacle bounding boxes from perception
                boxes, _, _ = process_points_gpu(verts_gpu)
                
                # Update shared state with detected obstacles
                with shared_state.lock:
                    shared_state.detected_obstacles = boxes
                
                gpu_time = (time.perf_counter() - start_time) * 1000

            # --- Drawing Step ---
            update_camera(camera, CameraMode.CAMERA_FREE)
            begin_drawing()
            clear_background(RAYWHITE)
            
            draw_scene(camera, shared_state, get_fps(), gpu_time)
            
            end_drawing()

    finally:
        # --- Shutdown Sequence ---
        print("\nShutdown sequence initiated...")
        shared_state.is_running = False
        
        if key_listener.is_alive():
            key_listener.stop()
        
        print("Waiting for control thread to finish...")
        control_thread.join(timeout=2.0)

        if body_controller:
            print("Disabling motors and closing connection...")
            body_controller.set_all_control_status(False)
            time.sleep(0.1)
            body_controller.reset_all()
            body_controller.close()
        
        if rs_components:
            print("Stopping RealSense pipeline...")
            rs_components[0].stop()
            
        close_window()
        print("Shutdown complete.")

if __name__ == '__main__':
    main()