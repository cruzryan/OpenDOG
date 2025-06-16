# intelligent_quad_control_v10.1_calibrated.py
# FEATURE: Adds a one-time yaw calibration at startup to remove IMU bias.
# The robot now measures its initial yaw for 0.5s and subtracts this offset
# from all future measurements for greater accuracy.
# UPDATE: ENGAGEMENT_DISTANCE changed to 1.0m.

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
import scipy.ndimage

# --- Neural Network Policy ---
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("FATAL ERROR: PyTorch is not installed. Please run: pip install torch")
    sys.exit(1)

# --- Robot Control ---
from pynput import keyboard
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.join(os.path.dirname(script_dir), "quadpilot")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from body import QuadPilotBody
except ImportError:
    print(f"FATAL ERROR: Could not import QuadPilotBody.")
    sys.exit(1)

# ==============================================================================
# --- 1. CONFIGURATION AND INITIALIZATION ---
# ==============================================================================

# --- Threading and State Management ---
class SharedState:
    def __init__(self):
        self.lock = threading.Lock(); self.is_running = True; self.user_command = None
        self.robot_mode = "IDLE" # IDLE, SEARCHING, FOLLOWING
        self.current_yaw, self.original_target_yaw, self.current_target_yaw = 0.0, 0.0, 0.0
        self.estimated_position = np.array([0.0, 0.0, 0.0]); self.estimated_path_history, self.local_obstacles, self.nearest_obstacle_index = [], [], -1
        ### --- CHANGE: Added a variable to store the yaw offset --- ###
        self.yaw_offset = 0.0

# --- GPU and PyRay Setup ---
if torch.cuda.is_available(): device = torch.device("cuda"); print(f"Using GPU for perception: {torch.cuda.get_device_name(0)}")
else: device = torch.device("cpu"); print("Warning: CUDA not available. Perception will run on CPU.")
screenWidth, screenHeight = 1280, 720

# --- Robot Control Configuration ---
NUM_MOTORS, ESP_WITH_IMU = 8, 1
MODEL_PATH = 'walk_policy.pth'
MIN_LIFT_ANGLE, MAX_LIFT_ANGLE = 20.0, 45.0
WALK_STEP_DURATION, WALK_SPEED_MPS = 0.1, 0.15
POLICY_AGGRESSIVENESS = 0.6

# --- Follow Behavior Configuration ---
FOLLOW_DISTANCE = 0.6
FOLLOW_DISTANCE_TOLERANCE = 0.05
FOLLOW_STEERING_GAIN = 50.0

# --- Perception Configuration ---
MAX_RENDER_DEPTH, OBSTACLE_VOXEL_GRID_SIZE = 2.5, 0.05
DANGER_ZONE_WIDTH, DANGER_ZONE_MIN_HEIGHT_ABOVE_FLOOR, DANGER_ZONE_MAX_HEIGHT_ABOVE_FLOOR = 0.8, 0.08, 0.8
### --- CHANGE: Updated engagement distance as requested --- ###
ENGAGEMENT_DISTANCE = 1.1

ACTUATOR_NAME_TO_INDEX_MAP = {
    "FL_tigh_actuator": 3, "FL_knee_actuator": 0, "FR_tigh_actuator": 1, "FR_knee_actuator": 2,
    "BR_tigh_actuator": 5, "BR_knee_actuator": 4, "BL_tigh_actuator": 7, "BL_knee_actuator": 6,
}
MOTOR_PINS = [
    (39, 40, 41, 42), (16, 15, 7, 6), (17, 18, 5, 4), (37, 38, 1, 2),
    (37, 38, 1, 2), (40, 39, 42, 41), (15, 16, 6, 7), (18, 17, 4, 5),
]


# ==============================================================================
# --- 2. NEURAL NETWORK AND UTILITY FUNCTIONS ---
# ==============================================================================
class WalkPolicy(nn.Module):
    def __init__(self):
        super(WalkPolicy, self).__init__(); self.network = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x): return self.network(x)

def clamp(v, min_v, max_v): return max(min_v, min(v, max_v))
def get_stance_angles():
    s = [0.0] * 8; s[3],s[0],s[1],s[2],s[5],s[4],s[7],s[6] = 45,45,-45,45,45,-45,45,-45; return s
def get_policy_action(yaw_error: float, model: WalkPolicy) -> tuple[float, float]:
    input_tensor = torch.FloatTensor([[yaw_error]])
    with torch.no_grad(): predicted_N, predicted_Y = model(input_tensor).squeeze().tolist()
    return predicted_N, predicted_Y

# ==============================================================================
# --- 3. ROBOT CONTROL LOGIC ---
# ==============================================================================
def execute_walk_step(body_controller, state, policy_model):
    with state.lock: current_yaw, target_yaw = state.current_yaw, state.current_target_yaw
    yaw_error = target_yaw - current_yaw
    while yaw_error > 180: yaw_error -= 360
    while yaw_error < -180: yaw_error += 360
    policy_input_error = yaw_error * POLICY_AGGRESSIVENESS
    N, Y = get_policy_action(policy_input_error, policy_model)
    N, Y = clamp(N, MIN_LIFT_ANGLE, MAX_LIFT_ANGLE), clamp(Y, MIN_LIFT_ANGLE, MAX_LIFT_ANGLE)
    idx_fr_k, idx_bl_k, idx_fl_k, idx_br_k = 2, 6, 0, 4
    stance_pose = get_stance_angles()
    step1 = stance_pose[:]; step1[idx_fr_k] = N; step1[idx_bl_k] = -N
    body_controller.set_angles(step1); time.sleep(WALK_STEP_DURATION)
    body_controller.set_angles(stance_pose); time.sleep(WALK_STEP_DURATION)
    step2 = stance_pose[:]; step2[idx_fl_k] = Y; step2[idx_br_k] = -Y
    body_controller.set_angles(step2); time.sleep(WALK_STEP_DURATION)
    body_controller.set_angles(stance_pose); time.sleep(WALK_STEP_DURATION)

def find_nearest_obstacle(obs):
    if not obs: return -1, float('inf'), None
    dist, idx, box = float('inf'), -1, None
    for i, (b_min, b_max) in enumerate(obs):
        if (d := b_min[2]) < dist: dist, idx, box = d, i, (b_min, b_max)
    return idx, dist, box

def robot_control_thread_func(body_controller, state, policy_model):
    print("Robot control thread started.")
    last_update_time = time.perf_counter()
    should_walk_forward = False

    while state.is_running:
        cmd = None
        with state.lock:
            if state.user_command: cmd, state.user_command = state.user_command, None
        if cmd:
            if cmd == "start_walk":
                with state.lock:
                    # The current_yaw is now calibrated, so this will be ~0
                    h = state.current_yaw
                    state.robot_mode = "SEARCHING"
                    state.current_target_yaw = h
                    state.estimated_position, state.estimated_path_history = np.zeros(3), [np.zeros(3)]
                print("System activated. Entering SEARCHING mode (standing by).")
            elif cmd=="stop_walk":
                with state.lock: state.robot_mode="IDLE"; body_controller.set_angles(get_stance_angles())
            elif cmd in ["stance","zero"]:
                with state.lock: state.robot_mode="IDLE"; body_controller.set_angles(get_stance_angles() if cmd=="stance" else [0.0]*8)

        ### --- CHANGE: Apply yaw offset to all IMU readings --- ###
        dmp = body_controller.get_latest_dmp_data_for_esp(ESP_WITH_IMU)
        if dmp and dmp.get('ypr_deg'):
            with state.lock:
                raw_yaw = dmp['ypr_deg'].get('yaw', state.current_yaw)
                yaw_offset = state.yaw_offset
                
                # Apply correction and normalize
                corrected_yaw = raw_yaw - yaw_offset
                while corrected_yaw > 180: corrected_yaw -= 360
                while corrected_yaw < -180: corrected_yaw += 360
                
                # Store the clean, calibrated value
                state.current_yaw = corrected_yaw

        current_mode = ""
        with state.lock: current_mode = state.robot_mode
        
        if current_mode in ["SEARCHING", "FOLLOWING"]:
            with state.lock: obs, c_yaw = list(state.local_obstacles), state.current_yaw
            n_idx, n_dist, n_box = find_nearest_obstacle(obs)
            is_target_in_range = n_dist < ENGAGEMENT_DISTANCE
            
            with state.lock:
                state.nearest_obstacle_index = n_idx
                if is_target_in_range:
                    if state.robot_mode == "SEARCHING": print(f"TARGET ACQUIRED at {n_dist:.2f}m (< {ENGAGEMENT_DISTANCE}m). Engaging FOLLOW.")
                    state.robot_mode = "FOLLOWING"
                    center_x = (n_box[0][0] + n_box[1][0]) / 2.0
                    steering_adjustment_deg = center_x * FOLLOW_STEERING_GAIN
                    state.current_target_yaw = c_yaw + steering_adjustment_deg
                    should_walk_forward = (n_dist > (FOLLOW_DISTANCE + FOLLOW_DISTANCE_TOLERANCE))
                else:
                    if state.robot_mode == "FOLLOWING": print(f"Target lost (moved > {ENGAGEMENT_DISTANCE}m away). Returning to SEARCHING.")
                    state.robot_mode = "SEARCHING"
                    should_walk_forward = False
                    state.current_target_yaw = c_yaw
                
                while state.current_target_yaw > 180: state.current_target_yaw -= 360
                while state.current_target_yaw < -180: state.current_target_yaw += 360

            if should_walk_forward:
                execute_walk_step(body_controller, state, policy_model)
            else:
                time.sleep(0.1)

            if should_walk_forward:
                dt = time.perf_counter() - last_update_time;
                with state.lock:
                    rad = math.radians(state.current_yaw)
                    state.estimated_position += np.array([WALK_SPEED_MPS*dt*math.sin(rad), 0, WALK_SPEED_MPS*dt*math.cos(rad)])
                    state.estimated_path_history.append(state.estimated_position.copy())
                    if len(state.estimated_path_history) > 300: state.estimated_path_history.pop(0)
            last_update_time = time.perf_counter()
            
        else: # IDLE mode
            time.sleep(0.1); last_update_time = time.perf_counter()


# ==============================================================================
# --- 4. MAIN EXECUTION AND FULL UTILITY FUNCTIONS ---
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

# (minified utility functions from previous versions for brevity)
def initialize_hardware():
    print("Initializing RealSense..."); pipeline=rs.pipeline(); config=rs.config(); config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
    try: pipeline.start(config)
    except RuntimeError as e: print(f"Failed RealSense: {e}"); return None, None
    pc=rs.pointcloud(); sf=rs.spatial_filter(); sf.set_option(rs.option.filter_magnitude,2); sf.set_option(rs.option.filter_smooth_alpha,0.5); sf.set_option(rs.option.filter_smooth_delta,20)
    print("Initializing QuadPilotBody..."); body_controller=None
    try:
        body_controller=QuadPilotBody(listen_for_broadcasts=True); time.sleep(2); body_controller.set_control_params(P=1.5,I=0.0,D=0.3,dead_zone=5,pos_thresh=5); body_controller.set_all_pins(MOTOR_PINS); time.sleep(0.5); body_controller.reset_all(); time.sleep(0.2); body_controller.set_all_control_status(True)
    except Exception as e:
        print(f"FATAL: QuadPilotBody: {e}");
        if pipeline: pipeline.stop()
        if body_controller: body_controller.close()
        return None,None
    return (pipeline,pc,sf), body_controller
def get_point_cloud_numpy(rs_components):
    pipeline,pc,sf=rs_components
    try: frames=pipeline.wait_for_frames(1000)
    except RuntimeError: return None
    depth_frame=sf.process(frames.get_depth_frame())
    if not depth_frame: return None
    points=pc.calculate(depth_frame); verts=np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1,3)
    return verts[(verts[:,2]>0.1)&(verts[:,2]<MAX_RENDER_DEPTH)]
def cluster_voxel_blobs_cpu(danger_points_gpu):
    if danger_points_gpu.shape[0]<20: return []
    voxel_coords=(danger_points_gpu/OBSTACLE_VOXEL_GRID_SIZE).floor().int(); unique_coords,inverse_indices=torch.unique(voxel_coords,dim=0,return_inverse=True)
    unique_coords_cpu=unique_coords.cpu().numpy(); min_c,max_c=unique_coords_cpu.min(axis=0),unique_coords_cpu.max(axis=0)
    grid=np.zeros(max_c-min_c+3,dtype=bool); grid[tuple((unique_coords_cpu-min_c+1).T)]=True
    labeled_grid,num_blobs=scipy.ndimage.label(grid,structure=np.ones((3,3,3)))
    if num_blobs==0: return []
    boxes=[]; labels=labeled_grid[tuple((unique_coords_cpu-min_c+1).T)]; labels_gpu=torch.from_numpy(labels).to(device)
    full_labels=labels_gpu[inverse_indices]
    for i in range(1,num_blobs+1):
        blob_points=danger_points_gpu[full_labels==i]
        if blob_points.shape[0]>20: min_b,_=torch.min(blob_points,dim=0); max_b,_=torch.max(blob_points,dim=0); boxes.append((min_b.cpu().numpy(),max_b.cpu().numpy()))
    return boxes
def process_points_gpu(verts_gpu):
    floor_cands=verts_gpu[verts_gpu[:,1]<0]; floor_y=torch.median(floor_cands[:,1]) if floor_cands.shape[0]>100 else 0.0
    obs_verts=verts_gpu[verts_gpu[:,1]>(floor_y+0.02)]; dist_floor=obs_verts[:,1]-floor_y
    mask=((torch.abs(obs_verts[:,0])<DANGER_ZONE_WIDTH/2)&(dist_floor>DANGER_ZONE_MIN_HEIGHT_ABOVE_FLOOR)&(dist_floor<DANGER_ZONE_MAX_HEIGHT_ABOVE_FLOOR)&(obs_verts[:,2]<MAX_RENDER_DEPTH))
    return cluster_voxel_blobs_cpu(obs_verts[mask])
def transform_obstacles_to_world_frame(local_obs,pos,yaw):
    world_obs=[]; rad=math.radians(yaw); c,s=math.cos(rad),math.sin(rad)
    def tf(p): return np.array([(p[0]*c+p[2]*s)+pos[0],p[1],(-p[0]*s+p[2]*c)+pos[2]])
    for min_b,max_b in local_obs:
        corners=[np.array([x,y,z]) for x in [min_b[0],max_b[0]] for y in [min_b[1],max_b[1]] for z in [min_b[2],max_b[2]]]
        w_corners=[tf(c) for c in corners]; world_obs.append((np.min(w_corners,axis=0),np.max(w_corners,axis=0)))
    return world_obs
def draw_scene(cam,state,fps,gpu_t):
    with state.lock: data = (list(state.local_obstacles),state.estimated_position.copy(),list(state.estimated_path_history),state.current_target_yaw,state.original_target_yaw,state.current_yaw,state.robot_mode,state.nearest_obstacle_index)
    local_obs,pos,history,t_yaw,o_yaw,c_yaw,mode,n_idx=data
    world_obs=transform_obstacles_to_world_frame(local_obs,pos,c_yaw)
    begin_mode_3d(cam); draw_grid(20,0.5); draw_cube(Vector3(pos[0],0.05,pos[2]),0.2,0.1,0.3,BLUE)
    if mode in ["SEARCHING", "FOLLOWING"]:
        circle_center = Vector3(pos[0], 0.01, pos[2])
        draw_cylinder_wires(circle_center, ENGAGEMENT_DISTANCE, ENGAGEMENT_DISTANCE, 0.01, 32, GOLD)
        draw_cylinder_wires(circle_center, FOLLOW_DISTANCE, FOLLOW_DISTANCE, 0.01, 32, DARKGREEN)
    if len(history)>1:
        for i in range(len(history)-1): draw_line_3d(Vector3(history[i][0],0.01,history[i][2]),Vector3(history[i+1][0],0.01,history[i+1][2]),MAGENTA)
    c_rad=math.radians(t_yaw); draw_line_3d(Vector3(pos[0],0.05,pos[2]),Vector3(pos[0]+0.7*math.sin(c_rad),0.05,pos[2]+0.7*math.cos(c_rad)),RED)
    for i,(min_b,max_b) in enumerate(world_obs): draw_bounding_box(BoundingBox(Vector3(min_b[0],min_b[1],min_b[2]),Vector3(max_b[0],max_b[1],max_b[2])),RED if i==n_idx else LIME)
    end_mode_3d()
    draw_fps(10, 10); draw_text(f"GPU Time: {gpu_t:.1f} ms", 10, 40, 20, BLACK); draw_text(f"Robot Mode: {mode}", 10, 70, 20, BLUE)
    draw_text(f"Target Yaw:   {t_yaw:.1f}", 10, 130, 20, RED); draw_text(f"Current Yaw:  {c_yaw:.1f}", 10, 160, 20, BLACK)
    draw_text("Yellow: Awareness Zone (1.0m)", screenWidth - 350, 10, 20, GOLD)
    draw_text("Green: Goal Zone (60cm)", screenWidth - 350, 40, 20, DARKGREEN)

def main():
    try:
        print(f"Loading walk policy from '{MODEL_PATH}'...");
        if not os.path.exists(MODEL_PATH): raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        walk_policy_model = WalkPolicy(); walk_policy_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')); walk_policy_model.eval()
        print("Walk policy model loaded.")
    except Exception as e: print(f"FATAL: Could not load model: {e}"); return
    
    rs_components, body_controller = initialize_hardware()
    if not rs_components or not body_controller: return
    
    shared_state = SharedState()
    
    # Go to stance before calibration
    body_controller.set_angles(get_stance_angles())
    time.sleep(1.0) # Give robot time to settle into stance

    ### --- CHANGE: New Yaw Calibration Sequence --- ###
    print("\n--- Calibrating IMU Yaw Offset ---")
    print("Please keep the robot still...")
    yaw_readings = []
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < 0.5:
        dmp = body_controller.get_latest_dmp_data_for_esp(ESP_WITH_IMU)
        if dmp and 'ypr_deg' in dmp and 'yaw' in dmp['ypr_deg']:
            yaw_readings.append(dmp['ypr_deg']['yaw'])
        time.sleep(0.02)

    if not yaw_readings:
        print("WARNING: Could not get yaw readings for calibration. Offset will be 0.")
    else:
        yaw_offset = sum(yaw_readings) / len(yaw_readings)
        with shared_state.lock:
            shared_state.yaw_offset = yaw_offset
        print(f"Calibration complete. Calculated Yaw Offset: {yaw_offset:.2f} degrees.")
    
    control_thread = threading.Thread(target=robot_control_thread_func, args=(body_controller, shared_state, walk_policy_model), daemon=True)
    control_thread.start()
    
    key_listener = keyboard.Listener(on_press=lambda key: on_key_press(key, shared_state)); key_listener.start()
    init_window(screenWidth, screenHeight, "Intelligent Quadruped Control v10.1 (Calibrated)"); camera = Camera3D(Vector3(0.0, 4.0, -4.0), Vector3(0.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0), 60.0, CAMERA_PERSPECTIVE); set_target_fps(60); gpu_time = 0.0
    print("\n" + "="*50 + "\nSystem Ready. Calibrated Standby Follower is LIVE.\nPress 'y' to begin searching for a target.\n" + "="*50)
    try:
        while not window_should_close() and shared_state.is_running:
            valid_verts_np = get_point_cloud_numpy(rs_components)
            if valid_verts_np is not None and valid_verts_np.shape[0] > 0:
                start_time = time.perf_counter(); verts_gpu = torch.from_numpy(valid_verts_np).to(device, non_blocking=True); local_boxes = process_points_gpu(verts_gpu)
                with shared_state.lock: shared_state.local_obstacles = local_boxes
                gpu_time = (time.perf_counter() - start_time) * 1000
            update_camera(camera, CameraMode.CAMERA_FREE); begin_drawing(); clear_background(RAYWHITE); draw_scene(camera, shared_state, get_fps(), gpu_time); end_drawing()
    finally:
        print("\nShutdown sequence initiated..."); shared_state.is_running = False;
        if key_listener.is_alive(): key_listener.stop()
        print("Waiting for control thread..."); control_thread.join(timeout=2.0)
        if body_controller: print("Disabling motors..."); body_controller.set_all_control_status(False); time.sleep(0.1); body_controller.reset_all(); body_controller.close()
        if rs_components: print("Stopping RealSense..."); rs_components[0].stop()
        close_window(); print("Shutdown complete.")

if __name__ == '__main__':
    main()