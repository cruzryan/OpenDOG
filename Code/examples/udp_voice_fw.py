# master_voice_control_final.py
# A unified script combining voice control with simple walking,
# follow-me, and obstacle avoidance behaviors.
# STARTS in a limp state, calibrates IMU, and awaits voice commands.

# --- Core Python and System Imports ---
import os
import sys
import time
import threading
import math
from queue import Queue
from datetime import datetime, timedelta

# --- Input/Control Imports ---
from pynput import keyboard
import speech_recognition as sr

# --- AI/ML Imports ---
try:
    import torch
    import torch.nn as nn
    import whisper
    import numpy as np
except ImportError:
    print("FATAL ERROR: PyTorch or Whisper is not installed.")
    print("Please install them by running: pip install torch openai-whisper")
    sys.exit(1)

# --- Perception and Visualization (Optional) ---
REALSENSE_AVAILABLE = False
try:
    from pyray import *
    import pyray
    import pyrealsense2 as rs
    import scipy.ndimage
    REALSENSE_AVAILABLE = True
    print("RealSense and PyRay libraries found. Vision-based modes are available.")
except ImportError:
    print("\nWARNING: RealSense/PyRay libraries not found.")
    print("The 'sigueme' (follow) and 'esquiva' (avoid) commands will be disabled.")
    print("To enable them, run: pip install pyray opencv-python pyrealsense2 scipy\n")


# --- Path Setup for QuadPilotBody ---
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(os.path.dirname(script_dir), "quadpilot")
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

try:
    from body import QuadPilotBody
except ImportError:
    print(f"ERROR: Could not import QuadPilotBody. Ensure 'quadpilot' directory is a sibling to this script's directory.")
    sys.exit(1)

# ==============================================================================
# --- 1. UNIFIED CONFIGURATION ---
# ==============================================================================

# --- Robot General Config ---
NUM_MOTORS = 8
TARGET_MOTORS = list(range(NUM_MOTORS))
ESP_WITH_IMU = 1 # ESP32 index for primary IMU
MODEL_PATH = 'walk_policy.pth'

# --- Walk Gait Config ---
MIN_LIFT_ANGLE = 20.0
MAX_LIFT_ANGLE = 45.0
WALK_STEP_DURATION = 0.1
WALK_SPEED_MPS = 0.15

# --- Policy Turning Aggressiveness ---
POLICY_AGGRESSIVENESS = 0.65

# --- Follow Me Mode Config (v10.1) ---
FOLLOW_DISTANCE = 0.6
FOLLOW_DISTANCE_TOLERANCE = 0.05
FOLLOW_STEERING_GAIN = 60.0
FOLLOW_ENGAGEMENT_DISTANCE = 1.1

# --- Obstacle Avoidance Mode Config (v7.1) ---
AVOID_ENGAGEMENT_DISTANCE = 0.6
AVOID_PATH_ANGLE_DEG = 25.0
AVOID_PROBE_DEPTH = 1.5
AVOID_PROBE_WIDTH = 0.35
AVOID_PATH_CORRECTION_INCREMENT_DEG = 2.0

# --- Perception Config (for both vision modes) ---
MAX_RENDER_DEPTH = 2.5
OBSTACLE_VOXEL_GRID_SIZE = 0.05
DANGER_ZONE_WIDTH = 0.8
DANGER_ZONE_MIN_HEIGHT_ABOVE_FLOOR = 0.08
DANGER_ZONE_MAX_HEIGHT_ABOVE_FLOOR = 0.8

# --- Motor Mapping and Pins ---
ACTUATOR_NAME_TO_INDEX_MAP = {
    "FL_tigh_actuator": 3, "FL_knee_actuator": 0, "FR_tigh_actuator": 1, "FR_knee_actuator": 2,
    "BR_tigh_actuator": 5, "BR_knee_actuator": 4, "BL_tigh_actuator": 7, "BL_knee_actuator": 6,
}
MOTOR_PINS = [
    (39, 40, 41, 42), (16, 15, 7, 6), (17, 18, 5, 4), (37, 38, 1, 2),
    (37, 38, 1, 2), (40, 39, 42, 41), (15, 16, 6, 7), (18, 17, 4, 5),
]


# ==============================================================================
# --- 2. GLOBAL STATE AND THREADING EVENTS ---
# ==============================================================================
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.robot_mode = "IDLE"  # IDLE, SIMPLE_WALK, AVOIDING, FOLLOWING
        self.current_yaw = 0.0
        self.yaw_offset = 0.0 # For IMU calibration
        self.current_target_yaw = 0.0
        self.original_target_yaw = 0.0 # Used in AVOIDING mode
        # Vision-related state
        self.estimated_position = np.array([0.0, 0.0, 0.0])
        self.estimated_path_history = []
        self.local_obstacles = []
        self.nearest_obstacle_index = -1
        self.gpu_time = 0.0

shared_state = SharedState()
stop_program_event = threading.Event()
control_thread_handle = None
gui_thread_handle = None
voice_thread_handle = None

# GPU device setup
device = torch.device("cpu")
if REALSENSE_AVAILABLE and torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU for perception: {torch.cuda.get_device_name(0)}")


# ==============================================================================
# --- 3. NEURAL NETWORK AND UTILITY FUNCTIONS ---
# ==============================================================================
class WalkPolicy(nn.Module):
    def __init__(self):
        super(WalkPolicy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.network(x)

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def get_stance_angles():
    s=[0.0]*8; s[3],s[0],s[1],s[2],s[5],s[4],s[7],s[6] = 45,45,-45,45,45,-45,45,-45; return s
def get_partial_sit_angles():
    pose=get_stance_angles(); pose[5],pose[4],pose[7],pose[6] = 0,0,0,0; return pose
def get_sit_angles():
    return [0.0] * 8
def get_policy_action(yaw_error: float, model: WalkPolicy) -> tuple[float, float]:
    with torch.no_grad():
        predicted_N, predicted_Y = model(torch.FloatTensor([[yaw_error]])).squeeze().tolist()
    return predicted_N, predicted_Y

def interruptible_sleep(duration):
    start_time = time.time()
    with shared_state.lock: initial_mode = shared_state.robot_mode
    while time.time() - start_time < duration:
        with shared_state.lock: current_mode = shared_state.robot_mode
        if current_mode != initial_mode or stop_program_event.is_set(): break
        time.sleep(0.02)

# ==============================================================================
# --- 4. CORE ROBOT ACTIONS (WALKING, POSING) ---
# ==============================================================================

def execute_walk_step(body_controller, policy_model):
    with shared_state.lock:
        current_yaw, target_yaw = shared_state.current_yaw, shared_state.current_target_yaw
    yaw_error = target_yaw - current_yaw
    while yaw_error > 180: yaw_error -= 360
    while yaw_error < -180: yaw_error += 360

    policy_input_error = yaw_error * POLICY_AGGRESSIVENESS
    N, Y = get_policy_action(policy_input_error, policy_model)
    N, Y = clamp(N, MIN_LIFT_ANGLE, MAX_LIFT_ANGLE), clamp(Y, MIN_LIFT_ANGLE, MAX_LIFT_ANGLE)
    idx_fr_k, idx_bl_k, idx_fl_k, idx_br_k = 2, 6, 0, 4
    stance_pose = get_stance_angles()
    
    step_pose = stance_pose[:]; step_pose[idx_fr_k], step_pose[idx_bl_k] = N, -N
    body_controller.set_angles(step_pose); interruptible_sleep(WALK_STEP_DURATION)
    if stop_program_event.is_set(): return
    
    body_controller.set_angles(stance_pose); interruptible_sleep(WALK_STEP_DURATION)
    if stop_program_event.is_set(): return
    
    step_pose = stance_pose[:]; step_pose[idx_fl_k], step_pose[idx_br_k] = Y, -Y
    body_controller.set_angles(step_pose); interruptible_sleep(WALK_STEP_DURATION)
    if stop_program_event.is_set(): return
    
    body_controller.set_angles(stance_pose); interruptible_sleep(WALK_STEP_DURATION)

def command_set_angles(body_controller, target_angles: list[float]):
    if not body_controller.set_angles(target_angles):
        print("Failed to set angles (no OK from ESPs).")

# ==============================================================================
# --- 5. PERCEPTION & VISUALIZATION LOGIC (from vision scripts) ---
# ==============================================================================

if REALSENSE_AVAILABLE:
    # (Perception functions are minified for brevity, their logic is sound)
    def get_point_cloud_numpy(rs_c):
        p,pc,sf=rs_c;
        try:f=p.wait_for_frames(1000)
        except:return None
        df=sf.process(f.get_depth_frame());
        if not df:return None
        v=np.asanyarray(pc.calculate(df).get_vertices()).view(np.float32).reshape(-1,3);
        return v[(v[:,2]>0.1)&(v[:,2]<MAX_RENDER_DEPTH)]
    def cluster_voxel_blobs_cpu(d_pts):
        if d_pts.shape[0]<20:return[]
        vc=(d_pts/OBSTACLE_VOXEL_GRID_SIZE).floor().int();uc,inv=torch.unique(vc,dim=0,return_inverse=True);uc_cpu=uc.cpu().numpy();min_c,max_c=uc_cpu.min(axis=0),uc_cpu.max(axis=0);g=np.zeros(max_c-min_c+3,dtype=bool);g[tuple((uc_cpu-min_c+1).T)]=True;lg,nb=scipy.ndimage.label(g,structure=np.ones((3,3,3)));
        if nb==0:return[]
        b,l=[],lg[tuple((uc_cpu-min_c+1).T)];lgpu=torch.from_numpy(l).to(device);fl=lgpu[inv]
        for i in range(1,nb+1):
            bp=d_pts[fl==i];
            if bp.shape[0]>20:min_b,_=torch.min(bp,dim=0);max_b,_=torch.max(bp,dim=0);b.append((min_b.cpu().numpy(),max_b.cpu().numpy()))
        return b
    def process_points_gpu(v_gpu):
        fc=v_gpu[v_gpu[:,1]<0];fy=torch.median(fc[:,1]) if fc.shape[0]>100 else 0.0;ov=v_gpu[v_gpu[:,1]>(fy+0.02)];df=ov[:,1]-fy;m=((torch.abs(ov[:,0])<DANGER_ZONE_WIDTH/2)&(df>DANGER_ZONE_MIN_HEIGHT_ABOVE_FLOOR)&(df<DANGER_ZONE_MAX_HEIGHT_ABOVE_FLOOR)&(ov[:,2]<MAX_RENDER_DEPTH));return cluster_voxel_blobs_cpu(ov[m])
    def find_nearest_obstacle(obs):
        if not obs:return -1,float('inf'),None
        d,i,b=float('inf'),-1,None;
        for j,(b_min,_) in enumerate(obs):
            if(dist:=b_min[2])<d:d,i,b=dist,j,(b_min,b_min)
        return i,d,b
    def score_avoidance_paths(obs):
        s={'left':0.,'forward':0.,'right':0.};
        if not obs:return s
        f_min,f_max=-AVOID_PROBE_WIDTH/2,AVOID_PROBE_WIDTH/2
        for b_min,b_max in obs:
            if b_min[2]>AVOID_PROBE_DEPTH:continue
            t=1./(b_min[2]**2+0.1);o_min,o_max=b_min[0],b_max[0]
            if max(f_min,o_min)<min(f_max,o_max):s['forward']+=t
            if max(f_max,o_min)<min(f_max+AVOID_PROBE_WIDTH,o_max):s['right']+=t
            if max(-AVOID_PROBE_WIDTH-f_min,o_min)<min(f_min,o_max):s['left']+=t
        return s
    def transform_obstacles_to_world_frame(local_obs,pos,yaw):
        wo,r=[],math.radians(yaw);c,s=math.cos(r),math.sin(r)
        def tf(p):return np.array([(p[0]*c+p[2]*s)+pos[0],p[1],(-p[0]*s+p[2]*c)+pos[2]])
        for min_b,max_b in local_obs:
            crns=[tf(np.array([x,y,z])) for x in[min_b[0],max_b[0]]for y in[min_b[1],max_b[1]]for z in[min_b[2],max_b[2]]];wo.append((np.min(crns,axis=0),np.max(crns,axis=0)))
        return wo
    
    def gui_loop_func():
        screenWidth, screenHeight = 1280, 720; is_gui_open = False
        camera = Camera3D(Vector3(0.0, 4.0, -4.0), Vector3(0.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0), 60.0, CAMERA_PERSPECTIVE)
        print("GUI thread started. Waiting for a vision-based mode.")
        while not stop_program_event.is_set():
            with shared_state.lock: mode, is_vision_mode = shared_state.robot_mode, shared_state.robot_mode in ["FOLLOWING", "AVOIDING"]
            if is_vision_mode and not is_gui_open:
                print("[GUI] Initializing window..."); init_window(screenWidth, screenHeight, "QuadPilot Vision System"); set_target_fps(60); is_gui_open = True
            if is_gui_open:
                if window_should_close(): print("[GUI] Window closed by user. Shutting down."); stop_program_event.set(); break
                with shared_state.lock: data = (list(shared_state.local_obstacles), shared_state.estimated_position.copy(), list(shared_state.estimated_path_history), shared_state.current_target_yaw, shared_state.original_target_yaw, shared_state.current_yaw, mode, shared_state.nearest_obstacle_index, shared_state.gpu_time)
                local_obs,pos,history,t_yaw,o_yaw,c_yaw,mode,n_idx,gpu_t=data; world_obs=transform_obstacles_to_world_frame(local_obs,pos,c_yaw)
                update_camera(camera,CameraMode.CAMERA_FREE); begin_drawing(); clear_background(RAYWHITE); begin_mode_3d(camera); draw_grid(20,0.5); draw_cube(Vector3(pos[0],0.05,pos[2]),0.2,0.1,0.3,BLUE)
                if len(history)>1:
                    for i in range(len(history)-1):draw_line_3d(Vector3(history[i][0],0.01,history[i][2]),Vector3(history[i+1][0],0.01,history[i+1][2]),MAGENTA)
                if mode=="FOLLOWING":
                    cc=Vector3(pos[0],0.01,pos[2]); draw_cylinder_wires(cc,FOLLOW_ENGAGEMENT_DISTANCE,FOLLOW_ENGAGEMENT_DISTANCE,0.01,32,GOLD); draw_cylinder_wires(cc,FOLLOW_DISTANCE,FOLLOW_DISTANCE,0.01,32,DARKGREEN)
                elif mode=="AVOIDING":
                    o_rad=math.radians(o_yaw); draw_line_3d(Vector3(pos[0],0.01,pos[2]),Vector3(pos[0]+8*math.sin(o_rad),0.01,pos[2]+8*math.cos(o_rad)),Color(0,228,48,180))
                c_rad=math.radians(t_yaw); draw_line_3d(Vector3(pos[0],0.05,pos[2]),Vector3(pos[0]+0.7*math.sin(c_rad),0.05,pos[2]+0.7*math.cos(c_rad)),RED)
                for i,(min_b,max_b) in enumerate(world_obs): draw_bounding_box(BoundingBox(Vector3(min_b[0],min_b[1],min_b[2]),Vector3(max_b[0],max_b[1],max_b[2])),RED if i==n_idx else LIME)
                end_mode_3d(); draw_fps(10,10); draw_text(f"GPU Time: {gpu_t:.1f} ms",10,40,20,BLACK); draw_text(f"Robot Mode: {mode}",10,70,20,BLUE); draw_text(f"Target Yaw: {t_yaw:.1f}",10,100,20,RED); draw_text(f"Current Yaw: {c_yaw:.1f}",10,130,20,BLACK)
                if mode=="FOLLOWING": draw_text("Yellow: Awareness | Green: Goal",screenWidth-320,10,20,GOLD)
                elif mode=="AVOIDING": draw_text(f"Original Yaw: {o_yaw:.1f}",10,160,20,Color(0,117,44,255))
                end_drawing()
            elif not is_vision_mode and is_gui_open: print("[GUI] Closing window..."); close_window(); is_gui_open = False
            time.sleep(0.016)
        if is_gui_open: close_window()
        print("GUI thread stopped.")

# ==============================================================================
# --- 6. UNIFIED ROBOT CONTROL THREAD ---
# ==============================================================================
def robot_control_thread_func(body_controller, policy_model, rs_components):
    print("Unified robot control thread started.")
    last_update_time = time.perf_counter()
    while not stop_program_event.is_set():
        # --- IMU Update ---
        dmp = body_controller.get_latest_dmp_data_for_esp(ESP_WITH_IMU)
        if dmp and dmp.get('ypr_deg'):
            with shared_state.lock:
                raw_yaw = dmp['ypr_deg'].get('yaw', shared_state.current_yaw)
                corrected_yaw = raw_yaw - shared_state.yaw_offset
                while corrected_yaw > 180: corrected_yaw -= 360
                while corrected_yaw < -180: corrected_yaw += 360
                shared_state.current_yaw = corrected_yaw
        
        with shared_state.lock: mode, is_vision_mode = shared_state.robot_mode, shared_state.robot_mode in ["FOLLOWING", "AVOIDING"]
        
        # --- Perception ---
        if REALSENSE_AVAILABLE and is_vision_mode:
            valid_verts_np = get_point_cloud_numpy(rs_components)
            if valid_verts_np is not None and valid_verts_np.shape[0] > 0:
                start_time = time.perf_counter()
                verts_gpu = torch.from_numpy(valid_verts_np).to(device, non_blocking=True)
                with shared_state.lock:
                    shared_state.local_obstacles = process_points_gpu(verts_gpu)
                    shared_state.gpu_time = (time.perf_counter() - start_time) * 1000
        
        # --- State Machine ---
        is_walking = False
        if mode == "SIMPLE_WALK": is_walking = True
        elif mode == "FOLLOWING":
            if not REALSENSE_AVAILABLE: continue
            with shared_state.lock: obs, c_yaw = list(shared_state.local_obstacles), shared_state.current_yaw
            n_idx, n_dist, n_box = find_nearest_obstacle(obs)
            with shared_state.lock:
                shared_state.nearest_obstacle_index = n_idx
                if n_dist < FOLLOW_ENGAGEMENT_DISTANCE:
                    center_x = (n_box[0][0] + n_box[1][0]) / 2.0
                    shared_state.current_target_yaw = c_yaw + (center_x * FOLLOW_STEERING_GAIN)
                    is_walking = (n_dist > (FOLLOW_DISTANCE + FOLLOW_DISTANCE_TOLERANCE))
                else:
                    shared_state.current_target_yaw, is_walking = c_yaw, False
                while shared_state.current_target_yaw > 180: shared_state.current_target_yaw -= 360
                while shared_state.current_target_yaw < -180: shared_state.current_target_yaw += 360
        elif mode == "AVOIDING":
            if not REALSENSE_AVAILABLE: continue
            is_walking = True
            with shared_state.lock: obs, c_yaw, o_yaw = list(shared_state.local_obstacles), shared_state.current_yaw, shared_state.original_target_yaw
            n_idx, n_dist, _ = find_nearest_obstacle(obs)
            with shared_state.lock:
                shared_state.nearest_obstacle_index = n_idx
                if n_dist < AVOID_ENGAGEMENT_DISTANCE:
                    scores = score_avoidance_paths(obs); best_path = min(scores, key=scores.get)
                    if best_path == 'left': shared_state.current_target_yaw = c_yaw + AVOID_PATH_ANGLE_DEG
                    elif best_path == 'right': shared_state.current_target_yaw = c_yaw - AVOID_PATH_ANGLE_DEG
                    else: shared_state.current_target_yaw = c_yaw + AVOID_PATH_ANGLE_DEG
                else:
                    yaw_diff = o_yaw - shared_state.current_target_yaw
                    while yaw_diff > 180: yaw_diff -= 360;
                    while yaw_diff < -180: yaw_diff += 360
                    shared_state.current_target_yaw += clamp(yaw_diff, -AVOID_PATH_CORRECTION_INCREMENT_DEG, AVOID_PATH_CORRECTION_INCREMENT_DEG)
                    if abs(yaw_diff) < 1.0: shared_state.current_target_yaw = o_yaw
                while shared_state.current_target_yaw > 180: shared_state.current_target_yaw -= 360
                while shared_state.current_target_yaw < -180: shared_state.current_target_yaw += 360

        if is_walking:
            execute_walk_step(body_controller, policy_model)
            dt = time.perf_counter() - last_update_time
            with shared_state.lock:
                rad = math.radians(shared_state.current_yaw)
                shared_state.estimated_position += np.array([WALK_SPEED_MPS*dt*math.sin(rad), 0, WALK_SPEED_MPS*dt*math.cos(rad)])
                shared_state.estimated_path_history.append(shared_state.estimated_position.copy())
                if len(shared_state.estimated_path_history)>300: shared_state.estimated_path_history.pop(0)
        else: time.sleep(0.1)
        last_update_time = time.perf_counter()
    print("Robot control thread stopped.")

# ==============================================================================
# --- 7. VOICE COMMAND PROCESSING ---
# ==============================================================================
def process_voice_command(command_text: str, body_controller):
    command_text = command_text.lower().strip().replace(",", "").replace(".", "")
    print(f"\n[VOICE COMMAND] Heard: '{command_text}'")

    activation_words = ["perrito", "verito", "para esto", "prito", "perito"]
    if not any(word in command_text for word in activation_words):
        print(f"[INFO] No activation keyword detected. Ignoring.")
        return

    with shared_state.lock: current_mode = shared_state.robot_mode
    is_turn_command = "derecha" in command_text or "izquierda" in command_text
    
    if current_mode != "IDLE" and not is_turn_command:
        print("[ACTION] Halting current activity to execute new command.")
        with shared_state.lock: shared_state.robot_mode = "IDLE"
        command_set_angles(body_controller, get_stance_angles()); time.sleep(0.5)

    if any(word in command_text for word in ["parate", "para ti", "arriba"]):
        print("[ACTION] Standing up."); command_set_angles(body_controller, get_stance_angles())
    elif "camina" in command_text:
        print("[ACTION] Starting simple walk forward.")
        with shared_state.lock:
            shared_state.current_target_yaw = shared_state.current_yaw; shared_state.robot_mode = "SIMPLE_WALK"; body_controller.set_all_control_status(True)
    elif "sigueme" in command_text or "sígueme" in command_text or "y gane" in command_text or "sigame" in command_text or "sígame" in command_text:
        if not REALSENSE_AVAILABLE: print("[ERROR] Cannot follow: RealSense camera not detected."); return
        print("[ACTION] Engaging FOLLOW ME mode.")
        with shared_state.lock:
            shared_state.robot_mode = "FOLLOWING"; shared_state.estimated_position, shared_state.estimated_path_history = np.zeros(3), [np.zeros(3)]; body_controller.set_all_control_status(True)
    elif any(word in command_text for word in ["esquiva", "esquivame", "esquívame"]):
        if not REALSENSE_AVAILABLE: print("[ERROR] Cannot avoid: RealSense camera not detected."); return
        print("[ACTION] Engaging OBSTACLE AVOIDANCE mode.")
        with shared_state.lock:
            shared_state.robot_mode, shared_state.original_target_yaw, shared_state.current_target_yaw = "AVOIDING", shared_state.current_yaw, shared_state.current_yaw
            shared_state.estimated_position, shared_state.estimated_path_history = np.zeros(3), [np.zeros(3)]; body_controller.set_all_control_status(True)
    elif "derecha" in command_text:
        with shared_state.lock:
            if shared_state.robot_mode != "IDLE": shared_state.current_target_yaw -= 15.0; print(f"[ACTION] Turning right. New target: {shared_state.current_target_yaw:.1f}")
            else: print("[INFO] Must be walking to turn.")
    elif "izquierda" in command_text:
        with shared_state.lock:
            if shared_state.robot_mode != "IDLE": shared_state.current_target_yaw += 15.0; print(f"[ACTION] Turning left. New target: {shared_state.current_target_yaw:.1f}")
            else: print("[INFO] Must be walking to turn.")
    elif "para" in command_text:
        print("[ACTION] Stopping and standing by.") # Logic at top of function already handled this
    elif any(word in command_text for word in ["sientate", "sentado"]):
        print("[ACTION] Sitting down."); command_set_angles(body_controller, get_partial_sit_angles())
    elif any(word in command_text for word in ["agachate", "agachado"]):
        print("[ACTION] Lying down."); command_set_angles(body_controller, get_sit_angles())
    elif "apaga" in command_text:
        print("[ACTION] Disabling motor control.");
        with shared_state.lock: shared_state.robot_mode = "IDLE"; body_controller.set_all_control_status(False)
    else:
        print("[INFO] Command heard, but no specific action matched.")

def voice_recognition_loop(body_controller):
    model_size, energy, rec_tout, phrase_tout = "base", 100, 2.0, 3.0
    print(f"Loading Whisper model '{model_size}'..."); audio_model = whisper.load_model(model_size); print("Whisper loaded.")
    q = Queue(); rec = sr.Recognizer(); rec.energy_threshold, rec.dynamic_energy_threshold = energy, True
    try: source = sr.Microphone(sample_rate=16000)
    except Exception as e: print(f"FATAL: No microphone found: {e}"); stop_program_event.set(); return
    with source: rec.adjust_for_ambient_noise(source)
    def cb(_, audio:sr.AudioData): q.put(audio.get_raw_data())
    print("Starting background listener..."); stop_listening = rec.listen_in_background(source, cb, phrase_time_limit=rec_tout)
    print("LISTENING... Say 'Perrito' + command.")
    phrase_time, phrase_bytes = None, bytes()
    while not stop_program_event.is_set():
        try:
            now = datetime.utcnow()
            if not q.empty():
                phrase_complete = False;
                if phrase_time and now-phrase_time > timedelta(seconds=phrase_tout): phrase_bytes, phrase_complete = bytes(), True
                phrase_time, audio_data = now, b''.join(q.queue); q.queue.clear(); phrase_bytes += audio_data
                if len(phrase_bytes)>0:
                    audio_np = np.frombuffer(phrase_bytes, dtype=np.int16).astype(np.float32)/32768.0
                    text = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available(), language="es")['text'].strip()
                    if text and phrase_complete: process_voice_command(text, body_controller); phrase_bytes = bytes()
            time.sleep(0.1)
        except Exception as e: print(f"Error in voice loop: {e}"); break
    stop_listening(wait_for_stop=False); print("Voice recognition thread stopped.")

# ==============================================================================
# --- 8. MAIN EXECUTION ---
# ==============================================================================
def on_key_press(key):
    if key == keyboard.Key.esc: print("\nESC pressed. Shutting down..."); stop_program_event.set(); return False

if __name__ == "__main__":
    body_controller, rs_components, walk_policy_model = None, None, None
    try:
        print("Initializing QuadPilotBody..."); body_controller = QuadPilotBody(listen_for_broadcasts=True); time.sleep(2.0)
        body_controller.set_control_params(P=1.5, I=0.0, D=0.3, dead_zone=5, pos_thresh=5)
        body_controller.set_all_pins(MOTOR_PINS); time.sleep(0.5); body_controller.reset_all(); time.sleep(0.2)
        body_controller.set_all_control_status(True); print("QuadPilotBody setup complete.")

        if REALSENSE_AVAILABLE:
            print("Initializing RealSense camera..."); pipeline=rs.pipeline(); config=rs.config(); config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30); pipeline.start(config)
            pc=rs.pointcloud(); sf=rs.spatial_filter(); sf.set_option(rs.option.filter_magnitude, 2); sf.set_option(rs.option.filter_smooth_alpha, 0.5); sf.set_option(rs.option.filter_smooth_delta, 20); rs_components = (pipeline, pc, sf); print("RealSense camera setup complete.")
        
        print(f"Loading walk policy from '{MODEL_PATH}'...");
        if not os.path.exists(MODEL_PATH): raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        walk_policy_model = WalkPolicy(); walk_policy_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')); walk_policy_model.eval(); print("Walk policy model loaded.")

        print("\n--- Calibrating IMU Yaw Offset (keep robot still)... ---")
        # Robot remains in its initial (limp) state for calibration. No set_angles here.
        yaw_readings = []
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < 0.5:
            dmp = body_controller.get_latest_dmp_data_for_esp(ESP_WITH_IMU)
            if dmp and 'ypr_deg' in dmp and 'yaw' in dmp['ypr_deg']: yaw_readings.append(dmp['ypr_deg']['yaw'])
            time.sleep(0.02)
        if not yaw_readings: print("WARNING: Could not get yaw for calibration. Offset is 0.")
        else:
            with shared_state.lock: shared_state.yaw_offset = sum(yaw_readings) / len(yaw_readings)
            print(f"Calibration complete. Yaw Offset: {shared_state.yaw_offset:.2f} degrees.")
    except Exception as e:
        print(f"FATAL ERROR during initialization: {e}")
        if body_controller: body_controller.close();
        if rs_components: rs_components[0].stop();
        sys.exit(1)

    print("\n" + "="*60 + "\n QuadPilot Master Voice Control System Activated\n" + "="*60)
    print(" Awaiting voice commands... ('Perrito ...')\n Press [Esc] key at any time to SHUT DOWN.\n" + "="*60 + "\n")

    voice_thread_handle = threading.Thread(target=voice_recognition_loop, args=(body_controller,), daemon=True); voice_thread_handle.start()
    control_thread_handle = threading.Thread(target=robot_control_thread_func, args=(body_controller, walk_policy_model, rs_components), daemon=True); control_thread_handle.start()
    if REALSENSE_AVAILABLE: gui_thread_handle = threading.Thread(target=gui_loop_func, daemon=True); gui_thread_handle.start()
    
    listener = keyboard.Listener(on_press=on_key_press)
    try:
        listener.start(); stop_program_event.wait()
    finally:
        print("\n--- Initiating Shutdown Sequence ---")
        with shared_state.lock: shared_state.robot_mode = "IDLE"
        if listener.is_alive(): listener.stop()
        for thread in [voice_thread_handle, control_thread_handle, gui_thread_handle]:
            if thread and thread.is_alive(): thread.join(timeout=2.0)
        if body_controller:
            print("Disabling motors and closing connection..."); body_controller.set_all_control_status(False); time.sleep(0.1); body_controller.reset_all(); body_controller.close()
        if rs_components: print("Stopping RealSense camera..."); rs_components[0].stop()
        print("Shutdown complete. Goodbye!")