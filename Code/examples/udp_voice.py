# quad_voice_control.py
# Combines real-time voice recognition with quadruped robot control.

import os
import sys
import time
import threading
import argparse
from queue import Queue
from datetime import datetime, timedelta

# --- Input/Control Imports ---
from pynput import keyboard
import speech_recognition as sr

# --- AI/ML Imports ---
# Ensure you have PyTorch and Whisper installed:
# pip install torch openai-whisper
try:
    import torch
    import torch.nn as nn
    import whisper
    import numpy as np
except ImportError:
    print("FATAL ERROR: PyTorch or Whisper is not installed.")
    print("Please install them by running: pip install torch openai-whisper")
    sys.exit(1)


# --- Path Setup for QuadPilotBody ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assumes 'quadpilot' directory is a sibling to the directory containing this script
code_dir = os.path.join(os.path.dirname(script_dir), "quadpilot")
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

try:
    from body import QuadPilotBody
except ImportError:
    print(f"ERROR: Could not import QuadPilotBody. "
          f"Ensure 'quadpilot' directory is a sibling to this script's directory.")
    print(f"Attempted to look in: {code_dir}")
    sys.exit(1)


# --- Robot Configuration ---
NUM_MOTORS = 8
TARGET_MOTORS = list(range(NUM_MOTORS))
ESP_WITH_IMU = 1 # The ESP32 index (0 or 1) that sends primary IMU data
MODEL_PATH = 'walk_policy.pth' # Path to your trained walking policy

# --- Yaw Auto-Correction / Walk Configuration ---
REAL_TARGET = 0.0 # Dynamic target, will be updated by the 'camina' command
MIN_LIFT_ANGLE = 20.0
MAX_LIFT_ANGLE = 45.0
WALK_STEP_DURATION = 0.1

# Mapping: Actuator Name -> QuadPilotBody Index (0-7)
ACTUATOR_NAME_TO_INDEX_MAP = {
    "FL_tigh_actuator": 3, "FL_knee_actuator": 0,
    "FR_tigh_actuator": 1, "FR_knee_actuator": 2,
    "BR_tigh_actuator": 5, "BR_knee_actuator": 4,
    "BL_tigh_actuator": 7, "BL_knee_actuator": 6,
}

# Motor pin configurations (ENCODER_A, ENCODER_B, IN1, IN2)
MOTOR_PINS = [
    (39, 40, 41, 42),  # Motor 0 (FL_knee) -> ESP1
    (16, 15, 7, 6),    # Motor 1 (FR_tigh) -> ESP1
    (17, 18, 5, 4),    # Motor 2 (FR_knee) -> ESP1
    (37, 38, 1, 2),    # Motor 3 (FL_tigh) -> ESP1
    (37, 38, 1, 2),    # Motor 4 (BR_knee) -> ESP2
    (40, 39, 42, 41),  # Motor 5 (BR_tigh) -> ESP2
    (15, 16, 6, 7),    # Motor 6 (BL_knee) -> ESP2
    (18, 17, 4, 5),    # Motor 7 (BL_tigh) -> ESP2
]


# --- Neural Network Definition (must match the training script) ---
class WalkPolicy(nn.Module):
    def __init__(self):
        super(WalkPolicy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)


# --- Load Trained Policy Model ---
walk_policy_model = None
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'.")
    print(f"Loading walk policy from '{MODEL_PATH}'...")
    walk_policy_model = WalkPolicy()
    walk_policy_model.load_state_dict(torch.load(MODEL_PATH))
    walk_policy_model.eval()
    print("Walk policy model loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load the PyTorch model: {e}")
    sys.exit(1)


# --- Initialize QuadPilotBody and Setup ---
body_controller = None
initial_setup_successful = False
try:
    body_controller = QuadPilotBody(listen_for_broadcasts=True)
    print("QuadPilotBody instance initialized. Waiting for ESPs...")
    time.sleep(2.0)

    if not body_controller.set_control_params(P=1.5, I=0.0, D=0.3, dead_zone=5, pos_thresh=5):
        raise Exception("Failed to set control parameters.")
    time.sleep(0.1)
    if not body_controller.set_all_pins(MOTOR_PINS):
        raise Exception("Failed to set motor pins.")
    time.sleep(0.5)
    if not body_controller.reset_all():
        raise Exception("Failed to reset all motors.")
    time.sleep(0.2)
    if not body_controller.set_all_control_status(True):
        raise Exception("Failed to enable control for all motors.")

    initial_setup_successful = True
    print("\nInitial Robot setup complete.")
except Exception as e:
    print(f"FATAL: Failed during QuadPilotBody initialization: {e}")
    if body_controller:
        body_controller.close()
    sys.exit(1)


# --- Global State Variables ---
motor_control_enabled = True
is_walking = False
stop_program_event = threading.Event() # Event to signal all threads to stop
walk_thread = None # A handle for the walking thread

# --- Helper Functions ---
def get_stance_angles():
    # "Perrito parate"
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

def get_partial_sit_angles():
    # "Perrito sientate" (partial sit, back legs down)
    # Start with standing pose and set back legs to 0
    pose = get_stance_angles()
    pose[ACTUATOR_NAME_TO_INDEX_MAP["BR_tigh_actuator"]] = 0.0
    pose[ACTUATOR_NAME_TO_INDEX_MAP["BR_knee_actuator"]] = 0.0
    pose[ACTUATOR_NAME_TO_INDEX_MAP["BL_tigh_actuator"]] = 0.0
    pose[ACTUATOR_NAME_TO_INDEX_MAP["BL_knee_actuator"]] = 0.0
    return pose

def get_sit_angles():
    # "Perrito agachate" (full sit/down, all motors to 0)
    return [0.0] * NUM_MOTORS

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def interruptible_sleep(duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        if not is_walking or stop_program_event.is_set(): break
        time.sleep(0.02)

def get_policy_action(yaw_error: float) -> tuple[float, float]:
    input_tensor = torch.FloatTensor([[yaw_error]])
    with torch.no_grad():
        predicted_N, predicted_Y = walk_policy_model(input_tensor).squeeze().tolist()
    return predicted_N, predicted_Y

# --- Core Action Functions ---
def command_set_angles(target_angles: list[float]):
    if not motor_control_enabled:
        print("Cannot set angles, motor control is disabled.")
        return
    if not body_controller.set_angles(target_angles):
        print("Failed to set angles (no OK from ESPs).")

def execute_autocorrect_walk():
    global is_walking
    print(f"\n--- ACTION: Starting Auto-Correcting Walk (Target Yaw: {REAL_TARGET:.1f}) ---")
    step_index = 0
    idx_fr_knee = ACTUATOR_NAME_TO_INDEX_MAP["FR_knee_actuator"]
    idx_bl_knee = ACTUATOR_NAME_TO_INDEX_MAP["BL_knee_actuator"]
    idx_fl_knee = ACTUATOR_NAME_TO_INDEX_MAP["FL_knee_actuator"]
    idx_br_knee = ACTUATOR_NAME_TO_INDEX_MAP["BR_knee_actuator"]
    stance_pose = get_stance_angles()

    try:
        while is_walking and motor_control_enabled and not stop_program_event.is_set():
            dmp_data = body_controller.get_latest_dmp_data_for_esp(ESP_WITH_IMU)
            current_yaw = dmp_data.get('ypr_deg', {}).get('yaw', 0.0) if dmp_data else 0.0
            yaw_error = current_yaw - REAL_TARGET

            N, Y = get_policy_action(yaw_error)
            N = clamp(N, MIN_LIFT_ANGLE, MAX_LIFT_ANGLE)
            Y = clamp(Y, MIN_LIFT_ANGLE, MAX_LIFT_ANGLE)

            print(f"\rWalk Cycle: Yaw={current_yaw:.1f}, Target={REAL_TARGET:.1f}, Err={yaw_error:.1f} -> Policy N={N:.1f}, Y={Y:.1f}", end="")

            # Step 1: Lift FR/BL
            step_pose = stance_pose[:]
            step_pose[idx_fr_knee] = N; step_pose[idx_bl_knee] = -N
            body_controller.set_angles(step_pose)
            interruptible_sleep(WALK_STEP_DURATION)
            if not is_walking: break

            # Step 2: Plant All
            body_controller.set_angles(stance_pose)
            interruptible_sleep(WALK_STEP_DURATION)
            if not is_walking: break

            # Step 3: Lift FL/BR
            step_pose = stance_pose[:]
            step_pose[idx_fl_knee] = Y; step_pose[idx_br_knee] = -Y
            body_controller.set_angles(step_pose)
            interruptible_sleep(WALK_STEP_DURATION)
            if not is_walking: break

            # Step 4: Plant All
            body_controller.set_angles(stance_pose)
            interruptible_sleep(WALK_STEP_DURATION)

            step_index += 4
    finally:
        is_walking = False
        print("\n--- Walk Sequence Finished ---")
        if not stop_program_event.is_set():
            print("Returning to stance pose.")
            command_set_angles(get_stance_angles())

# --- Voice Command Processing ---
def process_voice_command(command_text: str):
    global is_walking, motor_control_enabled, walk_thread, REAL_TARGET

    # Make matching capitalization and comma agnostic
    command_text = command_text.lower().strip().replace(",", "").replace(".", "")
    print(f"\n[VOICE COMMAND] Heard: '{command_text}'")

    activation_words = ["perrito", "verito", "para esto", "prito", "perito"]
    if not any(word in command_text for word in activation_words):
        print(f"[INFO] No activation keyword ({', '.join(activation_words)}) detected. Ignoring.")
        return

    # --- Stop walking first for any new *movement* command (but not for turning) ---
    is_turn_command = "derecha" in command_text or "izquierda" in command_text
    if is_walking and "camina" not in command_text and not is_turn_command:
        print("[ACTION] Stopping current walk to execute new command...")
        is_walking = False
        if walk_thread and walk_thread.is_alive():
            walk_thread.join(timeout=1.0)

    # --- Command Matching ---
    if any(word in command_text for word in ["parate", "para ti", "arriba"]):
        print("[ACTION] Standing up.")
        command_set_angles(get_stance_angles())

    elif "camina" in command_text:
        if not is_walking:
            dmp_data = body_controller.get_latest_dmp_data_for_esp(ESP_WITH_IMU)
            current_yaw = dmp_data.get('ypr_deg', {}).get('yaw', 0.0) if dmp_data else REAL_TARGET
            REAL_TARGET = current_yaw
            print(f"[ACTION] Starting walk. New target yaw set to: {REAL_TARGET:.1f} degrees.")

            is_walking = True
            motor_control_enabled = True
            body_controller.set_all_control_status(True)
            walk_thread = threading.Thread(target=execute_autocorrect_walk, daemon=True)
            walk_thread.start()
        else:
            print("[INFO] Already walking.")

    # --- NEW: Turning commands (only work while walking) ---
    elif "derecha" in command_text:
        if is_walking:
            REAL_TARGET += 10.0 # Increased for more noticeable turn
            print(f"[ACTION] Turning right. New target yaw: {REAL_TARGET:.1f}")
        else:
            print("[INFO] Cannot turn. Robot is not walking. Say 'Perrito camina' first.")

    elif "izquierda" in command_text:
        if is_walking:
            REAL_TARGET -= 10.0 # Increased for more noticeable turn
            print(f"[ACTION] Turning left. New target yaw: {REAL_TARGET:.1f}")
        else:
            print("[INFO] Cannot turn. Robot is not walking. Say 'Perrito camina' first.")

    elif "para" in command_text:
        print("[ACTION] Stopping.")
        is_walking = False

    elif any(word in command_text for word in ["sientate", "sentado"]):
        print("[ACTION] Sitting down (back legs only).")
        command_set_angles(get_partial_sit_angles())

    elif any(word in command_text for word in ["agachate", "agachado", "bachate", "gachado"]):
        print("[ACTION] Lying down (all motors to 0).")
        command_set_angles(get_sit_angles())

    elif "apagado" in command_text or "apagar" in command_text or "apágate" in command_text or "apagate" in command_text:
        print("[ACTION] Disabling motor control.")
        is_walking = False
        motor_control_enabled = False
        body_controller.set_all_control_status(False)

    elif not is_turn_command: # Avoids "no action matched" for turn commands
        print("[INFO] Command understood but no specific action matched.")


# --- Voice Recognition Loop (to be run in a thread) ---
def voice_recognition_loop():
    # --- Voice Config ---
    model_size = "base"
    energy_threshold = 100
    record_timeout = 2.0
    phrase_timeout = 3.0

    print(f"Loading Whisper model '{model_size}'...")
    audio_model = whisper.load_model(model_size)
    print("Whisper model loaded.")

    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = energy_threshold
    recorder.dynamic_energy_threshold = True

    try:
        source = sr.Microphone(sample_rate=16000)
    except Exception as e:
        print(f"FATAL: Could not find a microphone. Error: {e}")
        stop_program_event.set()
        return

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData):
        data = audio.get_raw_data()
        data_queue.put(data)

    print("Starting background listener...")
    stop_listening = recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    print("LISTENING... Say 'Perrito' or 'Verito' + command.")

    phrase_time = None
    phrase_bytes = bytes()

    while not stop_program_event.is_set():
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_bytes = bytes()
                    phrase_complete = True

                phrase_time = now

                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                phrase_bytes += audio_data

                if len(phrase_bytes) > 0:
                    audio_np = np.frombuffer(phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available(), language="es")
                    text = result['text'].strip()

                    if text and phrase_complete:
                        process_voice_command(text)
                        phrase_bytes = bytes()

            time.sleep(0.1)
        except Exception as e:
            print(f"Error in voice loop: {e}")
            break

    stop_listening(wait_for_stop=False)
    print("Voice recognition thread stopped.")


# --- Main Execution ---
def on_key_press(key):
    if key == keyboard.Key.esc:
        print("\nESC pressed. Initiating shutdown...")
        stop_program_event.set()
        return False

if __name__ == "__main__":
    if not initial_setup_successful or not walk_policy_model:
        print("Exiting due to initialization failure.")
        sys.exit(1)

    print("\n" + "="*60)
    print(" QuadPilot Voice Control System Activated")
    print("="*60)
    print(" --- VOICE COMMANDS (in Spanish) ---")
    print(" 'Perrito parate' / 'arriba'      -> Stand up")
    print(" 'Perrito camina'                 -> Start walking (maintains current direction)")
    print(" 'Perrito gira a la derecha'      -> Turn right (while walking)")
    print(" 'Perrito gira a la izquierda'    -> Turn left (while walking)")
    print(" 'Perrito para'                   -> Stop walking and stand")
    print(" 'Perrito sientate' / 'sentado'   -> Sit down (back legs only)")
    print(" 'Perrito agachate' / 'agachado'  -> Lie down (all motors to 0)")
    print(" 'Perrito apagate'                -> Disable motor power")
    print("\n --- SAFETY CONTROLS ---")
    print(" Press [Esc] key at any time to SHUT DOWN EVERYTHING.")
    print("="*60 + "\n")

    voice_thread = threading.Thread(target=voice_recognition_loop, daemon=True)
    voice_thread.start()

    listener = keyboard.Listener(on_press=on_key_press)
    try:
        listener.start()
        print("\nKeyboard listener for [Esc] started. Ready for commands.")
        stop_program_event.wait()

    finally:
        print("\n--- Initiating Shutdown Sequence ---")
        is_walking = False

        if listener.is_alive():
            listener.stop()

        if voice_thread.is_alive():
            voice_thread.join(timeout=2.0)

        if walk_thread and walk_thread.is_alive():
            walk_thread.join(timeout=2.0)

        if body_controller:
            print("Disabling motors and closing connection...")
            body_controller.set_all_control_status(False)
            time.sleep(0.1)
            body_controller.reset_all()
            body_controller.close()

        print("Shutdown complete. Goodbye!")