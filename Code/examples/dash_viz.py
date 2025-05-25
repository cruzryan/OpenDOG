# dash_web.py
import os
import sys
import time
import threading
import json
import webbrowser

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

try:
    from quadpilot import QuadPilotBody
except ImportError:
    print(f"ERROR: Could not import QuadPilotBody from 'quadpilot.py'. Ensure 'quadpilot.py' is in: {code_dir}")
    sys.exit(1)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'quad_dashboard_secret_final_fix_attempt_dmp!' # Slightly changed key
socketio = SocketIO(app, async_mode='threading')

body_monitor = None
MONITORING_INTERVAL = 0.05
DATA_SEND_INTERVAL = 0.1

MOTOR_CONFIG = [
    {"id": 0, "name": "FL_knee", "leg": "FL", "type": "Knee"}, {"id": 1, "name": "FR_tigh", "leg": "FR", "type": "Hip"},
    {"id": 2, "name": "FR_knee", "leg": "FR", "type": "Knee"}, {"id": 3, "name": "FL_tigh", "leg": "FL", "type": "Hip"},
    {"id": 4, "name": "BR_knee", "leg": "BR", "type": "Knee"}, {"id": 5, "name": "BR_tigh", "leg": "BR", "type": "Hip"},
    {"id": 6, "name": "BL_knee", "leg": "BL", "type": "Knee"}, {"id": 7, "name": "BL_tigh", "leg": "BL", "type": "Hip"},
]
NUM_MOTORS = len(MOTOR_CONFIG)
NUM_MOTORS_PER_ESP = 4
LEGS_ORDER = ["FR", "FL", "BR", "BL"]

DEFAULT_MOTOR_PINS = [
    (39, 40, 41, 42),  (16, 15, 7, 6),
    (17, 18, 5, 4),    (37, 38, 1, 2),
    (37, 38, 1, 2),    (40, 39, 42, 41),
    (15, 16, 6, 7),    (18, 17, 4, 5),
]

DEFAULT_P = 0.9; DEFAULT_I = 0.0; DEFAULT_D = 0.3; DEFAULT_DEAD_ZONE = 5; DEFAULT_POS_THRESH = 5
actual_robot_control_is_on = False

def perform_robot_initialization(robot_body: QuadPilotBody):
    global actual_robot_control_is_on
    print("Dashboard: Attempting robot initialization sequence...")
    results_detail = {}
    results_detail['params_set'] = robot_body.set_control_params(P=DEFAULT_P, I=DEFAULT_I, D=DEFAULT_D, dead_zone=DEFAULT_DEAD_ZONE, pos_thresh=DEFAULT_POS_THRESH)
    time.sleep(0.1)
    results_detail['pins_set'] = robot_body.set_all_pins(DEFAULT_MOTOR_PINS)
    time.sleep(0.5)
    results_detail['hw_reset'] = robot_body.reset_all()
    time.sleep(0.2)
    control_enable_api_success = robot_body.set_all_control_status(True)
    results_detail['control_enabled_api_call_success'] = control_enable_api_success
    time.sleep(DATA_SEND_INTERVAL * 2)
    esp0_reported_control = robot_body.is_esp_control_reported_on(0) if robot_body.is_data_available_from_esp(0) else False
    esp1_reported_control = robot_body.is_esp_control_reported_on(1) if robot_body.is_data_available_from_esp(1) else False
    actual_robot_control_is_on = esp0_reported_control and esp1_reported_control
    if results_detail['params_set'] and results_detail['pins_set'] and results_detail['hw_reset'] and control_enable_api_success and actual_robot_control_is_on:
        message = "Robot initialization sequence complete. Control ENABLED (confirmed by ESPs)."
    else:
        message = (f"Robot init status: API enable call: {'OK' if control_enable_api_success else 'FAIL'}. "
                   f"Actual ctrl (E0/E1): {esp0_reported_control}/{esp1_reported_control}. "
                   f"Final: {'ENABLED' if actual_robot_control_is_on else 'DISABLED'}. Details: {results_detail}")
    print(f"Dashboard: {message}")
    return actual_robot_control_is_on, message

def data_broadcasting_thread():
    global body_monitor, actual_robot_control_is_on
    last_send_time = 0
    while True:
        if body_monitor and not body_monitor._is_closed:
            esp0_reported_on = body_monitor.is_esp_control_reported_on(0) if body_monitor.is_data_available_from_esp(0) else False
            esp1_reported_on = body_monitor.is_esp_control_reported_on(1) if body_monitor.is_data_available_from_esp(1) else False
            current_actual_state = esp0_reported_on and esp1_reported_on
            if current_actual_state != actual_robot_control_is_on:
                actual_robot_control_is_on = current_actual_state

            all_data = {"timestamp": time.time(), "esps": [], "control_enabled_server_state": actual_robot_control_is_on}
            current_time_for_age = time.time()

            for esp_idx in range(len(body_monitor.ips)):
                last_rx_ts = body_monitor.get_last_packet_received_timestamp_for_esp(esp_idx)
                data_age = (current_time_for_age - last_rx_ts) * 1000 if last_rx_ts > 0 and body_monitor.is_data_available_from_esp(esp_idx) else -1.0

                esp_data_available = body_monitor.is_data_available_from_esp(esp_idx)
                is_dmp_ready_for_this_esp = body_monitor.is_dmp_ready_for_esp(esp_idx) if esp_data_available else False

                esp_packet = {
                    "ip": body_monitor.ips[esp_idx],
                    "data_available": esp_data_available,
                    "dmp_ready": is_dmp_ready_for_this_esp, # Changed from mpu_available
                    "dmp_data": {}, # Initialize for DMP data
                    "imu": {}, # Kept for potential legacy/fallback from get_imu_data command
                    "motors": [],
                    "esp_control_actually_on": body_monitor.is_esp_control_reported_on(esp_idx),
                    "data_age_ms": data_age
                }

                if esp_data_available:
                    if is_dmp_ready_for_this_esp:
                        # Get the structured DMP data
                        dmp_payload = body_monitor.get_latest_dmp_data_for_esp(esp_idx)
                        if dmp_payload: # Ensure it's not None or empty
                            esp_packet["dmp_data"] = dmp_payload
                    else:
                        # If DMP is not ready, you might still want to get raw IMU if available
                        # from a 'get_imu_data' command, but broadcast usually won't have it
                        # For simplicity, if DMP not ready, dmp_data remains empty.
                        # You could try to populate esp_packet["imu"] here with legacy data if needed.
                        # esp_packet["imu"] = body_monitor.get_latest_imu_data_for_esp(esp_idx) # Deprecated for broadcast
                        pass


                    motor_data_raw = body_monitor.get_latest_motor_data_for_esp(esp_idx)
                    if motor_data_raw:
                        for motor_local_idx in range(NUM_MOTORS_PER_ESP):
                            global_idx = esp_idx * NUM_MOTORS_PER_ESP + motor_local_idx
                            motor_conf = next((m for m in MOTOR_CONFIG if m["id"] == global_idx), None)
                            name = motor_conf["name"] if motor_conf else f"M{global_idx}"
                            angle = motor_data_raw['angles'][motor_local_idx] if motor_local_idx < len(motor_data_raw.get('angles',[])) else 0.0
                            # Add other motor data if needed by JS (encoderPos, targetPos)
                            esp_packet["motors"].append({
                                "id": global_idx,
                                "name": name,
                                "angle": angle,
                                "encoderPos": motor_data_raw['encoderPos'][motor_local_idx] if motor_local_idx < len(motor_data_raw.get('encoderPos',[])) else 0,
                                "targetPos": motor_data_raw['targetPos'][motor_local_idx] if motor_local_idx < len(motor_data_raw.get('targetPos',[])) else 0
                            })
                all_data["esps"].append(esp_packet)

            if time.time() - last_send_time >= DATA_SEND_INTERVAL:
                socketio.emit('robot_data_update', all_data)
                last_send_time = time.time()
        time.sleep(MONITORING_INTERVAL)


@app.route('/')
def index():
    motor_config_json_str = json.dumps(MOTOR_CONFIG)
    legs_order_json_str = json.dumps(LEGS_ORDER)
    return render_template('dashboard.html',
                           motor_config_list_json=motor_config_json_str,
                           legs_order_list_json=legs_order_json_str)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('initial_config', {
        'motors': MOTOR_CONFIG,
        'legs_order': LEGS_ORDER,
        'control_enabled': actual_robot_control_is_on
    })

@socketio.on('request_robot_init')
def handle_robot_init_request():
    if body_monitor:
        current_control_state_after_init, message = perform_robot_initialization(body_monitor)
        emit('command_status', {'command': 'init_robot',
                               'success': current_control_state_after_init,
                               'message': message,
                               'control_enabled': current_control_state_after_init})
    else:
        emit('command_status', {'command': 'init_robot', 'success': False, 'message': 'Robot body monitor not available.', 'control_enabled': actual_robot_control_is_on})

@socketio.on('request_toggle_control')
def handle_toggle_control():
    global actual_robot_control_is_on
    if body_monitor:
        target_state_to_set_by_command = not actual_robot_control_is_on
        api_call_success = body_monitor.set_all_control_status(target_state_to_set_by_command)
        message_prefix = ""
        if api_call_success:
            message_prefix = f"API call to set control to {'ENABLED' if target_state_to_set_by_command else 'DISABLED'} succeeded. "
            if not target_state_to_set_by_command:
                reset_success = body_monitor.reset_all()
                message_prefix += f"Hardware reset after disable {'succeeded.' if reset_success else 'failed.'} "
        else:
            message_prefix = f"API call FAILED to set control to {'ENABLED' if target_state_to_set_by_command else 'DISABLED'}. "
        time.sleep(DATA_SEND_INTERVAL * 2.5)
        esp0_reported_control = body_monitor.is_esp_control_reported_on(0) if body_monitor.is_data_available_from_esp(0) else False
        esp1_reported_control = body_monitor.is_esp_control_reported_on(1) if body_monitor.is_data_available_from_esp(1) else False
        newly_confirmed_robot_state = esp0_reported_control and esp1_reported_control
        actual_robot_control_is_on = newly_confirmed_robot_state
        final_message = message_prefix + f"Robot control is now confirmed {'ENABLED' if actual_robot_control_is_on else 'DISABLED'} by ESPs."
        emit('command_status', {'command': 'toggle_control',
                               'success': api_call_success and (target_state_to_set_by_command == actual_robot_control_is_on),
                               'message': final_message,
                               'control_enabled': actual_robot_control_is_on})
    else:
        emit('command_status', {'command': 'toggle_control', 'success': False, 'message': 'Robot body monitor not available.', 'control_enabled': actual_robot_control_is_on})

@socketio.on('request_reset_all_hw')
def handle_reset_all_hw():
    if body_monitor:
        success = body_monitor.reset_all()
        emit('command_status', {'command': 'reset_all_hw', 'success': success, 'message': 'Hardware reset all attempted.' if success else 'Hardware reset all failed.', 'control_enabled': actual_robot_control_is_on})

@socketio.on('request_set_angles')
def handle_set_angles(data):
    if body_monitor and 'angles' in data and isinstance(data['angles'], list) and len(data['angles']) == NUM_MOTORS:
        if not actual_robot_control_is_on:
            emit('command_status', {'command': 'set_angles', 'success': False, 'message': 'Control is disabled on server. Enable control before setting angles.', 'control_enabled': actual_robot_control_is_on})
            return
        angles_to_set = [float(a) for a in data['angles']]
        success = body_monitor.set_angles(angles_to_set)
        emit('command_status', {'command': 'set_angles', 'success': success,
                               'message': 'Set angles attempted.' if success else 'Set angles failed.',
                               'control_enabled': actual_robot_control_is_on})
    else:
        emit('command_status', {'command': 'set_angles', 'success': False, 'message': 'Invalid angle data.', 'control_enabled': actual_robot_control_is_on})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def open_browser():
      try: webbrowser.open_new_tab("http://127.0.0.1:5000/")
      except Exception: print("Please open http://127.0.0.1:5000/ manually.")

if __name__ == '__main__':
    print("Initializing QuadPilotBody for monitoring (web dashboard)...")
    try:
        body_monitor = QuadPilotBody(listen_for_broadcasts=True)
        print("QuadPilotBody (monitoring instance) initialized.")

        thread = threading.Thread(target=data_broadcasting_thread, daemon=True)
        thread.start()
        print("Data broadcasting thread started.")
        threading.Timer(1.5, open_browser).start() # Reduced delay for browser opening
        print("Starting Flask-SocketIO server on http://127.0.0.1:5000/")
        # Note: allow_unsafe_werkzeug=True is for development with newer Werkzeug versions.
        # For production, consider a proper WSGI server like Gunicorn or Waitress.
        socketio.run(app, host='127.0.0.1', port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"FATAL: Failed to initialize or run dashboard server: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if body_monitor: body_monitor.close()
        print("Web dashboard server has shut down.")