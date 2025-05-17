import numpy as np
from scipy.integrate import odeint

# Simulation parameters from Arduino code
COUNTS_PER_REV = 1975
DEAD_ZONE = 10
MAX_POWER = 255
POSITION_THRESHOLD = 5
dt = 0.002
SIM_TIME = 1.0  # Reduced for faster iterations
NUM_STEPS = int(SIM_TIME / dt)

# Motor simulation parameters (from datasheet)
TORQUE_CONSTANT = 0.167 / 255  # N·m per PWM unit (rated torque)
MAX_ANGULAR_VEL = 3.874  # rad/s (37 RPM rated speed)

# Target PID values (your working values)
TARGET_KP = 0.9
TARGET_KI = 0.001
TARGET_KD = 0.3

# Motor state
class Motor:
    def __init__(self):
        self.encoder_pos = 0.0
        self.target_pos = 0.0
        self.last_error = 0.0
        self.velocity = 0.0

# Simulate motor dynamics
def motor_dynamics(state, t, power, motor_inertia, friction, gain):
    pos, vel = state
    torque = (power / MAX_POWER) * 0.167
    torque_applied = torque * (1 - abs(vel) / MAX_ANGULAR_VEL if abs(vel) < MAX_ANGULAR_VEL else 0)
    accel = (torque_applied - friction * vel) / motor_inertia
    return [vel, accel * gain]

# Auto-tuning function to find ku
def find_ku(motor_inertia, friction, gain):
    motor = Motor()
    motor.target_pos = 50  # Reduced for faster response
    motor.encoder_pos = 0
    motor.velocity = 0
    motor.last_error = 0

    kp_test = 0.1
    max_iterations = 30  # Reduced iterations
    iteration = 0
    oscillation_detected = False
    crossing_times = []

    while kp_test < 1.8 and not oscillation_detected and iteration < max_iterations:
        t = np.linspace(0, SIM_TIME, NUM_STEPS)
        positions = []
        state = [0.0, 0.0]
        crossing_times = []

        for i in range(NUM_STEPS):
            error = motor.target_pos - motor.encoder_pos
            error_delta = error - motor.last_error
            scaled_error = np.clip(error / POSITION_THRESHOLD, -1.0, 1.0)
            power = kp_test * scaled_error * MAX_POWER
            power = np.clip(power, -MAX_POWER, MAX_POWER)

            t_span = [t[i], t[i] + dt]
            sol = odeint(motor_dynamics, state, t_span, args=(power, motor_inertia, friction, gain))
            state = sol[-1]
            motor.encoder_pos = state[0]
            motor.last_error = error
            positions.append(motor.encoder_pos)

            if i > 0 and positions[i] * positions[i-1] < 0:
                crossing_times.append(t[i])

            if abs(motor.encoder_pos) > 5000:
                break

        if len(crossing_times) >= 1:  # Reduced to 1 crossing
            periods = np.diff(crossing_times)
            if len(periods) > 0 and np.std(periods) < 0.5 * np.mean(periods):  # Relaxed criterion
                oscillation_detected = True
                ku = kp_test
                pu = np.mean(periods) * 2
                return ku, pu

        kp_test += 0.05
        iteration += 1
        motor.encoder_pos = 0
        motor.velocity = 0
        motor.last_error = 0

    return None, None

# Parameter solver
def solve_motor_parameters():
    # Narrowed parameter ranges
    motor_inertia_range = [5e-8, 1e-7, 5e-7]
    friction_range = [0.0001, 0.0003, 0.0005]
    gain_range = [1000, 2000, 3000]

    best_params = (5e-8, 0.0003, 2000)  # Initial guess
    best_error = float('inf')
    max_total_iterations = 50  # Overall limit to prevent hanging
    total_iteration = 0

    # Iterate over parameter ranges
    for mi in motor_inertia_range:
        for fr in friction_range:
            for g in gain_range:
                if total_iteration >= max_total_iterations:
                    print("Reached max total iterations. Using best parameters found so far.")
                    return best_params

                print(f"Testing combination {total_iteration + 1}: inertia={mi}, friction={fr}, gain={g}")
                ku, pu = find_ku(mi, fr, g)
                total_iteration += 1

                if ku is None:
                    print("No oscillations detected for this combination.")
                    continue

                # Compute tuned PID
                kp = 0.6 * ku
                ki = 1.2 * ku / pu
                kd = 3.0 * ku * pu / 40.0

                # Compute error (focus on kp)
                error = abs(kp - TARGET_KP)

                print(f"Result: kp={kp:.3f}, ki={ki:.3f}, kd={kd:.3f}, error={error:.3f}")

                if error < best_error:
                    best_error = error
                    best_params = (mi, fr, g)

                # Stop if we're close enough
                if error < 0.05:
                    return best_params

    return best_params

if __name__ == "__main__":
    motor_inertia, friction, gain = solve_motor_parameters()
    print(f"Solved motor parameters: MOTOR_INERTIA={motor_inertia}, FRICTION={friction}, GAIN={gain}")