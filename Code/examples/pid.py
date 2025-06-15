import matplotlib.pyplot as plt
import numpy as np

# Parámetros del PID (sacados de tu código)
kp = 0.9
ki = 0.1
kd = 31.3
DEAD_ZONE = 0
POSITION_THRESHOLD = 5
MAX_POWER = 255
dt = 0.002  # 2 ms

# Simulación mejorada del motor con fricción y ruido
def motor_dynamics(position, power, dt):
    # Dinámica del motor: potencia genera velocidad, pero hay fricción
    friction = 0.005  # Coeficiente de fricción (simula resistencia)
    velocity = power * 0.1 - friction * position  # La potencia mueve, pero la fricción resiste
    new_position = position + velocity * dt
    
    # Agregar ruido para simular imperfecciones del encoder
    noise = np.random.normal(0, 0.01)  # Ruido gaussiano con desviación estándar de 0.5 conteos
    new_position += noise
    return max(new_position, 0)  # No permitimos posiciones negativas

# Simulación del PID
def simulate_pid(target_pos, initial_pos, simulation_time):
    time_steps = int(simulation_time / dt)
    positions = np.zeros(time_steps)
    measured_positions = np.zeros(time_steps)  # Posición con ruido
    errors = np.zeros(time_steps)
    p_terms = np.zeros(time_steps)
    i_terms = np.zeros(time_steps)
    d_terms = np.zeros(time_steps)
    powers = np.zeros(time_steps)

    position = initial_pos
    integral_error = 0
    last_error = 0

    for t in range(time_steps):
        # Simular medición con ruido
        measured_position = position + np.random.normal(0, 0.5)  # Ruido en la medición
        measured_positions[t] = measured_position

        # Error (usamos la posición medida, como en un sistema real)
        error = target_pos - measured_position
        errors[t] = error

        # Término P
        if abs(error) <= DEAD_ZONE:
            scaled_error = 0
        else:
            scaled_error = np.clip(error / POSITION_THRESHOLD, -1.0, 1.0)
        p_term = kp * scaled_error * MAX_POWER
        p_terms[t] = p_term

        # Término I
        if ki != 0 and abs(error) < MAX_POWER / abs(ki):
            integral_error += error * dt
        elif ki != 0:
            integral_error = np.clip(integral_error, -MAX_POWER / abs(ki), MAX_POWER / abs(ki))
        else:
            integral_error = 0
        i_term = ki * integral_error
        i_terms[t] = i_term

        # Término D
        error_delta = error - last_error
        d_term = kd * (error_delta / dt)
        if abs(error) <= DEAD_ZONE * 5:
            d_term *= 3.0
        d_term = np.clip(d_term, -MAX_POWER / 2, MAX_POWER / 2)
        d_terms[t] = d_term
        last_error = error

        # Potencia total
        power = p_term + i_term + d_term
        power = np.clip(power, -MAX_POWER, MAX_POWER)
        powers[t] = power

        # Actualizar posición real
        position = motor_dynamics(position, power, dt)
        positions[t] = position

    return positions, measured_positions, errors, p_terms, i_terms, d_terms, powers

# Configuración de la simulación
target_pos = 100  # Posición objetivo
initial_pos = 50 #Posición inicial
simulation_time = 20.0  # 2 egundos de simulación
  
# Correr simulación
positions, measured_positions, errors, p_terms, i_terms, d_terms, powers = simulate_pid(target_pos, initial_pos, simulation_time)

# Crear gráficos
time = np.arange(0, simulation_time, dt)

# Gráfico 1: Posición real, medida y deseada
plt.figure(figsize=(10, 6))
plt.plot(time, positions, label='Posición real', color='b', alpha=0.7)
plt.plot(time, measured_positions, label='Posición medida (con ruido)', color='c', alpha=0.5, linestyle='--')
plt.axhline(y=target_pos, color='r', linestyle='--', label='Posición deseada')
plt.title('Comportamiento de la posición con PID (con ruido y fricción)')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Posición')
plt.grid(True)
plt.legend()
plt.savefig('posicion_pid.png')

# Gráfico 2: Error
plt.figure(figsize=(10, 6))
plt.plot(time, errors, label='Error', color='g', alpha=0.7)
plt.axhline(y=0, color='k', linestyle='--')
plt.title('Error de posición con ruido')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Error (conteos)')
plt.grid(True)
plt.legend()
plt.savefig('error_pid.png')

# Gráfico 3: Términos P, I, D
plt.figure(figsize=(10, 6))
plt.plot(time, p_terms, label='Término P', color='b', alpha=0.7)
plt.plot(time, i_terms, label='Término I', color='r', alpha=0.7)
plt.plot(time, d_terms, label='Término D', color='g', alpha=0.7)
plt.title('Términos del PID bajo condiciones realistas')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Contribución a la potencia')
plt.grid(True)
plt.legend()
plt.savefig('terminos_pid.png')

# Gráfico 4: Potencia
plt.figure(figsize=(10, 6))
plt.plot(time, powers, label='Potencia', color='m', alpha=0.7)
plt.title('Potencia aplicada al motor con ruido y fricción')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Potencia (PWM)')
plt.grid(True)
plt.legend()
plt.savefig('potencia_pid.png')

# Opcional: mostrar gráficos
# plt.show()