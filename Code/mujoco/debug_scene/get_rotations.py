import numpy as np
from scipy.spatial.transform import Rotation as R

def multiply_n_quaternions(*quaternions):
    """
    Multiplica una secuencia de cuaterniones en el orden en que se aplican.
    
    Parámetros:
        *quaternions: Lista de cuaterniones en formato (x, y, z, w)

    Retorna:
        Cuaternión resultante en formato (w, x, y, z) para MuJoCo.
    """
    # Convertimos la lista de cuaterniones a objetos Rotation
    rotations = [R.from_quat(q) for q in quaternions]

    # Multiplicamos en orden de aplicación (de izquierda a derecha)
    q_final = np.prod(rotations)  # Multiplica todos los cuaterniones en secuencia

    # Convertimos a formato (w, x, y, z) para MuJoCo
    quat_result = q_final.as_quat()
    return quat_result[3], quat_result[0], quat_result[1], quat_result[2]

# 🔹 Ejemplo con 3 rotaciones:
q1 = R.from_euler('x', -90, degrees=True).as_quat()  # 90° en X
q2 = R.from_euler('y', 90, degrees=True).as_quat()  # 90° en X
q3 = R.from_euler('z', 180, degrees=True).as_quat()
# Multiplicamos cualquier cantidad de cuaterniones
q_final = multiply_n_quaternions(q1, q3)

print("Cuaternión resultante para MuJoCo:", q_final)
