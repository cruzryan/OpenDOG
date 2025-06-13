import numpy as np

class ScaleActions:
    """
    Clase para convertir acciones de una red neuronal (rango [-1, 1]) a ángulos
    específicos para una simulación en MuJoCo o para un robot físico real.
    """

    def __init__(self, real_robot_action_range_deg=30.0):
        """
        Inicializa la clase con todas las configuraciones necesarias.

        Args:
            real_robot_action_range_deg (float): El rango de movimiento en grados
                (positivo y negativo) que se permitirá alrededor de la posición
                de pie para el robot real.
        """
        # Es crucial definir un orden consistente para los actuadores.
        # El vector de acción de la red neuronal debe seguir este orden.
        self.joint_order = [
            'FR_tigh', 'FR_knee',
            'FL_tigh', 'FL_knee',
            'BR_tigh', 'BR_knee',
            'BL_tigh', 'BL_knee'
        ]

        # --- Configuración para el Robot Real (en Grados) ---

        # Posición de pie (neutral) para el robot real en grados.
        self.real_robot_stand_pose_deg = {
            'FR_tigh': -45.0, 'FR_knee':  45.0,
            'FL_tigh':  45.0, 'FL_knee':  45.0,
            'BR_tigh':  45.0, 'BR_knee': -45.0,
            'BL_tigh':  45.0, 'BL_knee': -45.0,
        }

        # Rango de movimiento permitido desde la posición de pie (ej: +/- 30 grados)
        self.real_robot_action_range_deg = real_robot_action_range_deg


        # --- Configuración para la Simulación MuJoCo (en Radianes) ---

        # Límites de los actuadores en MuJoCo [min_radianes, max_radianes]
        # Nota: He corregido los nombres para que sean consistentes.
        self.mujoco_limits_rad = {
            'FR_tigh': (2.36, 2.8), 'FR_knee': (-1.8, -1.20),
            'FL_tigh': (2.36, 2.8), 'FL_knee': (-1.8, -1.20),
            'BR_tigh': (2.36, 2.8), 'BR_knee': (-1.8, -1.20),
            'BL_tigh': (2.36, 2.8), 'BL_knee': (-1.8, -1.20),
        }

        # La posición de pie en MuJoCo no es necesaria para el escalado,
        # pero es bueno tenerla como referencia.
        self.mujoco_stand_pose_rad = {
            'FR_tigh': 2.35619, 'FR_knee': -1.5708,
            'FL_tigh': 2.35619, 'FL_knee': -1.5708,
            'BR_tigh': 2.35619, 'BR_knee': -1.5708,
            'BL_tigh': 2.35619, 'BL_knee': -1.5708,
        }

    @staticmethod
    def _scale_value(value, in_min=-1.0, in_max=1.0, out_min=0.0, out_max=0.0):
        """
        Mapea linealmente un valor de un rango de entrada a un rango de salida.
        """
        # Asegurarse de que el valor de entrada esté dentro del rango [-1, 1]
        value = np.clip(value, in_min, in_max)
        
        # Fórmula de escalado lineal
        scaled_value = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        return scaled_value

    def to_mujoco_radians(self, nn_action_vector):
        """
        Convierte un vector de acción de la red neuronal [-1, 1] a ángulos en
        radianes para la simulación en MuJoCo, usando los límites absolutos.
        """
        mujoco_angles = []
        for i, joint_name in enumerate(self.joint_order):
            nn_value = nn_action_vector[i]
            min_rad, max_rad = self.mujoco_limits_rad[joint_name]
            
            scaled_angle = self._scale_value(nn_value, out_min=min_rad, out_max=max_rad)
            mujoco_angles.append(scaled_angle)
            
        return mujoco_angles

    def to_real_robot_degrees(self, nn_action_vector):
        """
        Convierte un vector de acción de la red neuronal [-1, 1] a ángulos en
        grados para el robot real. La acción se interpreta como un desplazamiento
        (offset) desde la posición de pie.
        """
        real_robot_angles = []
        for i, joint_name in enumerate(self.joint_order):
            nn_value = nn_action_vector[i]
            
            # 1. Calcular el desplazamiento en grados
            # Un valor de 1.0 corresponde al máximo desplazamiento positivo.
            # Un valor de -1.0 corresponde al máximo desplazamiento negativo.
            offset_deg = nn_value * self.real_robot_action_range_deg
            
            # 2. Sumar el desplazamiento a la posición de pie de esa articulación
            stand_angle_deg = self.real_robot_stand_pose_deg[joint_name]
            final_angle_deg = stand_angle_deg + offset_deg
            
            real_robot_angles.append(final_angle_deg)
            
        return real_robot_angles