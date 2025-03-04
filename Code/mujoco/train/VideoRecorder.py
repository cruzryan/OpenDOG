from stable_baselines3.common.callbacks import BaseCallback

import imageio
import os
from environments.walk_environment import WalkEnvironmentV0  # Asegúrate de tener estas importaciones correctas para tu proyecto
from environments.jump_environment import JumpEnvironmentV0  # Asegúrate de tener estas importaciones correctas para tu proyecto

class VideoRecorderCallback(BaseCallback):
    """
    Callback para grabar videos a intervalos de tiempo definidos.
    """
    def __init__(self, save_freq, save_path, env, env_id, fps=30, duration=10, verbose=0): # Añadir env_id
        super(VideoRecorderCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.vec_env = env  # Renombrar self.env a self.vec_env para claridad (es el VecEnv)
        self.env_id = env_id  # Almacenar env_id para crear el entorno de video correcto
        self.fps = fps
        self.duration = duration
        self.video_env = None # Inicializar video_env como None, se creará de forma lazy en record_video

    def _on_step(self):
        """
        Método ejecutado en cada paso del entrenamiento.
        """
        if self.num_timesteps % self.save_freq == 0:
            video_file = os.path.join(
                self.save_path,
                f"video_{self.num_timesteps}.mp4"
            )
            print(f"Grabbing video at step {self.num_timesteps}")
            self.record_video(video_file)
        return True

    def record_video(self, filename):
        """
        Graba un video del entorno.
        """
        frames = []
        if self.video_env is None: # Inicializar entorno individual SOLO si no se ha hecho aún
            print("Initializing self.video_env for recording")
            if self.env_id == "walk":
                self.video_env = WalkEnvironmentV0(render_mode="rgb_array") # Crear entorno INDIVIDUAL con render_mode="human"
            elif self.env_id == "jump":
                self.video_env = JumpEnvironmentV0(render_mode="rgb_array") # Crear entorno INDIVIDUAL con render_mode="human"
            else:
                raise ValueError(f"Unknown env_id: {self.env_id}")

        obs = self.video_env.reset() # Resetear el entorno INDIVIDUAL de grabación (modo humano)
        vec_obs = self.vec_env.reset() # Resetear el VecEnv para obtener la observación inicial del VecEnv para la predicción

        print("Starting frame recording loop")
        for i in range(self.fps * self.duration):
            action, _ = self.model.predict(vec_obs, deterministic=True) # Predecir usando la observación del VecEnv
            vec_obs, _, done, _ = self.vec_env.step(action) # Ejecutar paso en el VecEnv para avanzar el entrenamiento
            obs, _, _, _, _ = self.video_env.step(action[0]) # Ejecutar paso en el entorno INDIVIDUAL de grabación con la acción para UN entorno (action[0]) - IMPORTANTE: usar action[0]
            frame = self.video_env.render() # Renderizar el entorno INDIVIDUAL en modo rgb_array (para obtener el frame)
            if frame is None: # Comprobar si el frame es None
                print(f"WARNING: frame is None at step {i} during recording")
                continue # Saltar frames None

            frames.append(frame)
            if done.any():  # Si algún sub-entorno del VecEnv termina, resetear también el entorno de grabación para consistencia. Aunque los 'done' individuales podrían no estar alineados.
                self.video_env.reset()
                vec_obs = self.vec_env.reset() # Resetear también el vec_env, aunque el reset del vec_env ya ocurre en el bucle de entrenamiento.

        print(f"Finished frame recording loop, frames collected: {len(frames)}")

        print(f"Calling imageio.mimwrite with {len(frames)} frames, filename: {filename}")
        try: # Añadido bloque try-except para robustez
            imageio.mimwrite(filename, frames, fps=self.fps)
            print(f"Video saved successfully to: {filename}")
        except Exception as e:
            print(f"ERROR writing video file: {e}")