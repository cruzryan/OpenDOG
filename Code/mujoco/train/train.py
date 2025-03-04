from argparse import ArgumentParser
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from environments.walk_environment import WalkEnvironmentV0
from environments.jump_environment import JumpEnvironmentV0
from .VideoRecorder import VideoRecorderCallback
import os

config = {
	"policy": 'MlpPolicy',
	"total_timesteps": 30000000,
	"verbose": 1,
	"video_duration": 10  # Duración en segundos
}

def create_vec_env(env_id, n_envs, seed=None):
	"""Crea el entorno vectorizado para entrenamiento."""
	env_class = WalkEnvironmentV0 if env_id == "walk" else JumpEnvironmentV0
	vec_env = make_vec_env(
		env_class,
		env_kwargs={"render_mode": "rgb_array"},
		n_envs=n_envs,
		seed=seed,
		vec_env_cls=SubprocVecEnv
	)
	return vec_env

if __name__ == "__main__":
	# Argumentos del script
	parser = ArgumentParser()
	parser.add_argument('motion', choices=["walk", "jump"], help="Specify the motion type: 'walk' or 'jump'")
	parser.add_argument('--frequency', type=int, default=3000000, help="Frequency of video recording in steps")
	parser.add_argument('--duration', type=int, default=10, help="Duration of each video in seconds")
	parser.add_argument('--fps', type=int, default=30, help="Frames per second for the video")
	parser.add_argument('--n_envs', type=int, default=4, help="Number of parallel environments")
	parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
	args = parser.parse_args()

	# Configuración del entorno
	n_envs = args.n_envs
	env_id = args.motion
	vec_env = create_vec_env(env_id, n_envs, seed=args.seed)

	# Entorno de evaluación (consistente con env_id)
	eval_env_class = WalkEnvironmentV0 if env_id == "walk" else JumpEnvironmentV0
	eval_env = make_vec_env(
		eval_env_class,
		n_envs=1,
		vec_env_cls=DummyVecEnv,
		env_kwargs={"render_mode": "rgb_array"}
	)

	# Directorios de salida
	video_output_dir = f"./videos/{args.motion}/"
	model_output_dir = f"./models/{args.motion}/"
	best_model_path = os.path.join(model_output_dir, "best_model")
	os.makedirs(video_output_dir, exist_ok=True)
	os.makedirs(model_output_dir, exist_ok=True)

	# Inicializa y entrena el modelo
	model = PPO(
		config["policy"], 
		vec_env, 
		tensorboard_log="../logs/",  # Monitorización
		verbose=config["verbose"]
	)

	# Callback para grabación de videos
	video_callback = VideoRecorderCallback(
		save_freq=args.frequency,
		save_path=video_output_dir,
		env=vec_env,
		env_id=args.motion,
		fps=args.fps,
		duration=args.duration
	)

	# Callback para evaluar y guardar el mejor modelo cada 2,000,000 de pasos globales
	eval_freq = 2000000 // n_envs  # Ajuste para múltiples ambientes
	eval_callback = EvalCallback(
		eval_env,
		best_model_save_path=best_model_path,
		eval_freq=eval_freq,
		n_eval_episodes=5,
		deterministic=True,
		verbose=1
	)

	try:
		model.learn(
			total_timesteps=config["total_timesteps"],
			callback=[video_callback, eval_callback],
			reset_num_timesteps=False,
			progress_bar=True
		)
		# Guarda el modelo final
		model.save(f"{model_output_dir}/{args.motion}_{model.num_timesteps}")
	except (Exception, KeyboardInterrupt, RuntimeError) as e:
		print("Training interrupted or encountered an error:", e)
	finally:
		if vec_env is not None:
			vec_env.close()
		if eval_env is not None:
			eval_env.close()

	print(f"Videos saved to {video_output_dir}")
	print(f"Best model saved to {best_model_path}")