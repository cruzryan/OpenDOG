from argparse import ArgumentParser
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from environments.WalkEnvironment import WalkEnvironmentV0
from environments.JumpEnvironment import JumpEnvironmentV0
from environments.ScaleActionEnvironment import ScaleActionWrapper
from .VideoRecorder import VideoRecorderCallback
import os	

def create_vec_env(env_id, n_envs, dummy=False, seed=None, monitor_kwargs=None):
    base_env_class = WalkEnvironmentV0 if env_id == "walk" else JumpEnvironmentV0
    
    def make_env_fn():
        env = base_env_class(render_mode="rgb_array")
        env = ScaleActionWrapper(env) 
        if monitor_kwargs: 
            env = Monitor(env, filename=None, info_keywords=monitor_kwargs)
        return env

    if dummy:
        return make_vec_env(
            lambda: ScaleActionWrapper(base_env_class(render_mode="rgb_array")),
			n_envs=1,
            seed=seed,
            vec_env_cls=DummyVecEnv
        )
    
    vec_env = make_vec_env(
        make_env_fn,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv
    )
    return vec_env


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('motion', choices=["walk", "jump"], help="Specify the motion type: 'walk' or 'jump'")
	parser.add_argument('--evaluation_frequency', type=int, default=1000000, help="Frequency of video recording in steps")
	parser.add_argument('--record_frequency', type=int, default=1000000, help="Frequency of video recording in steps")
	parser.add_argument('--duration', type=int, default=10, help="Duration of each video in seconds")
	parser.add_argument('--fps', type=int, default=30, help="Frames per second for the video")
	parser.add_argument('--n_envs', type=int, default=4, help="Number of parallel environments")
	parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
	args = parser.parse_args()

	monitor_info_keywords = ('patterns_matches', 'x_position', 'distance_from_origin')

	vec_env = create_vec_env(args.motion, args.n_envs, seed=args.seed, monitor_kwargs=monitor_info_keywords)
	eval_env = create_vec_env(args.motion, args.n_envs, seed=args.seed)
	record_env = create_vec_env(args.motion, dummy=True ,n_envs=1, seed=args.seed)

	current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

	tensorboard_dir = f"./ppo_robot_tensorboard/{args.motion}_{current_time}/"
	video_output_dir = f"./videos/{args.motion}/"
	model_output_dir = f"./models/{args.motion}/"
	best_model_path = os.path.join(model_output_dir, "best_model")
	os.makedirs(video_output_dir, exist_ok=True)
	os.makedirs(model_output_dir, exist_ok=True)
	os.makedirs(tensorboard_dir, exist_ok=True)

	model = PPO(
		policy = "MlpPolicy",
		verbose = 2,
		env=vec_env, 
		learning_rate=1e-4,              # Más bajo que el default (3e-4)
		n_steps=2048,                    # Muestras por iteración
		batch_size=512,                  # Tamaño de batch
		n_epochs=10,                     # Épocas de optimización por iteración
		gamma=0.99,                      # Factor de descuento
		ent_coef=0.025,                   # Fomenta exploración
		clip_range=0.2,                  # Clipping de políticas
		max_grad_norm=0.5,               # Evita gradientes explosivos
		tensorboard_log=tensorboard_dir  # Monitorización
	)

	video_callback = VideoRecorderCallback(
		environment=record_env,
		save_frequency=int(args.record_frequency),
		save_path=video_output_dir,
		env_id=args.motion,
		fps=args.fps,
		duration=args.duration
	)				

	eval_callback = EvalCallback(
		eval_env=eval_env,
		best_model_save_path=best_model_path,
		eval_freq=int(args.evaluation_frequency),
		n_eval_episodes=20,
		deterministic=True,
		verbose=1
	)

	try:
		model.learn(
			total_timesteps = 30000000,
			callback=[video_callback, eval_callback],
			reset_num_timesteps=False,
			progress_bar=True
		)
		model.save(f"{model_output_dir}/{args.motion}_{model.num_timesteps}")
	except (Exception, KeyboardInterrupt, RuntimeError) as e:
		print("Training interrupted or encountered an error:", e)
	finally:
		if vec_env is not None:
			vec_env.close()
		if eval_env is not None:
			eval_env.close()
		if record_env is not None:
			record_env.close()

	print(f"Videos saved to {video_output_dir}")
	print(f"Best model saved to {best_model_path}")
