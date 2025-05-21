import os
import imageio
from stable_baselines3.common.callbacks import BaseCallback
from environments.ScaleActionEnvironment import ScaleActionWrapper
from environments.WalkEnvironment import WalkEnvironmentV0
from environments.JumpEnvironment import JumpEnvironmentV0

class VideoRecorderCallback(BaseCallback):

	def __init__(
			self,
			save_frequency,
			save_path,
			environment,
			env_id, 
			fps=30, duration=10, verbose=0
			):
		
		super(VideoRecorderCallback, self).__init__(verbose)
		self.save_frequency = save_frequency
		self.save_path = save_path
		self.vec_env = environment
		self.env_id = env_id
		self.fps = fps
		self.duration = duration
		self.video_env = None

	def _on_step(self):
		if self.num_timesteps % self.save_frequency == 0:
			video_file = os.path.join(
				self.save_path,
				f"video_{self.num_timesteps}.mp4"
			)
			print(f"Grabbing video at step {self.num_timesteps}")
			self.record_video(video_file)
		return True

	def record_video(self, filename):
		# reseteamos el único sub-entorno
		obs = self.vec_env.reset()
		frames = []
		for i in range(self.fps * self.duration):
			# PREDICE directamente sobre vec_env
			action, _ = self.model.predict(obs, deterministic=True)
			# STEP también en vec_env
			obs, _, done, _ = self.vec_env.step(action[0] if action.ndim>1 else action)
			frame = self.vec_env.render()
			if frame is not None:
				frames.append(frame)
			if done:
				obs = self.vec_env.reset()

		imageio.mimwrite(filename, frames, fps=self.fps)
		print(f"Video saved to {filename}")

		print(f"Finished frame recording loop, frames collected: {len(frames)}")

		print(f"Calling imageio.mimwrite with {len(frames)} frames, filename: {filename}")
		try: # Añadido bloque try-except para robustez
			imageio.mimwrite(filename, frames, fps=self.fps)
			print(f"Video saved successfully to: {filename}")
		except Exception as e:
			print(f"ERROR writing video file: {e}")