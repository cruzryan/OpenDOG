import matplotlib
matplotlib.use('Agg')

import os
import imageio
import numpy as np

import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure

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
		obs = self.vec_env.reset()
		pattern_data = []
		frames = []
		for i in range(self.fps * self.duration):
			action, _ = self.model.predict(obs, deterministic=False)
			obs, _, done, info = self.vec_env.step(action)
			pattern_data.append(info[0]['patterns_matches'])
			frame = self.vec_env.render()
			if frame is not None:
				frames.append(frame)
			if done.any():
				obs = self.vec_env.reset()

		try:
			if pattern_data:
				fig = plt.figure()
				ax = fig.add_subplot(1, 1, 1) # Es buena práctica obtener el objeto Axes
				ax.plot(pattern_data)
				ax.set_title(f"Patterns Matches Over Episode (Step {self.num_timesteps})")
				ax.set_xlabel("Time within episode")
				ax.set_ylabel("Patterns Matches Value")

				self.logger.record(f"trajectory/patterns_plot_{self.num_timesteps}", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
				plt.close(fig) # Asegúrate de cerrar la figura correcta
				print(f"Patterns plot logged to TensorBoard for step {self.num_timesteps}")
			else:
				print("Warning: No patterns_data collected to plot.")

			plt.close()
			imageio.mimwrite(filename, frames, fps=self.fps)
			print(f"Video saved successfully to: {filename}")
		except Exception as e:
			print(f"ERROR writing video file: {e}")

