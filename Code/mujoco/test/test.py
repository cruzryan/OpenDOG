import numpy as np
from argparse import ArgumentParser

from stable_baselines3 import PPO
from environments.ScaleActionEnvironment import ScaleActionWrapper
from environments.WalkEnvironment import WalkEnvironmentV0
from environments.JumpEnvironment import JumpEnvironmentV0

from .RealTimePlotter import RealTimePlotter
from .ScaleActions import ScaleActions

def test_model(env, motion):
	scaler = ScaleActions()
	model_path = f"/home/mau/Documentos/Escuela/TT/OpenDOG/Code/mujoco/models/{motion}/best_model/best_model"
	model = PPO.load(model_path)
	print(model.policy.observation_space)
	
	plotter = RealTimePlotter()
	obs, _ = env.reset()
	
	for i in range(5000):
		action, _states = model.predict(obs, deterministic=False)
		obs, reward, terminated, truncated, info = env.step(action)
		
		mujoco_angles = scaler.to_mujoco_radians(action)
		print(f"Grados en mujoco [...]:\n{np.round(mujoco_angles, 4)}\n")

		real_angles = scaler.to_real_robot_degrees(action)
		print(f"Grados en robot real [...]:\n{np.round(real_angles, 2)}\n")

		plotter.update_plot([
			info['paw_contact_forces'][4][2],
			info['paw_contact_forces'][7][2],
			info['paw_contact_forces'][10][2],
			info['paw_contact_forces'][13][2],
		])

			
	print("Simulación terminada.")
	
	env.close()
	
	plotter.close()

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('motion', choices=["walk", "jump"], help="Specify the motion type: 'walk' or 'jump'")
	args = parser.parse_args()

	if args.motion == "walk":
		env = WalkEnvironmentV0(render_mode="human")
	elif args.motion == "jump":
		env = JumpEnvironmentV0(render_mode="human")

	wrapped = ScaleActionWrapper(env)
	test_model(wrapped, args.motion)