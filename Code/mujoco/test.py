from stable_baselines3 import PPO
from environments.walk_environment import WalkEnvironmentV0
from environments.jump_environment import JumpEnvironmentV0
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

def test_model(env, motion):
	# Cargar el modelo correspondiente al tipo de movimiento
	model_path = f"/home/mau/Documentos/Escuela/TT/OpenDOG/Code/mujoco/models/{motion}/best_model/best_model"
	model = PPO.load(model_path)
	print (model.policy.observation_space)
	obs, _ = env.reset()
	FR_force = np.empty((0, 3))
	FL_force = np.empty((0, 3))
	BR_force = np.empty((0, 3))
	BL_force = np.empty((0, 3))

	ground_force = [3, 0, 5]
	z_curve = ground_force[2]* np.sin(np.linspace(0, np.pi, 50))
	x_curve = ground_force[0]* np.sin(np.linspace(0, 2*np.pi, 50))
	samples = np.zeros((4, 50), dtype=float)
	j = 0
	
	for i in range(9000):
		action, _states = model.predict(obs, deterministic=False)

		print(f"Action: {action}")
		print(f"States: {_states}")
		
		obs, reward, terminated, truncated, info = env.step(action)
		#print("obs", obs)
		#print("reward", reward)
		#print("terminated", terminated)
		#print("truncated", truncated)
		#print("info", info)
		print("patterns", info['paw_contact_forces'][4][2])

		np.put(samples[0], j, info['paw_contact_forces'][4][2]) 
		np.put(samples[1], j, info['paw_contact_forces'][7][2]) 
		np.put(samples[2], j, info['paw_contact_forces'][10][2]) 
		np.put(samples[3], j, info['paw_contact_forces'][13][2]) 

		j = (j + 1) % 50

		if j == 0:
			print (samples)
			print ("MSE fuerza pata delantera izquierda", np.square(samples[0] - z_curve).mean())
			print ("MSE fuerza pata delantera derecha", np.square(samples[1] - z_curve).mean())
			print ("MSE fuerza pata trasera izquierda", np.square(samples[2] - z_curve).mean())
			print ("MSE fuerza pata trasera derecha", np.square(samples[3] - z_curve).mean())

	env.close()

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('motion', choices=["walk", "jump"], help="Specify the motion type: 'walk' or 'jump'")
	args = parser.parse_args()

	# Elegir el entorno según el tipo de movimiento
	if args.motion == "walk":
		env = WalkEnvironmentV0(render_mode="human")
	elif args.motion == "jump":
		env = JumpEnvironmentV0(render_mode="human")

	# Llamar a la función de prueba con el entorno y tipo de movimiento especificado
	test_model(env, args.motion)