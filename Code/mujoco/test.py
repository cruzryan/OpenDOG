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

	for _ in range(1000):
		action, _states = model.predict(obs, deterministic=False)

		print(f"Action: {action}")
		print(f"States: {_states}")
		
		obs, reward, terminated, truncated, info = env.step(action)
		print("obs", obs)
		print("reward", reward)
		print("terminated", terminated)
		print("truncated", truncated)
		# print("info", info)
		print("patterns", info['patterns_matches'])
		FR_force = np.vstack((FR_force, info['paw_contact_forces'][7][:3]))
		FL_force = np.vstack((FL_force, info['paw_contact_forces'][4][:3]))
		BR_force = np.vstack((BR_force, info['paw_contact_forces'][13][:3]))
		BL_force = np.vstack((BL_force, info['paw_contact_forces'][10][:3]))

		print("Front rigth force", info['paw_contact_forces'][7][:3] )
		print("Front left force", info['paw_contact_forces'][4][:3] )
		print("Back rigth force", info['paw_contact_forces'][13][:3] )
		print("Back left force", info['paw_contact_forces'][10][:3] )

	env.close()

	header = "FR_x,FR_y,FR_z,FL_x,FL_y,FL_z,BR_x,BR_y,BR_z,BL_x,BL_y,BL_z"
	combined = np.hstack((FR_force, FL_force, BR_force, BL_force))
	np.savetxt('fuerzas_patas.csv', combined, delimiter=',', header=header)

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

	data = np.loadtxt('fuerzas_patas.csv', delimiter=',', skiprows=1)

	FR_z = data[:, 2]   # Columna 2: Fuerza Z pata delantera derecha (FR)
	FL_z = data[:, 5]   # Columna 5: Fuerza Z pata delantera izquierda (FL)
	BR_z = data[:, 8]   # Columna 8: Fuerza Z pata trasera derecha (BR)
	BL_z = data[:, 11]  # Columna 11: Fuerza Z pata trasera izquierda (BL)

	# 3. Crear el gráfico
	plt.style.use('Solarize_Light2')
	plt.figure(figsize=(12, 6))

	# Graficar cada pata con estilo diferente
	plt.scatter(np.arange(len(FR_z)), FR_z, label='FR (Front Right)', color='red', marker='o', s=5)  # 'o' = círculo
	plt.scatter(np.arange(len(FL_z)), FL_z, label='FL (Front Left)', color='blue', marker='s', s=5)   # 's' = cuadrado
	plt.scatter(np.arange(len(BR_z)), BR_z, label='BR (Back Right)', color='green', marker='^', s=5)  # '^' = triángulo
	plt.scatter(np.arange(len(BL_z)), BL_z, label='BL (Back Left)', color='purple', marker='D', s=4)  # 'D' = diamante

	# Personalizar el gráfico
	plt.title('Fuerza Vertical (Z) en las Patas', fontsize=14)
	plt.xlabel('Muestras', fontsize=12)
	plt.ylabel('Fuerza Z (N)', fontsize=12)
	plt.legend(fontsize=10, loc='upper right')  # Leyenda con ubicación personalizada

	# Mostrar el gráfico
	plt.show()