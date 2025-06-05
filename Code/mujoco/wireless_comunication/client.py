import socket
import struct
import time
import msgpack
import numpy as np
import matplotlib.pyplot as plt
import math

class client_mujoco_communication:
	def __init__(self):
		self.UDP_IP = "localhost"
		self.UDP_PORT = 12345
		initial_message = struct.pack('i', 0)

		self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.sock.sendto(initial_message, (self.UDP_IP, self.UDP_PORT))
		print("Mensaje inicial")

		plt.ion()
		self.set_plot_paws_forces()


	def recieve_info(self):
		try:
			while True:
				try:
					self.sock.settimeout(1.0)
					data_recibida, addr = self.sock.recvfrom(2048)

					unpacked_data_dict = msgpack.unpackb(data_recibida, raw=False)

					timestamp = unpacked_data_dict['timestamp']
					num_qpos = unpacked_data_dict['num_qpos']
					num_qvel = unpacked_data_dict['num_qvel']
					num_act = unpacked_data_dict['num_act']
					qpos_values = unpacked_data_dict['qpos_data']
					qvel_values = unpacked_data_dict['qvel_data']
					ctrl_values = unpacked_data_dict['ctr_data']
					ground_reaction_forces = unpacked_data_dict['contact_forces_data']
					active_contacts = unpacked_data_dict['active_contacts']
					#print(ground_reaction_forces)
					"""
					print(f"Cliente UDP (MessagePack): Datos recibidos - Timestamp: {timestamp:.6f}")
					print(f"Posición del tronco ({num_qpos}): {qpos_values[:3]}")
					print(f"Velocidad del tronco ({num_qvel}): {qvel_values[:3]}")
					print(f"Posición de actuadores en radianes ({num_act}): {ctrl_values[:]}.")
					print(f"Fuerzas de reacción ({num_act}): {ground_reaction_forces[:]}.")
					print(f"Active contacts ({active_contacts}): {active_contacts}")
					"""

					self.update_plot_paws_force(ground_reaction_forces)

				except socket.timeout:
					print("Timeout al recibir datos.")
				except msgpack.UnpackException as e: # Capturar excepción específica de msgpack
					print(f"Error al desempaquetar datos msgpack: {e}")
				except Exception as e_client_loop:
					print(f"Error en el bucle del cliente UDP (MessagePack): {e_client_loop}")

				time.sleep(0.001) # No consumir CPU en bucle apretado

		except KeyboardInterrupt:
			print("Cliente UDP cerrado por interrupción del usuario.")
		finally:
			self.sock.close()
	
	def set_plot_paws_forces(self):
		plt.style.use('fast')

		self.x_plot = 0.5 + np.arange(12)
		y = np.zeros(12)

		figure_paw_forces, ax = plt.subplots()

		bar_colors = ['blue', 'green', 'red'] * 4 
		self.bars = ax.bar(self.x_plot, y, width=1, color=bar_colors)

		ax.set(	
			xlim=(0, 13), xticks=np.arange(1, 13), 
			ylim=(-10, 10), yticks=np.arange(-10, 10)
			)
		ax.grid(axis='y', linestyle='--', alpha=0.7)
		ax.grid(axis='x', linestyle='--', alpha=0.7)

		patches = [plt.Rectangle((0, 0), 1, 1, color=bar_colors[i*3]) for i in range(4)] # Create a rectangle patch for each color group
		labels = ['Paw 1 Forces', 'Paw 2 Forces', 'Paw 3 Forces', 'Paw 4 Forces'] # Labels for each group

		ax.legend(patches, labels) # Add legend to the axes

		self.figure_paw_forces = figure_paw_forces
		self.ax = ax
	
	def update_plot_paws_force(self, ground_reaction_forces):
		for i, bar in enumerate(self.bars.patches):
			bar.set_height(ground_reaction_forces[math.floor(i/3)*6+(i%3)])

		self.ax.relim() # Recalculate data limits
		self.ax.autoscale_view() # Autoscale axes to fit data

		self.figure_paw_forces.canvas.flush_events() # Process pending GUI events, necessary for real-time update

if __name__ == "__main__":
	client = client_mujoco_communication()
	client.recieve_info()
