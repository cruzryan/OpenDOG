import socket

from jax import numpy as jp
import jax
import numpy as np
import threading
import time
import struct
import msgpack
import mujoco
import mujoco._functions
import mujoco._simulate
from mujoco import mjx
from mujoco.mjx._src.types import ConeType
import mujoco.plugin
from rewards.walk_environment_reward_calc import WalkEnvironmentRewardCalc


class mujoco_communication:
    def __init__(self, simulate_instance, sending_frequency=30):
        print("Servidor de movilidad inicializado")        
        self.walk_rewards = None
        self._simulate_instance = simulate_instance
        self.socket_thread = threading.Thread(target=self._socket_server_loop)
        self.socket_thread.daemon = True
        self.sending_frequency = sending_frequency
        self.sleep_time = 1.0 / sending_frequency if sending_frequency > 0 else 0.01
        self.client_address = None  # Almacena la dirección del cliente

    def start_server(self):
        self.socket_thread.start()
        print("Servidor de movilidad (UDP) para envío continuo escuchando en localhost:12345")

    def stop_server(self):
        self._simulate_instance.exitrequest = True
        if self.socket_thread.is_alive():
            self.socket_thread.join()
        print("Servidor de movilidad (UDP) cerrado")

    def _socket_server_loop(self):
        """
        Bucle principal del servidor de sockets UDP para enviar datos continuamente.
        """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = ('localhost', 12345)
        server_socket.bind(server_address)

        print(f"Servidor de movilidad (UDP) para envío continuo escuchando en {server_address[0]}:{server_address[1]}")

        while not self._simulate_instance.exitrequest:
            try:
                server_socket.settimeout(0.1)
                try:
                    data, addr = server_socket.recvfrom(1024)
                    if self.client_address is None:
                        self.client_address = addr
                        print(f"Dirección del cliente UDP registrada: {self.client_address}")
                        print("Cliente conectado")
                    command = struct.unpack('i', data)[0]
                    print(f"Comando UDP desconocido: {command}")
                except socket.timeout:
                    pass

                if self.client_address:
                    self._send_simulation_data_udp(server_socket, self.client_address)


                time.sleep(self.sleep_time)

            except Exception as e_server_loop:
                print(f"Error en el bucle principal del socket server (UDP): {e_server_loop}")
                break

        server_socket.close()
        print("Servidor de movilidad (UDP) cerrado")


    def get_joint_indexes(self):
            body_type = mujoco.mjtObj.mjOBJ_BODY
            body_names = [ 'trunk', 
                          'FL_tigh', 'FL_calf', 'FL_paw', 
                          'FR_tigh', 'FR_calf', 'FR_paw', 
                          'BL_tigh', 'BL_calf', 'BL_paw', 
                          'BR_tigh', 'BR_calf', 'BR_paw'
            ]
            body_indices = []

            for joint_name in body_names:
                    joint_index = mujoco.mj_name2id(self._simulate_instance.m, body_type, joint_name)
                    body_indices.append((joint_name, joint_index))
            
            print(body_indices)
            return body_indices

    def _get_simulation_data(self):
        d = self._simulate_instance.d
        m = self._simulate_instance.m
        qpos_data = d.qpos[0:3].copy()
        qvel_data = d.qvel[0:3].flat.copy()
        control_data = d.ctrl.flat.copy()
        timestamp = time.time()
        self.walk_rewards.diagonal_gait_reward(d, m)
        paw_contact_forces = self.walk_rewards.get_paw_contact_forces(d, m)
        num_qpos = len(m.qpos0)
        num_qvel = m.nv
        num_act = m.na

        return {
            'timestamp': timestamp,
            'num_qpos': num_qpos,
            'num_qvel': num_qvel,
            'num_act': num_act,
            'qpos_data': qpos_data.tolist(),
            'qvel_data': qvel_data.tolist(),
            'ctr_data': control_data.tolist(),
            'contact_forces_data': np.concatenate(list(paw_contact_forces.values())).tolist(),
            'active_contacts' : d.ncon
        }

    def _send_simulation_data_udp(self, server_socket, client_address):
        d = self._simulate_instance.d
        if d is not None: 
            self.walk_rewards = WalkEnvironmentRewardCalc(gravity=self._simulate_instance.m.opt.gravity,
                            default_joint_position=self._simulate_instance.m.key_ctrl[0],
                            actuator_range=self._simulate_instance.m.actuator_ctrlrange)

            with self._simulate_instance.lock():  # Critic area protection
                data_dict = self._get_simulation_data()
                data_to_send = msgpack.packb(data_dict)
                try:
                    server_socket.sendto(data_to_send, client_address)
                except Exception as e:
                    print(f"Error al enviar datos de simulación por UDP (MessagePack): {e.with_traceback}")
        else:
            print("Error: Datos de MuJoCo no disponibles, no se pueden enviar datos de sensores.")