import socket

import jax
from jax import numpy as jp
import threading
import time
import struct
import msgpack
import mujoco
import mujoco._functions
import mujoco._simulate
from mujoco import mjx
from mujoco.mjx._src.types import ConeType
import numpy as np

import mujoco.plugin

class mujoco_communication:
    def __init__(self, simulate_instance, sending_frequency=30):
        """
        Inicializa el servidor de movilidad.
        """
        print("Servidor de movilidad inicializado")
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

    def detect_paws(self):
        d = self._simulate_instance.d
        m = self._simulate_instance.m
        paws_in_ground = [False, False, False, False]
        contacts = np.zeros((3,6))
        for i in (0,4):
            mujoco._functions.mj_contactForce(m, d, 1, contacts[i])

        return []
    
    """
    """
    def paws_in_ground(self):
        d = self._simulate_instance.d
        m = self._simulate_instance.m
        paws_body_indexes = [4, 7, 10, 13] 
        ground_body_index = 0
        paws_contact_id = [-1,-1,-1,-1]
        
        paw_index_to_contact_index = {} 

        for contact_index in range(d.ncon): 
            contact = d.contact[contact_index] 
            
            geom1 = contact.geom1
            geom2 = contact.geom2

            for paw_index_enum, paw_body_index in enumerate(paws_body_indexes):
                if (geom1 == ground_body_index and m.geom_bodyid[geom2] == paw_body_index) or \
                   (geom2 == ground_body_index and m.geom_bodyid[geom1] == paw_body_index):
                    paws_contact_id[paw_index_enum] = contact_index
                    paw_index_to_contact_index[paw_body_index] = contact_index
                    break

        return paws_contact_id, paw_index_to_contact_index


    def get_paw_contact_forces(self):
        d = self._simulate_instance.d
        m = self._simulate_instance.m
        paws_body_indexes = [4, 7, 10, 13]
        paw_contact_forces = {paw_body_index: np.zeros(6) for paw_body_index in paws_body_indexes}
        paw_contact_indices, paw_index_to_contact_index = self.paws_in_ground()

        for paw_body_index in paws_body_indexes:
            contact_index = paw_index_to_contact_index.get(paw_body_index, -1)

            if contact_index != -1:
                force_array = np.zeros(6)
                mujoco._functions.mj_contactForce(m, d, contact_index, force_array)
                paw_contact_forces[paw_body_index] = force_array
            else:
                paw_contact_forces[paw_body_index] = np.zeros(6)

        return paw_contact_forces

    def get_reward_ground_reaction_force(
            self,
            data,
            model):
        forces = self.get_paw_contact_forces()
        reward = 0
        for i in [4, 7, 10, 13]:
            reward = reward + forces[i][0]
        return reward

    def _get_simulation_data(self):
        d = self._simulate_instance.d
        m = self._simulate_instance.m
        qpos_data = d.qpos[0:3].copy()
        qvel_data = d.qvel[0:3].flat.copy()
        control_data = d.ctrl.flat.copy()
        timestamp = time.time()

        paw_contact_forces = self.get_paw_contact_forces()

        print(paw_contact_forces[4][0])
        print(self.get_reward_ground_reaction_force(d,m))

        contact_forces_data = d.qfrc_applied[:]
        

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
            'act_data': control_data.tolist(),
            'contact_forces_data': contact_forces_data.tolist(),
            'active_contacts' : d.ncon
        }

    def _send_simulation_data_udp(self, server_socket, client_address):
        d = self._simulate_instance.d
        if d is not None: 
            with self._simulate_instance.lock():  # Critic area protection
                data_dict = self._get_simulation_data()
                data_to_send = msgpack.packb(data_dict)
                try:
                    server_socket.sendto(data_to_send, client_address)
                except Exception as e:
                    print(f"Error al enviar datos de simulación por UDP (MessagePack): {e}")
        else:
            print("Error: Datos de MuJoCo no disponibles, no se pueden enviar datos de sensores.")