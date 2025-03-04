import socket
import struct
import time
import msgpack

UDP_IP = "localhost"
UDP_PORT = 12345

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP

initial_message = struct.pack('i', 0)
sock.sendto(initial_message, (UDP_IP, UDP_PORT))
print("Mensaje inicial enviado al servidor para registrar la dirección del cliente.")

try:
    while True:
        try:
            sock.settimeout(1.0)
            data_recibida, addr = sock.recvfrom(2048)

            unpacked_data_dict = msgpack.unpackb(data_recibida, raw=False)

            timestamp = unpacked_data_dict['timestamp']
            num_qpos = unpacked_data_dict['num_qpos']
            num_qvel = unpacked_data_dict['num_qvel']
            num_act = unpacked_data_dict['num_act']
            qpos_values = unpacked_data_dict['qpos_data']
            qvel_values = unpacked_data_dict['qvel_data']
            ctrl_values = unpacked_data_dict['act_data']
            ground_reaction_forces = unpacked_data_dict['contact_forces_data']
            active_contacts = unpacked_data_dict['active_contacts']

            print(f"Cliente UDP (MessagePack): Datos recibidos - Timestamp: {timestamp:.6f}")
            print(f"Qpos ({num_qpos}): {qpos_values[:3]}")
            print(f"Qvel ({num_qvel}): {qvel_values[:3]}")
            print(f"Act ({num_act}): {ctrl_values[0:]}.")
            print(f"GRF ({num_act}): {ground_reaction_forces[0:]}.")
            print(f"Active contacts ({active_contacts}): {active_contacts}")

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
    sock.close()