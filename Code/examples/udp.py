import socket
import json
import time

def test_udp(ip, port=12345, retries=3, timeout=2.0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)  # Increase timeout to 2 seconds
    # Bind the socket to port 12345 to receive responses
    try:
        sock.bind(('0.0.0.0', port))
    except Exception as e:
        print(f"Failed to bind socket to port {port}: {e}")
        sock.close()
        return False
    
    command = {"command": "reset_all"}
    attempt = 0
    
    while attempt < retries:
        try:
            print(f"Sending to {ip}:{port} (attempt {attempt + 1}/{retries}): {command}")
            sock.sendto(json.dumps(command).encode('utf-8'), (ip, port))
            
            # Keep receiving packets until we get the correct response or timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    data, addr = sock.recvfrom(1024)
                    response = json.loads(data.decode('utf-8'))
                    print(f"Received from {addr}: {response}")
                    # Check if this is the command response
                    if "status" in response and response["status"] == "OK":
                        sock.close()
                        return True
                    # If it's an angle update, keep looping
                    else:
                        print(f"Ignoring angle update packet from {addr}")
                except socket.timeout:
                    break  # Timeout occurred, break inner loop and retry
                except json.JSONDecodeError:
                    print(f"Received invalid JSON from {addr}, ignoring")
        
            print(f"Timeout (attempt {attempt + 1}/{retries})")
            attempt += 1
            if attempt == retries:
                print("All attempts failed")
                sock.close()
                return False
            time.sleep(0.1)
        except Exception as e:
            print(f"Error: {e}")
            sock.close()
            return False

if __name__ == "__main__":
    test_udp("192.168.137.100")
    test_udp("192.168.137.101")