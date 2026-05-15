import socket
import struct
import json
import threading
import numpy as np
import base64
import cv2
import time


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_frame(sock):
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack("!I", header)
    return _recv_exact(sock, length)


def send_frame(sock, body):
    sock.sendall(struct.pack("!I", len(body)) + body)


class ClientCom():

    def __init__(self):
        self.server_ip = socket.gethostbyname(socket.gethostname())
        self.server_port = 5000  # Port for communication
        self.lock = threading.Lock()

        self.camera_frame_arrived = False

        self.camera_frame_rgb = None
        self.camera_frame_depth = None
        self.current_position = None
        self.current_yaw_degree = None

        self.load_prefix()

        self.client_main()

    def load_prefix(self):
        self.stop_message_config = "stop"
        self.go_forward_config = "forward"
        self.turn_left_config = "left"
        self.turn_right_config = "right"

        self.stream_enable_config = "stream_enable"
        self.stream_disable_config = "stream_disable"
    
    def get_current_sensor_frame(self):

        while not self.camera_frame_arrived:
            time.sleep(0.01)

        return self.camera_frame_rgb, self.camera_frame_depth, self.current_position, self.current_yaw_degree
        
    def receive_messages(self):
        """Continuously receive messages from the server."""

        frame_count = 0
        while True:
            try:
                body = recv_frame(self.client_socket)

                if body is None:
                    print("[ComClient] connection closed by server.")
                    break

                frame_count += 1
                if frame_count <= 3 or frame_count % 50 == 0:
                    print(f"[ComClient] frame #{frame_count} received ({len(body)} bytes)")

                try:
                    message = json.loads(body.decode('utf-8'))
                except json.JSONDecodeError:
                    print(f"[ComClient] bad JSON frame ({len(body)} bytes), skipping")
                    continue  # skip to next loop
                
                # Decode base64 frame
                raw_frame_rgb = message["message"]["rgbcamera_frame"]
                raw_frame_depth = message["message"]["depthcamera_frame"]

                if (raw_frame_rgb is not None) and (raw_frame_depth is not None):
                    frame_data_rgb = base64.b64decode(raw_frame_rgb)
                    np_arr_rgb = np.frombuffer(frame_data_rgb, np.uint8)
                    self.camera_frame_rgb = cv2.imdecode(np_arr_rgb, cv2.IMREAD_COLOR)

                    frame_data_depth = base64.b64decode(raw_frame_depth)
                    np_arr_depth = np.frombuffer(frame_data_depth, np.uint8)
                    self.camera_frame_depth = cv2.imdecode(np_arr_depth, cv2.IMREAD_COLOR)

                    current_pos_x = message["message"]["sensor_data"]["posx"]
                    current_pos_y = message["message"]["sensor_data"]["posy"]
                    current_pos_z = message["message"]["sensor_data"]["posz"]
                    current_yaw = message["message"]["sensor_data"]["yaw"]

                    # self.current_position = np.array([current_pos_x, current_pos_y, current_pos_z])
                    self.current_position = np.array([current_pos_x, current_pos_y])
                    self.current_yaw_degree = current_yaw

                    self.camera_frame_arrived = True
            
            except (ConnectionResetError, OSError):
                print("trying again")
                break

        try:
            self.client_socket.close()
        except OSError:
            pass

    def client_main(self):
        self.client_name = "Com Client"

        print(f"[ComClient] connecting to {self.server_ip}:{self.server_port}")
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_ip, self.server_port))
        print(f"[ComClient] connected, sending handshake name={self.client_name!r}")

        send_frame(self.client_socket, self.client_name.encode())  # Send name as the first framed message

        # Start a thread to receive messages
        threading.Thread(target=self.receive_messages, daemon=True).start()
        print("[ComClient] receive thread started, sleeping 5s")

        time.sleep(5)
        print("[ComClient] client_main done")

    def send_request_sensor_stream(self, command):
        to_client = "Pioneer 3AT"
        command_text = command

        client_message = {
            "From": self.client_name,
            "To": to_client,
            "command": command_text
        }

        send_frame(self.client_socket, json.dumps(client_message).encode())

# com_client_ = ClientCom()

# while True:
#     com_client_.send_request_sensor_stream("stream_enable")
#     time.sleep(0.1)
#     com_client_.send_request_sensor_stream("right")

#     while True:
#         camera_rgb, camera_depth, current_position, current_yaw_degree = com_client_.get_current_sensor_frame()
#         cv2.imshow("Camera RGB", camera_rgb)

#         # Normalize depth values to 0–255 for display
#         normalized_depth = cv2.normalize(camera_depth, None, 0, 255, cv2.NORM_MINMAX)
#         depth_uint8 = normalized_depth.astype(np.uint8)

#         # Optional: apply a colormap for better visualization
#         colored_depth = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

#         # Show the depth image
#         cv2.imshow("Depth Frame", colored_depth)

#         # Wait for 1ms for UI update, break if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
