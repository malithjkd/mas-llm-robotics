from controller import Supervisor
import pygame
import math
import socket
import struct
import json
import threading
import numpy as np
import cv2
import base64
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


class PIONEER_3AT():
    
    def __init__(self):
        # define essential parameters
        self.timestep = 64
        self.position_thresh = np.array([0, 0, 0])
        self.yaw_degree_thresh = 90
        self.lock = threading.Lock()
        self.sensor_stream_enable = False
        
        self.current_position = None
        self.current_yaw = None
        self.current_rgb_frame = None
        self.current_depth_frame = None
        
        # Set the target velocity (rad/s)
        self.target_speed = 4
        
        # define the robot
        self.robot = Supervisor()
        
        self.robot_node = self.robot.getFromDef("PIONEER_3AT") 
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")
        
        # define the cameras
        self.camera_rgb = self.robot.getDevice('CAM_RGB')
        self.camera_rgb.enable(self.timestep)
        self.camera_depth = self.robot.getDevice('CAM_DEPTH')
        self.camera_depth.enable(self.timestep)
        
        # define robot wheel motors
        self.front_left_motor = self.robot.getDevice('front left wheel')
        self.front_right_motor = self.robot.getDevice('front right wheel')
        self.back_left_motor = self.robot.getDevice('back left wheel')
        self.back_right_motor = self.robot.getDevice('back right wheel')
        
        # Set the motor positions and velocities
        self.front_left_motor.setPosition(float('inf')) 
        self.front_right_motor.setPosition(float('inf')) 
        self.back_left_motor.setPosition(float('inf')) 
        self.back_right_motor.setPosition(float('inf')) 
        
        self.front_left_motor.setVelocity(0)
        self.front_right_motor.setVelocity(0)
        self.back_left_motor.setVelocity(0)
        self.back_right_motor.setVelocity(0)
        
        # initialise the client communication
        self.ClientCom_init()
        
        # incoming message command types
        self.stop_message_config = "stop"
        self.go_forward_config = "forward"
        self.turn_left_config = "left"
        self.turn_right_config = "right"
        
        self.sensor_stream_enable_config = "stream_enable"
        self.sensor_stream_disable_config = "stream_disable"
        
        # outgoing and incoming message format
        self.outgoing_message = {
                            "From": self.client_name, 
                            "To"  : "Com Client", 
                            "message": {
                                        "rgbcamera_frame": None, 
                                        "depthcamera_frame": None, 
                                        "sensor_data": {
                                                        "posx": None, 
                                                        "posy": None, 
                                                        "posz": None, 
                                                        "yaw": None
                                                        }
                                        }
                               }
        
    def ClientCom_init(self):
        self.client_name = "Pioneer 3AT"
         
        # server_ip = socket.gethostbyname(socket.gethostname())
        server_ip = "172.28.193.38"
        server_port = 5000  # Port for communication
    
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((server_ip, server_port))
        
        # Send name as the first framed message
        send_frame(self.client_socket, self.client_name.encode())
        
    def receive_messages(self):
        """Continuously receive messages from the server."""
        
        while self.robot.step(self.timestep) != -1:
            try:
                body = recv_frame(self.client_socket)

                if body is None:
                    print("[Pioneer] connection closed by server")
                    break

                # Parse the received JSON message
                try:
                    message_data = json.loads(body.decode('utf-8'))
                except json.JSONDecodeError:
                    continue  # skip to next loop
                    
                incoming_msg_command = message_data.get("command")
                
                # check the commands to the robot
                if incoming_msg_command == self.sensor_stream_enable_config:
                    print("Sensor stream enable. Starting stream .....")
                    self.sensor_stream_enable = True
                    
                elif incoming_msg_command == self.sensor_stream_disable_config:
                    print("Sensor stream disabled.")
                    self.sensor_stream_enable = False
 
                elif incoming_msg_command == self.stop_message_config:
                    print("Stop")
                    self.front_left_motor.setVelocity(0)
                    self.front_right_motor.setVelocity(0)
                    self.back_left_motor.setVelocity(0)
                    self.back_right_motor.setVelocity(0)
                    
                elif incoming_msg_command == self.go_forward_config:
                    print("Go forward")
                    self.front_left_motor.setVelocity(self.target_speed)
                    self.front_right_motor.setVelocity(self.target_speed)
                    self.back_left_motor.setVelocity(self.target_speed)
                    self.back_right_motor.setVelocity(self.target_speed) 
                    
                elif incoming_msg_command == self.turn_left_config:
                    print("Go left")
                    self.front_left_motor.setVelocity(-self.target_speed * 1/8)
                    self.front_right_motor.setVelocity(self.target_speed * 1/8)
                    self.back_left_motor.setVelocity(-self.target_speed * 1/8)
                    self.back_right_motor.setVelocity(self.target_speed * 1/8)
                    
                elif incoming_msg_command == self.turn_right_config:
                    print("Go right")
                    self.front_left_motor.setVelocity(self.target_speed * 1/8)
                    self.front_right_motor.setVelocity(-self.target_speed * 1/8)
                    self.back_left_motor.setVelocity(self.target_speed * 1/8)
                    self.back_right_motor.setVelocity(-self.target_speed * 1/8)
                         
            except ConnectionResetError:
                print("Error occurs")
    
        self.client_socket.close()

    def get_sensor_data(self):
        current_pos = np.array(self.translation_field.getSFVec3f()) * np.array([1, -1, 1]) + self.position_thresh
        
        rotation = self.rotation_field.getSFRotation()
    
        # Extract yaw (rotation around Z-axis)
        rz = rotation[2]  # Z component of rotation axis
        theta = rotation[3]  # Rotation angle in radians
        
        # Ensure yaw is within -π to π range
        yaw = theta if rz >= 0 else -theta
        current_yaw_degrees = math.degrees(yaw) + self.yaw_degree_thresh
        
        current_yaw_degrees = current_yaw_degrees if current_yaw_degrees < 180 else current_yaw_degrees - 360
        
        return current_pos, current_yaw_degrees
    
    def get_current_rgb_frame(self):
        # Capture the image from the camera
        image = self.camera_rgb.getImage()
        
        # Convert the image to a NumPy array
        width = self.camera_rgb.getWidth()
        height = self.camera_rgb.getHeight()
        image_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
        
        # Remove the alpha channel for display (if required)
        image_array = image_array[:, :, :3]
        
        return image_array
        
    def get_current_depth_frame(self):
        # Capture the image from the camera
        depth_image = self.camera_depth.getRangeImage()
        
        # Convert the image to a NumPy array
        width = self.camera_depth.getWidth()
        height = self.camera_depth.getHeight()
        depth_array = np.array(depth_image, dtype=np.float32).reshape((height, width))
        
        return depth_array
        
    def send_sensor_data(self):
        while self.robot.step(self.timestep) != -1:
            # check whether sensor stream enable or disable and then stream the sensor data
            if self.sensor_stream_enable:
                # take the current RGB camera frame
                self.current_rgb_frame = self.get_current_rgb_frame()
                _, buffer_rgb = cv2.imencode('.jpg', self.current_rgb_frame)
                encoded_frame_rgb = base64.b64encode(buffer_rgb).decode('utf-8')

                # take the current depth camera frame
                self.current_depth_frame = self.get_current_depth_frame()
                _, buffer_depth = cv2.imencode('.jpg', self.current_depth_frame)
                encoded_frame_depth = base64.b64encode(buffer_depth).decode('utf-8')
                
                # take the current sensor data
                self.current_position, self.current_yaw = self.get_sensor_data()
                
                # print(self.current_position)
                # print(self.current_yaw)

                # outgoing message
                self.outgoing_message["message"]["rgbcamera_frame"] = encoded_frame_rgb
                self.outgoing_message["message"]["depthcamera_frame"] = encoded_frame_depth
                
                self.outgoing_message["message"]["sensor_data"]["posx"] = int(self.current_position[0] * 100)
                self.outgoing_message["message"]["sensor_data"]["posy"] = int(self.current_position[1] * 100)
                self.outgoing_message["message"]["sensor_data"]["posz"] = int(self.current_position[2] * 100)
                self.outgoing_message["message"]["sensor_data"]["yaw"] = self.current_yaw
                            
                # send the data (length-prefixed frame)
                send_frame(self.client_socket, json.dumps(self.outgoing_message).encode('utf-8'))

    def execute_robot(self):
        # Start the thread for receiving messages
        threading.Thread(target=self.receive_messages, daemon=True).start()
        
        # start streaming the sensor data
        self.send_sensor_data()
        
        # while self.robot.step(self.timestep) != -1:
        #     self.current_depth_frame = self.get_current_depth_frame()
        #     print(self.current_depth_frame.shape)

        #     # Normalize depth values to 0–255 for display
        #     normalized_depth = cv2.normalize(self.current_depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        #     depth_uint8 = normalized_depth.astype(np.uint8)

        #     # Optional: apply a colormap for better visualization
        #     colored_depth = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

        #     # Show the depth image
        #     cv2.imshow("Depth Frame", colored_depth)

        #     # Wait for 1ms for UI update, break if 'q' is pressed
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

roboto = PIONEER_3AT()
roboto.execute_robot()






        
        
        