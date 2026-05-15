from controller import Supervisor
import pygame
import math
import socket
import json
import threading
import numpy as np
import cv2
import base64

timestep = 64

shared_params = {
                    "sensor_stream_enable": False
                }

position_thresh = np.array([14.84, 5.76, 0])
yaw_degree_thresh = 90

# Robot initialisation
def Robot_init():
    robot = Supervisor() 
    
    robot_node = robot.getFromDef("PIONEER_3AT") 
    translation_field = robot_node.getField("translation")
    rotation_field = robot_node.getField("rotation")
    
    # init the camera
    # Initialize camera
    camera = robot.getDevice('CAM_front')
    camera.enable(timestep)
    
    # Get the motor devices
    front_left_motor = robot.getDevice('front left wheel')
    front_right_motor = robot.getDevice('front right wheel')
    back_left_motor = robot.getDevice('back left wheel')
    back_right_motor = robot.getDevice('back right wheel')
    
    # Set the target velocity (rad/s)
    target_speed_left = 4
    target_speed_right = 4
    
    # Set the motors positions
    front_left_motor.setPosition(float('inf')) 
    front_right_motor.setPosition(float('inf')) 
    back_left_motor.setPosition(float('inf')) 
    back_right_motor.setPosition(float('inf')) 
    
    # Set the target velocity
    front_left_motor.setVelocity(0)
    front_right_motor.setVelocity(0)
    back_left_motor.setVelocity(0)
    back_right_motor.setVelocity(0)
    
    return (front_left_motor, front_right_motor, back_left_motor, back_right_motor, 
            target_speed_left, target_speed_right, translation_field, rotation_field, robot, camera)

def ClientCom_init():
    server_ip = socket.gethostbyname(socket.gethostname())
    server_port = 5000  # Port for communication
    lock = threading.Lock()
    
    client_name = "Pioneer 3AT"

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))
    
    # Send name as the first message
    client_socket.send(client_name.encode())
    
    lock = threading.Lock()

    return client_socket, lock

def receive_messages(client_socket, lock, robot, sources):
    """Continuously receive messages from the server."""
    
    # sources 
    front_left_motor = sources[0]
    front_right_motor = sources[1]
    back_left_motor = sources[2]
    back_right_motor = sources[3]
    target_speed_left = sources[4]
    target_speed_right = sources[5]
    
    # incoming command message types
    Stop_message_config = "stop"
    go_forward_config = "forward"
    turn_left_config = "left"
    turn_right_config = "right"
    
    global shared_params
    
    while robot.step(timestep) != -1:
        try:
            message = client_socket.recv(1024).decode()
            if not message:
                break  
                
            # Parse the received JSON message
            message_data = json.loads(message)
            incoming_msg_command = message_data.get("command")
            incoming_msg_sensor_stream_enable = message_data.get("sensor_stream_enable")
            
            # check camera stream and sensor stream enable      
            if incoming_msg_sensor_stream_enable:
                with lock:
                    shared_params["sensor_stream_enable"] = True 
            else:
                with lock:
                    shared_params["sensor_stream_enable"] = False       
                     
            
            # check the commands to the robot
            if incoming_msg_command == Stop_message_config:
                print("Stop")
                front_left_motor.setVelocity(0)
                front_right_motor.setVelocity(0)
                back_left_motor.setVelocity(0)
                back_right_motor.setVelocity(0)
                
            elif incoming_msg_command == go_forward_config:
                print("Go forward")
                front_left_motor.setVelocity(target_speed_left)
                front_right_motor.setVelocity(target_speed_left)
                back_left_motor.setVelocity(target_speed_left)
                back_right_motor.setVelocity(target_speed_left) 
                
            elif incoming_msg_command == turn_left_config:
                print("Go left")
                front_left_motor.setVelocity(-target_speed_left * 1/8)
                front_right_motor.setVelocity(target_speed_left * 1/8)
                back_left_motor.setVelocity(-target_speed_left * 1/8)
                back_right_motor.setVelocity(target_speed_left * 1/8)
                
            elif incoming_msg_command == turn_right_config:
                print("Go right")
                front_left_motor.setVelocity(target_speed_left * 1/8)
                front_right_motor.setVelocity(-target_speed_left * 1/8)
                back_left_motor.setVelocity(target_speed_left * 1/8)
                back_right_motor.setVelocity(-target_speed_left * 1/8)
                     
        except ConnectionResetError:
            break

    client_socket.close()
    
def get_sensor_data(translation_field, rotation_field, position_thresh, yaw_degree_thresh):
    current_pos = np.array(translation_field.getSFVec3f()) * np.array([1, -1, 1]) + position_thresh
    
    rotation = rotation_field.getSFRotation()

    # Extract yaw (rotation around Z-axis)
    rz = rotation[2]  # Z component of rotation axis
    theta = rotation[3]  # Rotation angle in radians
    
    # Ensure yaw is within -π to π range
    yaw = theta if rz >= 0 else -theta
    current_yaw_degrees = math.degrees(yaw) + yaw_degree_thresh
    
    current_yaw_degrees = current_yaw_degrees if current_yaw_degrees < 180 else current_yaw_degrees - 360
    
    return current_pos, current_yaw_degrees

def get_image_webot_camera(camera):
    # Capture the image from the camera
    image = camera.getImage()
    
    # Convert the image to a NumPy array
    width = camera.getWidth()
    height = camera.getHeight()
    image_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    
    # Remove the alpha channel for display (if required)
    image_array = image_array[:, :, :3]
    
    return image_array
 
def Execution():
    # initialise the communication
    client_socket, lock = ClientCom_init()
    # initialise the robot
    sources = Robot_init()
    
    translation_field = sources[6]
    rotation_field = sources[7]
    robot = sources[8]
    camera = sources[9]
    
    # Start a thread to receive messages
    threading.Thread(target=receive_messages, args=(client_socket, lock, robot, sources), daemon=True).start()
    
    request_arrived = False
    
    outgoing_message = {
                            "From": "Pioneer 3AT", 
                            "To"  : "Com Client", 
                            "message": {
                                        "camera_frame": None, 
                                        "sensor_data": {
                                                        "posx": None, 
                                                        "posy": None, 
                                                        "posz": None, 
                                                        "yaw": None
                                                        }
                                        }
                       }
    
    while robot.step(timestep) != -1:
        # check and start sensor stream
        current_frame = get_image_webot_camera(camera)
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', current_frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')
        
        pos, yaw = get_sensor_data(translation_field, rotation_field, position_thresh, yaw_degree_thresh)
        
        with lock: 
            if shared_params["sensor_stream_enable"]:
                request_arrived = True
                
                outgoing_message["message"]["camera_frame"] = encoded_frame
                
                outgoing_message["message"]["sensor_data"]["posx"] = int(pos[0] * 100)
                outgoing_message["message"]["sensor_data"]["posy"] = int(pos[1] * 100)
                outgoing_message["message"]["sensor_data"]["posz"] = int(pos[2] * 100)
                outgoing_message["message"]["sensor_data"]["yaw"] = yaw
            else:
                request_arrived = False
                      
        # send the data
        if request_arrived:
            client_socket.sendall(json.dumps(outgoing_message).encode('utf-8'))
    

Execution()
    
