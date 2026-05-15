import socket
import time
import threading
from controller import Robot
import numpy as np

##################################################################

# Create the Robot instance
robot = Robot()

# Time step of the simulation in milliseconds
TIME_STEP = 64

# Get the motor devices
front_left_motor = robot.getDevice('front left wheel')
front_right_motor = robot.getDevice('front right wheel')
back_left_motor = robot.getDevice('back left wheel')
back_right_motor = robot.getDevice('back right wheel')

# Set the target velocity (rad/s)
# You can modify these speeds as needed
target_speed_left = 3  # Example speed in rad/s
target_speed_right = 3  # Example speed in rad/s

# Set the motors to velocity mode
front_left_motor.setPosition(float('inf'))  # Infinite position for velocity mode
front_right_motor.setPosition(float('inf'))  # Infinite position for velocity mode
back_left_motor.setPosition(float('inf'))  # Infinite position for velocity mode
back_right_motor.setPosition(float('inf'))  # Infinite position for velocity mode

# Set the target velocity
front_left_motor.setVelocity(0)
front_right_motor.setVelocity(0)
back_left_motor.setVelocity(0)
back_right_motor.setVelocity(0)

Kp = 0.1

#####################################################################

# Function to handle a connected client
def handle_client(conn):
    conn.setblocking(False)  # Set the connection to non-blocking mode
    while True:
        try:
            data = conn.recv(1024)  # Try to receive data
            if data:
                try:         
                    angle = float(data.decode())
                    print(f"received angle is {angle}")
                    # left and right motor speeds
                    speed_left = target_speed_left + Kp * angle
                    speed_right = target_speed_right - Kp * angle
                    
                    bounded_speed_left = np.clip(speed_left, -1, 3)
                    bounded_speed_right = np.clip(speed_right, -1, 3)
                    
                    print(f"speed values {bounded_speed_left}, {bounded_speed_right}")
                    # set the wheel speeds
                    front_left_motor.setVelocity(bounded_speed_left)
                    back_left_motor.setVelocity(bounded_speed_left)
                    front_right_motor.setVelocity(bounded_speed_right)
                    back_right_motor.setVelocity(bounded_speed_right)
                except:
                    # Input string
                    my_string = f"{data.decode()}"
                    print(data.decode())
                    # check for the validity
                    if (my_string == "Launch obstacle avoidance") or (my_string == "Launch path planning") or (my_string == "Launch the robot"):
                        pass
                    else:    
                        # Convert string to a list of characters
                        char_list = list(my_string)
                        
                        if char_list[-1] == 'h':
                            front_left_motor.setVelocity(0)
                            back_left_motor.setVelocity(0)
                            front_right_motor.setVelocity(0)
                            back_right_motor.setVelocity(0)

        except BlockingIOError: 
            pass  # No data received, continue running the loop
        # time.sleep(0.1)  # Small delay to avoid excessive CPU usage

# Create a socket object
receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
receiver_socket.setblocking(False)  # Set the socket to non-blocking mode

# Host and port configuration
host = '192.168.8.249' # Host to listen on
port = 5000         # Port to listen on


try:
    # Bind to the host and port
    receiver_socket.bind((host, port))
    receiver_socket.listen(5)
    print(f"Receiver is listening on {host}:{port}...")
    
    # Main loop for "server is running"
    while robot.step(TIME_STEP) != -1:
        try:
            # Try to accept a connection
            conn, addr = receiver_socket.accept()
            print(f"Connected by {addr}")
            # Start a new thread to handle the connected client
            threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
        except BlockingIOError:
            pass  # No connection attempt, continue running the loop
        
        # Print "server is running"
        #print("server is running")
except Exception as e:
    print(f"Error: {e}")
finally:
    receiver_socket.close()