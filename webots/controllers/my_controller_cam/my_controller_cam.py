from controller import Robot
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import cv2
import time
import socket
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO(r"D:\Tensorflow learning\Langchain tutorials\new\Yolo\best.pt")  # Replace with the path to your trained YOLOv8 model

goal_label = -1
robot_bbox = []

class RRT:
    class Node:
        def __init__(self, coord):
            self.coord = coord
            self.parent = None

    def __init__(self, start, goal, obstacled_image, orginal_image, rand_area, robot_size, expand_dis=50, goal_sample_rate=10, max_iter=1000):
        self.start = self.Node(start)
        self.end = self.Node(goal)
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacled_image = obstacled_image
        self.orginal_image = orginal_image.copy()
        self.robot_size = robot_size
        self.min_distance_goal_to_end = 20

    def get_random_node(self):
        if random.randint(self.min_rand, self.max_rand) > self.goal_sample_rate:
            return self.Node((random.randint(self.min_rand, self.max_rand), random.randint(self.min_rand, self.max_rand)))
        else:
            return self.Node(self.end.coord)

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [math.hypot((node.coord[0] - rnd_node.coord[0]), (node.coord[1] - rnd_node.coord[1])) for node in node_list]
        min_index = dlist.index(min(dlist))
        return min_index

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.coord)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.coord = (int(new_node.coord[0] + min(d, extend_length) * math.cos(theta)),
                          int(new_node.coord[1] + min(d, extend_length) * math.sin(theta)))

        new_node.parent = from_node
        return new_node

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.coord[0] - from_node.coord[0]
        dy = to_node.coord[1] - from_node.coord[1]
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta
    
    def mask_path_to_image(self, binary_image, p1, p2, p3, p4):
        polygon_1_points = np.array([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]], [p4[0], p4[1]]], np.int32)
        polygon_2_points = np.array([[p1[0], p1[1]], [p2[0], p2[1]], [p4[0], p4[1]], [p3[0], p3[1]]], np.int32)

        cv2.fillPoly(binary_image, [polygon_1_points], color=0)  # Fill the polygon with white (255)
        cv2.fillPoly(binary_image, [polygon_2_points], color=0)  # Fill the polygon with white (255)

        return binary_image
    
    def detect_changes_fast(self, image1, image2):
        assert image1.shape == image2.shape, "Images must have the same dimensions."
        changes = (image1 ^ image2)
        
        # from xor operation you get a binary array with 1 and 0s. If there are any changes you will get 1 in the array.
        # Otherwise 0. So inorder to show the image you have to make true values to 255. 
        changes_binary = (changes > 0).astype(np.uint8) * 255

        # Check for any changes and print message
        if np.any(changes_binary):
            return False
        else:
            return True

    def collision_check(self, from_node, to_node, binary_image, width):
        # dublicate_binary_image
        binary_image_org = binary_image.copy()
        binary_image_dub = binary_image.copy()

        # Given points
        from_coord = np.array([from_node.coord[0], from_node.coord[1], 0])  # Point 1
        to_coord = np.array([to_node.coord[0], to_node.coord[1], 0])  # Point 2

        # Step 1: Calculate the slope of the line connecting the two points
        vector_1 = to_coord - from_coord
        vector_2 = np.array([0, 0, 1])

        # Step 1: Calculate the cross product
        perpendicular_vector = np.cross(vector_1, vector_2)

        # Step 2: Calculate the magnitude of the cross product
        magnitude = np.linalg.norm(perpendicular_vector)
        # Normalize the perpendicular vector
        unit_perpendicular_vector = perpendicular_vector / magnitude
        print(unit_perpendicular_vector)

        # Perpendicular points at (x1, y1)
        p1_perp1 = from_coord + unit_perpendicular_vector * width
        p1_perp2 = from_coord - unit_perpendicular_vector * width

        # Perpendicular points at (x2, y2)
        p2_perp1 = to_coord + unit_perpendicular_vector * width
        p2_perp2 = to_coord - unit_perpendicular_vector * width
        
        print(p1_perp1)
        print(p1_perp2)
        print(p2_perp1)
        print(p2_perp2)
        print(" ")

        masked_image = self.mask_path_to_image(binary_image_dub, p1_perp1, p1_perp2, p2_perp1, p2_perp2)
        safe_path = self.detect_changes_fast(masked_image, binary_image_org)

        return safe_path

    def planning(self):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.collision_check(nearest_node, new_node, self.obstacled_image, self.robot_size):
                self.node_list.append(new_node)
                if self.calc_distance_and_angle(new_node, self.end)[0] <= self.expand_dis:
                    final_node = self.steer(new_node, self.end, self.expand_dis)
                    if self.collision_check(new_node, final_node, self.obstacled_image, self.robot_size):
                        return self.generate_final_course(len(self.node_list) - 1)
                    
            # if i % 10 == 0:
            #     self.draw_current(rnd_node)

        return None  # cannot find

    def generate_final_course(self, goal_ind):
        path = [[self.end.coord[0], self.end.coord[1]]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.coord[0], node.coord[1]])
            node = node.parent
        path.append([self.start.coord[0], self.start.coord[1]])
        
        # modify the path
        modified_path = []
        main_coord = path[0]
        modified_path.append(main_coord)
        
        for i in range(1, len(path)):
            next_coord = path[i]
            if (self.collision_check(self.Node(main_coord), self.Node(next_coord), self.obstacled_image, self.robot_size)):
                if i == len(path) - 1:
                    modified_path.append(path[i])
                    break
                continue
            else:
                modified_path.append(path[i-1])
                main_coord = path[i-1]
                if i == len(path) - 1:
                    modified_path.append(path[i])
        
        return modified_path

    def draw_graph(self, rnd=None):
        pass

    @staticmethod
    def plot_circle(x, y, size):
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, "-k")


def save_images(img):
    if (interation % 30) == 0:
        cv2.imwrite(rf"D:\Tensorflow learning\Langchain tutorials\new\vision part test\test5\topview_{interation}.png", img)

def detect_obstacles(img):
    robot_detected = False
    # Perform predictions
    results = model.predict(source=img, save=False, conf=0.25)  # Set conf to the desired confidence threshold
    obstacle_list = []
    # Access bounding box information
    for box in results[0].boxes:
        bounding_box_coord =  box.xyxy.cpu().numpy()[0]  # Bounding box coordinates
        confident_score = box.conf.cpu().numpy()[0]      # Confidence scores
        class_indices = box.cls.cpu().numpy()[0]        # Class indices
        
        if class_indices != 1 and confident_score > 0.5:
            if class_indices != goal_label:
                obstacle_list.append(bounding_box_coord)
        elif class_indices == 1 and confident_score > 0.5:
            global robot_bbox
            robot_bbox = bounding_box_coord
            robot_detected = True
    
    return obstacle_list, robot_detected
    
def detect_direction(img):
    h, w, _ = img.shape
    
    # Create a mask filled with the given color (e.g., blue in BGR)
    color = (0, 0, 0)  # Black
    mask = np.full_like(img, color, dtype=np.uint8)
    
    # robot bounding box border
    x1, y1 = int(robot_bbox[0]), int(robot_bbox[1])
    x2, y2 = int(robot_bbox[2]), int(robot_bbox[3])
    
    # Exclude the bounding box region from the mask
    mask[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    
    # Define the HSV range for blue color
    lower_blue = np.array([100, 150, 50])    
    upper_blue = np.array([140, 255, 255])
    
    # Define the HSV range for green color
    lower_green = np.array([40, 50, 150])     
    upper_green = np.array([80, 255, 255])
    
    # Create masks for blue and green colors
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours for blue dots
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find contours for green dots
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return blue_contours, green_contours
    
def show_contours(img, obstacle_contours, blue_contours, green_contours):
    # Loop through the contours and draw bounding boxes
    for cnt in obstacle_contours:
        # Get the bounding box coordinates
        x1, y1, x2, y2 = [int(x) for x in cnt]
        # Draw the bounding box on the original image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)  # Green box
            
    for cnt in blue_contours:
        # Get the bounding box or minimum enclosing circle
        if cv2.contourArea(cnt) > 10:  # Filter out very small contours
            x, y, w, h = cv2.boundingRect(cnt)
            # Calculate the center of the blue dot
            center_x, center_y = x + w // 2, y + h // 2
            cv2.circle(img, (center_x, center_y), 1, (0, 0, 0), -1)  # Mark the center (blue dot)
            # print(f"Blue dot position: ({center_x}, {center_y})")

    for cnt in green_contours:
        # Get the bounding box or minimum enclosing circle
        if cv2.contourArea(cnt) > 10:  # Filter out very small contours
            x, y, w, h = cv2.boundingRect(cnt)
            # Calculate the center of the green dot
            center_x, center_y = x + w // 2, y + h // 2
            cv2.circle(img, (center_x, center_y),  1, (0, 0, 0), -1)  # Mark the center (green dot)
            
    return img

def create_binary_image(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Create a binary version of the image (initialize with zeros for black)
    binary_image = np.zeros_like(gray_image)

    return binary_image

def fill_binary_image_with_obstacles(img, obstacle_contours):
    # Loop through the contours and fill bounding boxes
    for cnt in obstacle_contours:
        # Get the bounding box coordinates
        x1, y1, x2, y2 = [int(x) for x in cnt]
        # Fill the bounding box with white color
        cv2.rectangle(img, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)

    return img

def draw_final_path(path, img_org):
    # path drawing
    for i in range(1, len(path)):
        cv2.line(img_org, path[i-1], path[i], (0,255,0), 5)
    # start point and end point drawing
    cv2.circle(img_org, path[0], 5, (255,0,0), -1)
    cv2.circle(img_org, path[-1], 5, (0,0,255), -1)

    cv2.imshow("Final path", img_org)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def draw_only_final_path(path, img_org):
    # path drawing
    for i in range(1, len(path)):
        cv2.line(img_org, path[i-1], path[i], (0,255,0), 1)
        
    return img_org       

def initialise_robot():
    # Initialize robot
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    
    # Initialize camera
    camera = robot.getDevice('CAM')
    camera.enable(timestep)
    
    return robot, camera, timestep

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
    
def initialise_sender():
    # Create a socket object
    sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # port configuration
    host = '192.168.8.249' # Host to listen on
    port = 5000         # Port to listen on
    
    # connect host and port
    sender_socket.connect((host, port))
    
    return sender_socket

def send_data_sockets(sender_socket, signed_angle_degrees):
    message = f"{signed_angle_degrees}"
    sender_socket.sendall(message.encode())
      
def robot_controller(robo_front_coord, robo_back_coord, path, to_point_index):
    # remaining angle between path direction and current head direction and the remaining distance
    # find the necessary coordinates
    to_point_coord = path[to_point_index]
    
    # if cv2.contourArea(robo_front_coord[0]) > 5:  # Filter out very small contours
    x, y, w, h = cv2.boundingRect(robo_front_coord[0])
    robo_front_coord_x, robo_front_coord_y = x + w // 2, y + h // 2
        
    # if cv2.contourArea(robo_back_coord[0]) > 5:  # Filter out very small contours
    x, y, w, h = cv2.boundingRect(robo_back_coord[0])
    robo_back_coord_x, robo_back_coord_y = x + w // 2, y + h // 2
    
    # find the vectors    
    robo_front_coord = np.array([robo_front_coord_x, robo_front_coord_y, 0])
    robo_back_coord = np.array([robo_back_coord_x, robo_back_coord_y, 0])
    to_point_coord = np.array([*to_point_coord, 0])
    
    # robot head direction vector
    vect_robo_direction = robo_front_coord - robo_back_coord
    # path head direction vector
    vect_path_direction = to_point_coord - robo_front_coord
    # vector dot product
    dot_product = np.dot(vect_robo_direction, vect_path_direction)
    # vector magnitudes
    mag_vect_robo_direction = np.linalg.norm(vect_robo_direction)
    mag_vect_path_direction = np.linalg.norm(vect_path_direction)
    # cos angle between two directions
    cos_theta = dot_product / (mag_vect_robo_direction * mag_vect_path_direction)
    # angle in radians
    angle_degrees = np.degrees(np.arccos(cos_theta))
    
    # check wether angle is + or -
    cross_product = np.cross(vect_robo_direction, vect_path_direction)
    # Determine sign based on the z-component
    sign = np.sign(cross_product[-1])  # Use the z-component for 2D
    signed_angle_degrees = sign * angle_degrees
    
    if mag_vect_path_direction < 20:
        to_point_index += 1
    
    return signed_angle_degrees, to_point_index 

def Goal_label_coord_detection(expected_img_width, expected_img_height, robot, camera, timestep):
    # get the image from webot camera
    if robot.step(timestep) != -1:
        webot_camera_image_org = get_image_webot_camera(camera)

    # Resize the image
    resized_image_org = cv2.resize(webot_camera_image_org, (expected_img_width, expected_img_height))
    
    # Perform predictions
    results = model.predict(source=resized_image_org, save=False, conf=0.25)  # Set conf to the desired confidence threshold
    
    annotated_image = results[0].plot()
    
    goal_pos = []
    start_pos = []
    
    # Access bounding box information
    for box in results[0].boxes:
        bounding_box_coord =  box.xyxy.cpu().numpy()[0]  # Bounding box coordinates
        confident_score = box.conf.cpu().numpy()[0]         # Confidence scores
        class_indices = box.cls.cpu().numpy()[0]        # Class indices
        
        if class_indices == goal_label and confident_score > 0.5:
            goal_pos = [(bounding_box_coord[0] + bounding_box_coord[2]) / 2, (bounding_box_coord[1] + bounding_box_coord[3]) / 2]
        if class_indices == 1 and confident_score > 0.5:
            start_pos = [(bounding_box_coord[0] + bounding_box_coord[2]) / 2, (bounding_box_coord[1] + bounding_box_coord[3]) / 2]
    
    # Display the image with predictions
    cv2.imshow("Predicted Image", annotated_image)
    
    # Wait for a key press and close the window
    cv2.waitKey(0)
    return [int(i) for i in goal_pos], [int(i) for i in start_pos]

def secondary_sender(host, port, message):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((host, port))
        print(f"Connected to receiver at {host}:{port}")
        
        client_socket.sendall(message.encode('utf-8'))
        print("message successfully sent to the robot")
        time.sleep(1)

def secondary_receiver(host, port, waited_message, robot, timestep):
    # Create a socket object
    receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind to a specific host and port
    receiver_socket.bind((host, port))
    receiver_socket.listen(1)

    # print(f"Receiver is listening on {host}:{port}...")

    # Accept a connection
    conn, addr = receiver_socket.accept()
    # print(f"Connected by {addr}")

    try:
        while True:
            # Receive data in chunks of 1024 bytes
            data = conn.recv(1024)
            if not data:
                break
            elif data.decode() == waited_message:
                print("data received: ", data.decode())
                print("Received the message from agent")
                break
            elif data.decode()[:-2] == waited_message:
                print("data received: ", data.decode())
                print("Received the message from agent with label")
                global goal_label
                goal_label = int(data.decode()[-1])
                break
            
    except KeyboardInterrupt:
        print("\nReceiver stopped.")
    finally:
        conn.close()
        receiver_socket.close()

          
def main():
    # initialise robot
    robot, camera, timestep = initialise_robot()
    
    host = '192.168.8.249' # Host to listen on
    # port configuration
    port = 21042         # Port to listen on
    
    expected_img_width = 800
    expected_img_height = 800
    expected_robot_size = 50
    start_pos = [700, 100]
    goal_pos = [200, 200]
    
    iteration_no = 0
   
    while robot.step(timestep) != -1:
        time.sleep(0.1)
        iteration_no += 1
        print(iteration_no)
        if iteration_no > 20:
            break
    
    # waiting for message from label deciding agent
    waited_message = "Launch label detection"
    secondary_receiver(host, port, waited_message, robot, timestep)
    
    print("Start Label detection")
    time.sleep(1)
    
    # goal_label coordinate detection
    goal_pos, start_pos = Goal_label_coord_detection(expected_img_width, expected_img_height, robot, camera, timestep)
    
    # label detection completed message
    to_agent_message = "completed"
    secondary_sender(host, port, to_agent_message)
    
    print("Start pos: ", start_pos)
    print("Goal pos: ", goal_pos)
            
    # waiting for message from obstacle detection agent
    waited_message = "Launch obstacle avoidance"
    secondary_receiver(host, port, waited_message, robot, timestep)
    
    print("Start Obstacle detection")
    time.sleep(1)
    
    # get the image from webot camera
    if robot.step(timestep) != -1:
        webot_camera_image_org = get_image_webot_camera(camera)

    # Resize the image
    resized_image_org = cv2.resize(webot_camera_image_org, (expected_img_width, expected_img_height))
    resized_image = resized_image_org.copy()
    # cv2.imshow("Original image", resized_image)
    
    # detect obstacles contours (here output is in list format)
    obstacle_contours, _ = detect_obstacles(resized_image)
    # detect red and green dot contours
    blue_contours, green_contours = detect_direction(resized_image)
    
    # starting position should be the blue contour since it represent the head of the robot
    for cnt in blue_contours:
        # Get the bounding box or minimum enclosing circle
        if cv2.contourArea(cnt) > 10:  # Filter out very small contours
            x, y, w, h = cv2.boundingRect(cnt)
            # Calculate the center of the blue dot
            center_x, center_y = x + w // 2, y + h // 2
            start_pos = [center_x, center_y]
    
    # show contours
    detection_image = show_contours(resized_image, obstacle_contours, blue_contours, green_contours)
    cv2.imshow("Detection image", detection_image) # Display the obstacle and robo key point detected image.
    cv2.waitKey(0)
    
    # create the same size binary image with background black
    plane_binary_image = create_binary_image(resized_image)
    
    # fill the obstacles with white boxes
    obstacle_filled_binary_image = fill_binary_image_with_obstacles(plane_binary_image, obstacle_contours)
    cv2.imshow("Obstacle filled binary image", obstacle_filled_binary_image) # Display the obstacle filled binary image.
    cv2.waitKey(0)
    
    # obstacle detection completed message
    to_agent_message = "completed"
    secondary_sender(host, port, to_agent_message)
        
    # waiting for message from path planning agent
    waited_message = "Launch path planning"
    secondary_receiver(host, port, waited_message, robot, timestep)    
    
    # initialise RRT algorithm
    rrt = RRT(start=start_pos, goal=goal_pos, obstacled_image=obstacle_filled_binary_image, orginal_image = resized_image_org, 
              rand_area=[0, expected_img_width], robot_size=expected_robot_size)
    # planning the path          
    path = rrt.planning()
    
    # draw the path
    if path is None:
        print("Cannot find path")
        return 
    else:
        print("Found path!!")
        # flip the list to adjust start to end
        path.reverse()

        # Draw final path
        draw_final_path(path, resized_image_org)
        
    # path palnning completed message
    to_agent_message = "completed"
    secondary_sender(host, port, to_agent_message) 
    
    time.sleep(2)
    
    # waiting for message from path planning agent
    waited_message = "Launch the robot"
    secondary_receiver(host, port, waited_message, robot, timestep)   
        
    # initialise socket sender
    sender_socket = initialise_sender()
    
    next_pathpoint_no = 0
        
    while robot.step(timestep) != -1:
        # get the image from webot
        webot_camera_image_org = get_image_webot_camera(camera)
        
        # Resize the image
        resized_image = cv2.resize(webot_camera_image_org, (expected_img_width, expected_img_height))
        
        # detect obstacles contours
        obstacle_contours, robot_detected = detect_obstacles(resized_image)
        
        if not robot_detected:
            print("robot is missed from frame")
            continue
        
        # detect red and green dot contours
        blue_contours, green_contours = detect_direction(resized_image)
        
        # show the path
        resized_image = draw_only_final_path(path, resized_image)
        
        # show contours
        resized_image = show_contours(resized_image, obstacle_contours, blue_contours, green_contours)
        
        # Robot controller (traveller speed commands)
        angle, next_pathpoint_no = robot_controller(blue_contours, green_contours, path, next_pathpoint_no)
        
        # Finish at the end of the path
        if next_pathpoint_no == len(path):
            send_data_sockets(sender_socket, "finish")
            time.sleep(0.1)
            break
        
        # send angle to adjust speed
        send_data_sockets(sender_socket, angle)
          
        # Display the resized camera image
        cv2.imshow("Resized Camera Output", resized_image)
        
        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
       
    # Clean up
    time.sleep(5)
    # path palnning completed message
    to_agent_message = "completed"
    secondary_sender(host, port, to_agent_message) 
    
    cv2.destroyAllWindows()
    
    # close sockets
    sender_socket.close()
        
    
if __name__ == '__main__':
    main()