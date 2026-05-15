from controller import Robot
import numpy as np
import cv2
import pygame
import os
from ultralytics import YOLO
import math

# Load the trained YOLOv8 model
model = YOLO(r"D:\Tensorflow learning\Langchain tutorials\new\Yolo\yolo_custom_train_lab_sim\runs\detect\train\weights\best.pt")  # Replace with the path to your trained YOLOv8 model

# Initialize Pygame
pygame.init()

# Initialize Webots robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Create a window (required for capturing key events)
screen = pygame.display.set_mode((100, 100))
pygame.display.set_caption("Continuous Key Press Detection")

# Get the range finder device
range_finder = robot.getDevice("range_finder")
range_finder.enable(timestep)

# Get the camera device
camera = robot.getDevice("CAM_front")  # Make sure your camera is named "CAM" in Webots
camera.enable(timestep)

# Get the motor devices
front_left_motor = robot.getDevice('front left wheel')
front_right_motor = robot.getDevice('front right wheel')
back_left_motor = robot.getDevice('back left wheel')
back_right_motor = robot.getDevice('back right wheel')

# You can modify these speeds as needed
forward_backward_speed = 6  # Example speed in rad/s
rotate_speed = 2  # Example speed in rad/s

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

# object positions

football = [-14, 0, 0.852]
tennisball = [-13.8, 1.42, 0.772]
soda = [-13.9, 2.04, 0.801]
cpu = [-14.3, 3.17, 0.77]
monitor = [-14.4, 4.31, 0.74]
wallart = [-13.6, 5.71, 1.63]
washing_machine = [-12.3, 5.2, 0.53]
couch = [-10.8, 5.13, 0]
exit_panel = [-10.6, 5.84, 2.25]
tree = [-9.05, 4.95, 0]
cat = [-7.85, 4.82, 0]
sink = [-5.8, 5.38, 0.65]
chair = [-6.5, 3.28, -0.001]
clock = [-5.85, 2.81, 1.91]
fire_estinguisher = [-6.1, 2, -0.004]
fire_exit_panel = [-5.85, 0.97, 2.51]
door = [-5.68, 0.9, 0]
box = [-6.46, -0.0606, 0.3]
biscuit_packet = [-6.49, 0, 0.6]
wateringcan = [-8.8, -0.807, 0]

object_array = [biscuit_packet, box, cat, chair, clock, couch, cpu, door, exit_panel, fire_estinguisher, 
                fire_exit_panel, football, monitor, sink, soda, tennisball, tree, wallart, washing_machine, 
                wateringcan]

while robot.step(timestep) != -1:
    # pygame key control
    # Get all pressed keys
    keys = pygame.key.get_pressed()

    # Check for key holds and print continuously
    if keys[pygame.K_w]:
        front_left_motor.setVelocity(forward_backward_speed)
        back_left_motor.setVelocity(forward_backward_speed)
        front_right_motor.setVelocity(forward_backward_speed)
        back_right_motor.setVelocity(forward_backward_speed)
    elif keys[pygame.K_s]:
        front_left_motor.setVelocity(-forward_backward_speed)
        back_left_motor.setVelocity(-forward_backward_speed)
        front_right_motor.setVelocity(-forward_backward_speed)
        back_right_motor.setVelocity(-forward_backward_speed)
    elif keys[pygame.K_d]:
        front_left_motor.setVelocity(rotate_speed)
        back_left_motor.setVelocity(rotate_speed)
        front_right_motor.setVelocity(-rotate_speed)
        back_right_motor.setVelocity(-rotate_speed)
    elif keys[pygame.K_a]:
        front_left_motor.setVelocity(-rotate_speed)
        back_left_motor.setVelocity(-rotate_speed)
        front_right_motor.setVelocity(rotate_speed)
        back_right_motor.setVelocity(rotate_speed)
    else:
        front_left_motor.setVelocity(0)
        back_left_motor.setVelocity(0)
        front_right_motor.setVelocity(0)
        back_right_motor.setVelocity(0)

    # Handle events like quitting
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False



    # Get depth image data
    depth_image = range_finder.getRangeImage()
    
    # Get image from camera
    image = camera.getImage()
    
    # Convert depth data to a NumPy arrqay
    width = range_finder.getWidth()
    height = range_finder.getHeight()
    
    # Get image dimensions
    width = camera.getWidth()
    height = camera.getHeight()
    
    depth_array = np.array(depth_image).reshape((height, width))  # Corrected reshape
    img_array = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))  # Webots returns BGRA format
    
    # 1. Clamp depth values greater than 20 to 20
    depth_array[depth_array > 20] = 20
    
    # Normalize depth values for visualization
    depth_normalized = (depth_array / 20) * 255
    depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    
    # Convert BGRA to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
    
    # Perform predictions
    results = model.predict(source=img_bgr, save=False, conf=0.5)  # Set conf to the desired confidence threshold
    
    # Extract annotated image
    annotated_image = results[0].plot()
    
    # Display the depth image
    cv2.imshow("Depth Image", depth_colored)
    # Show the camera image
    cv2.imshow("Webots Camera", annotated_image)
    
    depth_list = []
    obj_position_lst = []
    obj_pos_inimage_lst = []
    
    # Access bounding box information
    if len(results[0].boxes) >= 6:
    
        for box in results[0].boxes:
            coordinates_box = box.xyxy.cpu().numpy()[0]  # Bounding box coordinates
            confidence = box.conf.cpu().numpy()[0]         # Confidence scores
            class_index = box.cls.cpu().numpy()[0]         # Class indices
            
            x1, y1, x2, y2 = coordinates_box
        
            center_x = int((x1 + x2) // 2)
            center_y = int((y1 + y2) // 2)
            
            print("center_x", center_x, type(center_x))
            print("center_y", center_y, type(center_y))
        
            depth_value = depth_array[center_y, center_x]
            
            depth_list.append(depth_value)
            obj_position_lst.append(object_array[int(class_index)])
            obj_pos_inimage_lst.append([center_x, center_y])
            
        # calculate the current position and orientation
        # Extended to 6 known 3D coordinates of object points
        object_points = np.array(obj_position_lst, dtype=float)
        
        # Corresponding 2D coordinates on the camera image
        image_points = np.array(obj_pos_inimage_lst, dtype=float)
        
        # Camera matrix (ensure you replace with your camera's calibration data)
        camera_matrix = np.array([
            [519.6, 0, 300],
            [0, 519.6, 300],
            [0, 0, 1]
        ], dtype=float)
        
        # Assuming no lens distortion
        dist_coeffs = np.zeros(4)
        
        # Solve for pose using SOLVEPNP_ITERATIVE
        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        rmat, _ = cv2.Rodrigues(rotation_vector)
        tvec = translation_vector
        
        # Check if solvePnP was successful
        if success:
            print("Rotation Vector:\n", rotation_vector)
            print("Translation Vector:\n", translation_vector)
        else:
            print("solvePnP failed to find a solution.")
    else: 
        print("detected object number if lesser")
    
    # focal_length = camera.getFocalLength()
    # fov = camera.getFov()
    # width = camera.getWidth()

    # print("Focal Length:", focal_length)
    # print("fov:", fov)
    # print("width:", width)
    
    # focal_length_pixels = width / (2 * (math.tan(fov / 2)))
    # print("Computed Focal Length in Pixels:", focal_length_pixels)
    
    # # Access class names
    # class_names = results[0].names
    # print("Class Names:", class_names)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
pygame.quit()
