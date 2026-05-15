from controller import Robot
import numpy as np
import cv2
import pygame
import os
import math
import time


# Initialize Pygame
pygame.init()

# Initialize Webots robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Create a window (required for capturing key events)
width_height_pygame_window = 600
screen = pygame.display.set_mode((width_height_pygame_window, width_height_pygame_window))
pygame.display.set_caption("2D trajectory map")

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
rotate_speed = 1  # Example speed in rad/s

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


def moving_robot_key_command_check():
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

def get_depth_RGB_array():
    # Get depth image data
    depth_image = range_finder.getRangeImage()
    # Get image from camera
    image = camera.getImage()
    
    # Convert depth data to a NumPy array
    depthimage_width = range_finder.getWidth()
    depthimage_height = range_finder.getHeight()
    
    # Get image dimensions
    rgbimage_width = camera.getWidth()
    rgbimage_height = camera.getHeight()
    
    depth_array = np.array(depth_image).reshape((depthimage_height, depthimage_width))  # Corrected reshape
    img_array = np.frombuffer(image, dtype=np.uint8).reshape((rgbimage_height, rgbimage_width, 4))  # Webots returns BGRA format
    
    # Convert BGRA to BGR for OpenCV
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
    
    return depth_array, img_array

def visualize_depth_rgb_images(depth_img, rgb_img):
    # Normalize depth to range [0, 255]
    min_depth = np.min(depth_img)
    max_depth = np.max(depth_img)
    depth_normalized = ((depth_img - min_depth) / (max_depth - min_depth)) * 255
    
    # apply color map
    depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    # Display the depth image
    cv2.imshow("Depth Image", depth_colored)
    # Show the camera image
    cv2.imshow("RGB Image", rgb_img)
    
def extract_features(image):
    orb = cv2.ORB_create(nfeatures=1500)
    # Find the keypoints and descriptors with ORB
    kp, des = orb.detectAndCompute(image, None)
    
    return kp, des

def visualize_features(image, kp):
    image_with_features = cv2.drawKeypoints(image, kp, None, color=(0,255,0), flags=0)
    # Display the featured image
    cv2.imshow("Featured Image", image_with_features)
    
def match_features(des1, des2):
    # Define FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6,
                        key_size = 12,
                        multi_probe_level = 1)
    search_params = dict(checks = 50)
    
    # Initiate FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # Find matches with FLANN
    match = flann.knnMatch(des1, des2, k=2)
    
    return match

def filter_matches_distance(match, dist_threshold):
    # filtered match
    filtered_match = [item[0] for item in match if len(item) == 2 and item[0].distance < (dist_threshold * item[1].distance)]

    return filtered_match

def visualize_matches(image1, kp1, image2, kp2, match):
    image_matches = cv2.drawMatches(image1,kp1,image2,kp2,match,None)
    # Display the featured image
    cv2.imshow("Feature matched Image", image_matches)

def estimate_motion(match, kp1, kp2, k, discoeff, depth1=None):
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []
    
    objectpoints = []
    
    # Iterate through the matched features
    for m in match:
        # Get the pixel coordinates of features f[k - 1] and f[k]
        u1, v1 = kp1[m.queryIdx].pt
        u2, v2 = kp2[m.trainIdx].pt
        
        # Get the scale of features f[k - 1] from the depth map
        s = depth1[int(v1), int(u1)]
        
        # Check for valid scale values
        if s < 1000:
            # Transform pixel coordinates to camera coordinates using the pinhole camera model
            p_c = np.linalg.inv(k) @ (s * np.array([u1, v1, 1]))

            # Save the results
            image1_points.append([u1, v1])
            image2_points.append([u2, v2])
            objectpoints.append(p_c)
        
    # Convert lists to numpy arrays
    objectpoints = np.vstack(objectpoints)
    imagepoints = np.array(image2_points)
    
    # Determine the camera pose from the Perspective-n-Point solution using the RANSAC scheme
    _, rvec, tvec, _ = cv2.solvePnPRansac(objectpoints, imagepoints, k, discoeff)
    
    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)
    
    return rmat, tvec, image1_points, image2_points
    
def show_in_pygame_window(width_height_pg, robot_pos_points):
    # start from the middle point
    threshold = width_height_pg // 2
    
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)

    screen.fill(BLACK)
    
    # Draw points
    for point in robot_pos_points:
        adjusted_x = int(point[0] * 1000 + threshold)
        adjusted_y = int(point[1] * 1000 + threshold)
        pygame.draw.circle(screen, RED, (adjusted_x, adjusted_y), 5)
    
    pygame.display.flip()
        
def show_trajectory(iteration, current_trf_mtx, start_pos, rmat, tvec, robot_pos_points):
    if iteration % 2 == 0:
        instant_trf_mtx = np.eye(4)
        instant_trf_mtx[0:3, 0:3] = rmat
        instant_trf_mtx[0:3, 3] = tvec.T 
        
        full_trf_mtx = current_trf_mtx @ np.linalg.inv(instant_trf_mtx)
        
        # Calculate current camera position from origin
        current_position = full_trf_mtx @ start_pos
        # Build trajectory
        current_position_2d = current_position[0:2]
        print(current_position_2d)
        robot_pos_points.append(current_position_2d)
        
        # show in pygame window
        show_in_pygame_window(width_height_pygame_window, robot_pos_points)
        
        return full_trf_mtx, robot_pos_points
        
    else:
        return current_trf_mtx, robot_pos_points
    
    
    

if __name__ == "__main__":
    
    # filtering threshold for feature match
    filter_dist_threshold = 0.4
    # camera intrinsic matrix
    k = np.array([[500.715474,   0, 295.657257],
               [  0, 501.582101, 296.64938],
               [  0,   0,   1]])
               
    discoeff = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    # take the first keypoints and descriptors to match with rest
    if robot.step(timestep) != -1:
        depth_image1, rgb_image1 = get_depth_RGB_array()
        kp1, des1 = extract_features(rgb_image1)
        print(depth_image1[300, 300])
    
    iteration = 0
    current_trf_mtx = np.eye(4)
    robot_pos_points = []
    start_pos = np.array([0., 0., 0., 1.])
    
    while robot.step(timestep) != -1:
        
        # Move the robot with key commands
        moving_robot_key_command_check()
        
        # get depth and rgb images
        depth_image2, rgb_image2 = get_depth_RGB_array()
        
        # extract features
        kp2, des2 = extract_features(rgb_image2)
        
        # match the features
        matches = match_features(des1, des2)
        
        # filter the matches
        filtered_matches = filter_matches_distance(matches, filter_dist_threshold)
        
        # visualize the matches
        visualize_matches(rgb_image1, kp1, rgb_image2, kp2, filtered_matches)
        
        # estimate the trajectory
        rmat, tvec, _, _ = estimate_motion(filtered_matches, kp1, kp2, k, discoeff, depth_image1)
        
        # show the trajectory in pygame window
        current_trf_mtx, robot_pos_points = show_trajectory(iteration, current_trf_mtx, start_pos, rmat, tvec, robot_pos_points)
        
        # visualize depth and RGB images
        visualize_depth_rgb_images(depth_image2, rgb_image2)
        
        # parameter adjustments for next loop
        depth_image1 = depth_image2
        rgb_image1 = rgb_image2
        kp1 = kp2
        des1 = des2
        iteration += 1
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    cv2.destroyAllWindows()
    pygame.quit()
