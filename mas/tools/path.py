"""There is a issue with the current path planning algorithm. 
if the goal position is inside an obstacle and distance between the goal and the last position is greater
than expanded distance, the algorithm will not be able to find the path.
"""

import cv2
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import os

class RRT:
    class Node:
        def __init__(self, coord):
            self.coord = coord
            self.parent = None

    def __init__(self, binary_map, door_list):
        self.min_rand_x = 20
        self.max_rand_x = binary_map.shape[1] - 20
        self.min_rand_y = 20
        self.max_rand_y = binary_map.shape[0] - 20
        self.expand_dis = 50
        self.goal_sample_rate_1 = 40
        self.goal_sample_rate_2 = 100
        self.max_iter = 3000
        self.binary_map = binary_map
        self.robot_size = 70
        self.min_distance_goal_to_end = 25

        self.base_path = os.path.dirname(__file__)
        self.plan_image_path = os.path.join(self.base_path, r"properties\binary_map\binary_map_current.png")
        
        self.scale_factor_rt_display = 0.5
        self.display_rt_plan_image_dimension = None
        self.door_list = door_list
        self.current_door = None

        # random.seed(1)

        # random.seed(0)  # for reproducibility

    def update_binary_map(self, binary_map):
        self.binary_map = binary_map

    def get_random_node(self):
        if random.randint(self.min_rand_x, self.max_rand_x) > self.goal_sample_rate_2:
            return self.Node((random.randint(self.min_rand_x, self.max_rand_x), random.randint(self.min_rand_y, self.max_rand_y)))
        elif random.randint(self.min_rand_x, self.max_rand_x) > self.goal_sample_rate_1:
            x_range = (self.door_list[self.current_door][0], self.door_list[self.current_door][0] + self.door_list[self.current_door][2])
            y_range = (self.door_list[self.current_door][1], self.door_list[self.current_door][1] + self.door_list[self.current_door][3])

            return self.Node((random.randint(x_range[0], x_range[1]), random.randint(y_range[0], y_range[1])))
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

        cv2.fillPoly(binary_image, [polygon_1_points], color=0)  # Fill the polygon with black
        cv2.fillPoly(binary_image, [polygon_2_points], color=0)  # Fill the polygon with black

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

        # if magnitude = 0, there can be an error. so return as false to indicate that path is not safe
        if magnitude == 0:
            return False

        # Normalize the perpendicular vector
        unit_perpendicular_vector = perpendicular_vector / magnitude

        # Perpendicular points at (x1, y1)
        p1_perp1 = from_coord + unit_perpendicular_vector * width / 2
        p1_perp2 = from_coord - unit_perpendicular_vector * width / 2

        # Perpendicular points at (x2, y2)
        p2_perp1 = to_coord + unit_perpendicular_vector * width / 2
        p2_perp2 = to_coord - unit_perpendicular_vector * width / 2

        masked_image = self.mask_path_to_image(binary_image_dub, p1_perp1, p1_perp2, p2_perp1, p2_perp2)

        safe_path = self.detect_changes_fast(masked_image, binary_image_org)

        return safe_path

    def planning(self, start_pos, goal_pos, current_door_num = 1, rt_planning_display=False):
        self.current_door = current_door_num - 1
        self.start = self.Node(start_pos)
        self.end = self.Node(goal_pos)

        if rt_planning_display:
            rt_plan_image = cv2.imread(self.plan_image_path)
            width_rt_plan_img, height_rt_plan_img = rt_plan_image.shape[1], rt_plan_image.shape[0]
            self.display_rt_plan_image_dimension = ( int(width_rt_plan_img * self.scale_factor_rt_display),  int(height_rt_plan_img * self.scale_factor_rt_display))

            cv2.circle(rt_plan_image, self.start.coord, radius=5, color=(255, 0, 0), thickness=-1)
            cv2.putText(rt_plan_image, "start", (self.start.coord[0], self.start.coord[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            cv2.circle(rt_plan_image, self.end.coord, radius=5, color=(0, 0, 255), thickness=-1)
            cv2.putText(rt_plan_image, "end", (self.end.coord[0], self.end.coord[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            prev_index = 0

        self.node_list = [self.start]

        try: 
            for i in range(self.max_iter):
                rnd_node = self.get_random_node()
                nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
                nearest_node = self.node_list[nearest_ind]

                new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

                if self.collision_check(nearest_node, new_node, self.binary_map, self.robot_size):
                    self.node_list.append(new_node)
                    if self.calc_distance_and_angle(new_node, self.end)[0] <= self.expand_dis:
                        final_node = self.steer(new_node, self.end, self.expand_dis)
                        if self.collision_check(new_node, final_node, self.binary_map, self.robot_size):
                            return self.generate_final_course(len(self.node_list) - 1)
                        
                if rt_planning_display:
                    if i % 10 == 0:
                        for j in range(prev_index, len(self.node_list)):
                            cv2.circle(rt_plan_image, self.node_list[j].coord, radius=1, color=(0, 255, 0), thickness=-1)
                        prev_index = len(self.node_list)
                        display_img = rt_plan_image.copy()
                        display_img = cv2.resize(display_img, self.display_rt_plan_image_dimension)
                        cv2.imshow("Realtime path nodes", display_img)
                        cv2.waitKey(1)
        
        except:
            return None

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
            if (self.collision_check(self.Node(main_coord), self.Node(next_coord), self.binary_map, self.robot_size)):
                if i == len(path) - 1:
                    modified_path.append(path[i])
                    break
                continue
            else:
                modified_path.append(path[i-1])
                main_coord = path[i-1]
                if i == len(path) - 1:
                    modified_path.append(path[i])
        
        modified_path.reverse()

        print(modified_path)

        # remodify the path
        remodified_path = []
        choosen_index_main = 0
        remodified_path.append(modified_path[choosen_index_main])
        len_modified_path = len(modified_path)
        
        while True:
            choosen_coord_main = modified_path[choosen_index_main]
            choosen_index_sec = len_modified_path - 1
            while True:
                choosen_coord_sec = modified_path[choosen_index_sec]

                if (self.collision_check(self.Node(choosen_coord_main), self.Node(choosen_coord_sec), self.binary_map, self.robot_size)):
                    remodified_path.append(choosen_coord_sec)
                    choosen_index_main = choosen_index_sec
                    break
                else:
                    choosen_index_sec -= 1

            if choosen_index_main >= len_modified_path - 1:
                break
        
        print(remodified_path, flush=True)
        print("[path_generate.generate_final_course] RETURN", flush=True)

        return remodified_path
    
    def display_all_nodes(self):
        plan_image = cv2.imread(self.plan_image_path)

        cv2.circle(plan_image, self.start.coord, radius=5, color=(255, 0, 0), thickness=-1)
        cv2.putText(plan_image, "start", (self.start.coord[0], self.start.coord[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.circle(plan_image, self.end.coord, radius=5, color=(0, 0, 255), thickness=-1)
        cv2.putText(plan_image, "end", (self.end.coord[0], self.end.coord[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        for j in range(0, len(self.node_list)):
            cv2.circle(plan_image, self.node_list[j].coord, radius=1, color=(0, 255, 0), thickness=-1)

        if self.display_rt_plan_image_dimension is not None:
            plan_image = cv2.resize(plan_image, self.display_rt_plan_image_dimension)

        cv2.imshow("Path node distribution", plan_image)
        cv2.waitKey(1)

        cv2.destroyWindow("Path node distribution")
        
        try:
            if cv2.getWindowProperty("Path node distribution", cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow("Path node distribution")
        except cv2.error:
            # Window doesn't exist or already closed
            pass