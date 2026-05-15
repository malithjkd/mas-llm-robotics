import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class Binary_map_generator:
    def __init__(self):
        self.width = 2230    # map width is 2220
        self.height = 1200   # map height is 1170
        self.obstacle_list = [                                              # x, y, w, h
                            (0, 0, 0.5, 39),                                # wall 1
                            (73.5, 0, 0.5, 39),                             # wall 2
                            (0, 0, 74, 0.5),                                # wall 3
                            (0.5, 31.5, 0.5, 1),                            # wall 4
                            (4.5, 31.5, 45.5, 1),                           # wall 5
                            (53.5, 31.5, 20, 1),                            # wall 6
                            (30.5, 17, 0.5, 14.5),                          # wall 7
                            (49, 17, 0.5, 14.5),                            # wall 8
                            (30.5, 13, 0.5, 0.5),                           # wall 9
                            (49, 0, 0.5, 13.5),                             # wall 10
                            (0, 38.5, 74, 0.5),                             # wall 11

                            (0.5, 0.5, 3, 21),                              # table 1
                            (0.5, 0.5, 12, 3),                              # table 2
                            (10.9, 21.5, 19.6, 10),                         # dr. ruwanthika's office
                            (10.9, 18.5, 12, 3),                            # table 4
                            (24, 0, 25, 13),                                # msc and wasantha aiya's office
                            (31, 21.5, 4, 10),                              # table 6
                            (31, 27.5, 18, 4),                              # table 7
                            (45, 21.5, 4, 10),                              # table 8
                            (49.5, 0.5, 24, 12.5),                          # table 9
                            (66.5, 0.5, 7, 31),                             # table 10
                            (53.5, 24.5, 20, 7),                            # table 11
                            (9.5, 0, 21, 13),                               # test square covering beam bag
                            ]  
        
        self.door_list =    [
                            (1, 31.5, 3.5, 1),                              # door 1
                            (30.5, 13.5, 0.5, 3.5),                         # door 2
                            (49, 13.5, 0.5, 3.5),                           # door 3
                            (50, 31.5, 3.5, 1),                             # door 4
                            ]
        
        self.doors_status = [False, False, False, False]  # all doors are closed at the beginning

        self.obstacle_list = (np.array(self.obstacle_list) * 30).astype(np.int32)
        self.door_list = (np.array(self.door_list) * 30).astype(np.int32)

        self.base_path = os.path.dirname(__file__)
        
        self.binary_map_save_location = os.path.join(self.base_path, r"properties\binary_map\binary_map.png")
        self.current_binary_map_save_location = os.path.join(self.base_path, r"properties\binary_map\binary_map_current.png")              
                                
    def generate_basic_binary_map(self):
        # create a blank image
        binary_map = 255 * np.zeros((self.height, self.width), np.uint8)

        # draw obstacles
        for (x, y, w, h) in self.obstacle_list:
            cv2.rectangle(binary_map, (x, y), (x + w, y + h), 255, -1)

        # draw all the doors closed
        for (x, y, w, h) in self.door_list:
            cv2.rectangle(binary_map, (x, y), (x + w, y + h), 255, -1)    # door is closed

        return binary_map
    
    def update_door_status(self, door_num, status):
        binary_map = self.load_binary_map()
        self.doors_status = [False, False, False, False]
        
        self.doors_status[door_num - 1] = status
        (x, y, w, h)  = self.door_list[door_num - 1]

        cv2.rectangle(binary_map, (x, y), (x + w, y + h), 0, -1)    # open the door in the map

        self.save_current_binary_map(binary_map)

        return binary_map
    
    def save_binary_map(self, binary_map):
        cv2.imwrite(self.binary_map_save_location, binary_map)
        cv2.imwrite(self.current_binary_map_save_location, binary_map)

    def save_current_binary_map(self, binary_map):
        cv2.imwrite(self.current_binary_map_save_location, binary_map)

    def load_binary_map(self):
        binary_map = cv2.imread(self.binary_map_save_location, cv2.IMREAD_GRAYSCALE)  
        
        return binary_map  

    def display_binary_map(self, binary_map):
        plt.imshow(binary_map, cmap='gray')
        plt.show()