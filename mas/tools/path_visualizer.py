import pygame
import math
import numpy as np

class PathWindow:
    def __init__(self, width, height, obstacle_list, scaling_factor=0.8):
        self.scaling_factor=scaling_factor
        self.WIDTH = int(width * scaling_factor)
        self.HEIGHT = int(height * scaling_factor)
        self.obstacle_list = (np.array(obstacle_list) * self.scaling_factor).astype(int)
        self.robot_triangle_radius = int(20 * scaling_factor)

        self.background_color = (0, 0, 0)
        self.obstacle_color = (255, 255, 255)
        self.robot_color = (255, 0, 0)
        self.planned_path_color = (40, 222, 235)
        self.rt_path_color = (240, 236, 5)

        self.close_window_message = False

        # keyboard event flags
        self.key_Y_pressed = False

    def update_obstacle_list(self, obstacle_list, door_list, door_status):
        iter = 0
        for (x, y, w, h) in door_list:
            if door_status[iter] == False:
                obstacle_list = np.append(obstacle_list, [door_list[iter]], axis=0)    # if door is closed, add it to the obstacle list
            iter += 1
        self.obstacle_list = (np.array(obstacle_list) * self.scaling_factor).astype(int)

    def start_window(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Robot trajectory")

        # Fill screen with black color
        self.screen.fill(self.background_color)
        # Update display
        pygame.display.flip()

    def update_window(self, robot_current_pos, robot_current_yaw, planned_path, rt_path):
        # Fill screen with black color
        self.screen.fill(self.background_color)

        # draw the obstacles
        for obstacle in self.obstacle_list:
            pygame.draw.rect(self.screen, self.obstacle_color, obstacle)

        # draw the robot
        self.draw_robot(robot_current_pos, robot_current_yaw, self.robot_triangle_radius)

        # draw the planned path
        if planned_path is not None:
            self.draw_paths(planned_path, rt_path)

        # flip the display
        # flipped_surface = pygame.transform.flip(self.screen, False, True)  # Flip vertically
        # self.screen.blit(flipped_surface, (0, 0))
        pygame.display.update()
        # pygame.display.flip()

    def draw_robot(self, current_pos, yaw, size):
        robot_x = current_pos[0] * self.scaling_factor
        robot_y = current_pos[1] * self.scaling_factor
        robot_radius = size
        robot_yaw = math.radians(yaw)

        head_point = (int(robot_x + robot_radius * math.sin(robot_yaw)), int(robot_y + robot_radius * math.cos(robot_yaw)))
        tail_point_1 = (int(robot_x + robot_radius * math.sin(robot_yaw + math.radians(150))), int(robot_y + robot_radius * math.cos(robot_yaw + math.radians(150))))
        tail_point_2 = (int(robot_x + robot_radius * math.sin(robot_yaw + math.radians(-150))), int(robot_y + robot_radius * math.cos(robot_yaw + math.radians(-150))))

        pygame.draw.polygon(self.screen, self.robot_color, [head_point, tail_point_1, tail_point_2])

    def draw_paths(self, planned_path, rt_path):
        planned_path = (np.array(planned_path) * self.scaling_factor).astype(int)
        rt_path = (np.array(rt_path) * self.scaling_factor).astype(int)

        if len(planned_path) > 1:
            for i in range(len(planned_path) - 1):
                pygame.draw.line(self.screen, self.planned_path_color, planned_path[i], planned_path[i + 1], 2)

        if len(rt_path) > 6:
            for i in range(0, len(rt_path) - 1, 5):
                pygame.draw.line(self.screen, self.rt_path_color, rt_path[i], rt_path[i + 1], 2)

    def keyboard_event_check(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close_window_message = True
                self.close_window()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close_window_message = True
                    self.close_window()
                if event.key == pygame.K_y:
                    self.key_Y_pressed = True
            else:
                self.key_Y_pressed = False    

    def close_window(self):
        pygame.quit()
