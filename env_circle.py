"""
Class: ME5406
Author: Ravi Girish
"""
import pybullet as p
import time
import cv2
from math import cos, sin
import math
import numpy as np

##############################
FOV = 60
CAMERA_START_POSE   = [0, 0, 1]
TARGET_START_POSE   = [5, 0, 0.48]
HALF_VIEW_RANGE     = abs(TARGET_START_POSE[0]-CAMERA_START_POSE[0])*math.tan(math.pi/(180/(FOV/2)))
IMAGE_WIDTH         = 84
IMAGE_HEIGHT        = 84
NUM_FRAMES          = 1
NUM_CHANNELS_BINARY = 1
NUM_CHANNELS_RGB    = 4
LEFT_SPEED          = 25
RIGHT_SPEED         = 40
AMPLITUDE           = 5
NUM_STEPS           = 300
STEP                = 2 * math.pi / NUM_STEPS
DELTA_ANGLE         = 2 * np.pi / NUM_STEPS
##############################

class Env:

    def __init__(self):
        # Set up camera
        self.fov = FOV
        self.aspect = 1.0
        self.near = 0.01
        self.far = 10
        self.action = 0
        self.camera_pos = CAMERA_START_POSE.copy()
        self.target_pos = TARGET_START_POSE.copy()
        self.change = 0
        self.up_vec     = [0, 0, 1]
        self.width      = IMAGE_WIDTH
        self.height     = IMAGE_HEIGHT
        self.num_frames = NUM_FRAMES
        self.delta_angle = DELTA_ANGLE
        #self.client     = p.connect(p.GUI)
        self.client = p.connect(p.DIRECT)
        self.gravity    = p.setGravity(0, 0, -9.81)
        self.plane      = plane = p.loadURDF("/home/thebird/repos/bullet3/data/plane_with_restitution.urdf")
        self.robot      = robot = p.loadURDF("/home/thebird/repos/bullet3/data/r2d2.urdf", TARGET_START_POSE.copy(),
                           useFixedBase=False)
        self.camera = p.computeViewMatrix(self.camera_pos, self.target_pos, self.up_vec)
        self.projection = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, 
                                                       self.far) # you need this for 
                                                        # the projection
        self.robot_start_pos = p.getBasePositionAndOrientation(self.robot)[0]
        self.robot_start_ori = p.getBasePositionAndOrientation(self.robot)[1]
        self.total_reward = 0
        self.robot_current_step = 0
        self.robot_current_pose = TARGET_START_POSE.copy()
        self.direction = 0
        self.is_binary = True
        self.out_of_frame = False


    """
    Get a raw RGB image from the current camera matrix and process it to a binary image
    If the Model trains on RGB image, it should set is_binary to False
    There are two images in src folder to show the difference between these two
    """            
    def get_image(self):
        current_camera_matrix    = p.computeViewMatrix(self.camera_pos, self.target_pos, self.up_vec)
        current_projectio_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        rgb_img = p.getCameraImage(self.width, self.height, current_camera_matrix, current_projectio_matrix, shadow=False, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
        gray_img                 = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        ret, binary_img          = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)
        if self.is_binary:
            return binary_img
        else:
            return rgb_img
        
    """
    The reward function is actually depends on the absolute distance between robot and camera on 
    Y-axis. Currently we have four stages: Good, Okay, Half_view_range and Bad. The intuition of 
    the reward function is that the reward will be higher when distance is smaller. If the object
    is slip out of Okay level, there will be a big penalty, if the object is out of the view angle,
    there will be a huge penalty which can end the episode directly
    """
    def get_reward(self):
        current_robot_pos = p.getBasePositionAndOrientation(self.robot)[0]
        # print(self.is_point_in_fov(self.camera_pos, self.target_pos, current_robot_pos))
        current_robot_y   = current_robot_pos[1]
        current_camera_y  = self.camera_pos[1]
        absolute_distance_y = abs(current_robot_y - current_camera_y)
        frame = self.is_point_in_fov(self.camera_pos, self.target_pos, current_robot_pos)
        if frame == 0:
            return 40
        elif frame == 1:
            return 10
        elif frame == 2:
            return 0
        else:
            self.out_of_frame = True
            return -1000
        
    def get_observation(self):
        if self.is_binary:
            observation = np.zeros((self.height, self.width, self.num_frames), dtype=np.uint8)
            for i in range(self.num_frames):
                observation[:, :, i] = np.array(self.get_image())
                return observation
        else:
            observation = np.zeros((self.height, self.width, self.num_channels, self.num_frames), dtype=np.uint8)
            for i in range(self.num_frames):
                observation[:, :, :, i] = np.array(self.get_image())
                return observation
        
    def is_point_in_fov(self, camera_pos, camera_dir, point_pos, fov = FOV):
        # Calculate vector from camera position to point position
        rel_pos = (point_pos[0] - camera_pos[0], point_pos[1] - camera_pos[1], point_pos[2] - camera_pos[2])
        
        # Calculate angle between camera direction and vector to point
        dot_prod = rel_pos[0] * camera_dir[0] + rel_pos[1] * camera_dir[1] + rel_pos[2] * camera_dir[2]
        mag_dir = math.sqrt(camera_dir[0]**2 + camera_dir[1]**2 + camera_dir[2]**2)
        mag_rel = math.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)
        angle = math.acos(dot_prod / (mag_dir * mag_rel))
        # Check if angle is within FOV
        if np.rad2deg(angle) <= fov/10: # within the center +/- 6 degs
            return 0
        elif np.rad2deg(angle) <= fov/4: # within frame but not center +/- 30 degs
            return 1
        elif np.rad2deg(angle) <= fov/2: # within frame but not center +/- 30 degs
            return 2
        else:
            return 3 # out of frame
    """
    The reset function will set every variable to its original state. Both the robot, virtual camera 
    and red ball will be reset to its original position. Then the observation will be the stack of
    images at the initial state
    """   
    def reset(self):
        # Set cumulative reward to 0
        self.total_reward = 0
        # Set current robot step to 0
        self.robot_current_step = 0
        # Set current robot pose to start pose
        # self.robot_current_pose = TARGET_START_POSE.copy()
        # Reset robot to start position
        self,p.resetBasePositionAndOrientation(self.robot, self.robot_start_pos, self.robot_start_ori)
        # Reset camera pose to start pose
        self.camera_pos = CAMERA_START_POSE.copy()
        # Reset target pose to start pose
        # self.target_pos = TARGET_START_POSE.copy()
        self.target_pos = self.robot_current_pose
        self.camera = p.computeViewMatrix(self.camera_pos, 
                                          self.target_pos, 
                                          self.up_vec)
        # Stack num_frames frames at the original position
        observation = self.get_observation()
        return np.array(observation)

    
    """
    The robot moves in a circular trajectory with the origin (0,0,0) in the center. 
    """
    def move_robot(self):
        """
        This is so that every 50, the target has a random chance of changing direction. 
        This adds a level of uncertainty that can be certaintity.
        """
        if(self.direction % 50 == 0):
            if(np.random.randint(2)):
                global STEP
                STEP = STEP * -1
        self.direction += 1
        self.robot_current_step += STEP
        self.robot_current_pose[0] = AMPLITUDE * math.cos(self.robot_current_step)
        self.robot_current_pose[1] = AMPLITUDE * math.sin(self.robot_current_step)
        p.resetBasePositionAndOrientation(self.robot, self.robot_current_pose, self.robot_start_ori)

    def step(self, action):
        x1 = self.target_pos[0]
        y1 = self.target_pos[1]
        if action == 0:
            self.change = 0
        elif action == 1:
            self.change = 1
        elif action == 2:
            self.change = -1
        theta = self.change * self.delta_angle
        x2 = cos(theta)* x1 - sin(theta)* y1 
        y2 = sin(theta)* x1 + cos(theta)* y1 
        self.target_pos = [x2, y2, 0.5]
        self.camera = p.computeViewMatrix(self.camera_pos, 
                                          self.target_pos, 
                                          self.up_vec)
        # Capture the observation states with new camera position
        observation = self.get_observation()
        # Get current reward from current state and add it to total_reward
        reward = self.get_reward()
        self.total_reward += reward
        # Set the threshold of terminal states with the total reward 
        if self.total_reward < 0:
            done = True
        elif self.total_reward > 4000:
            done = True
        else:
            done = False
        if self.out_of_frame:
            done = True
        self.out_of_frame = False
        # Move the robot for each action
        self.move_robot()
        return np.array(observation), reward, done
    
