"""
Class: ME5406
Author: Liu Chenchen
"""
import pybullet as p
import math
import cv2
import numpy as np

##############################
FOV = 60
CAMERA_START_POSE   = [0, 0, 1]
TARGET_START_POSE   = [5, 0, 0.48]
HALF_VIEW_RANGE     = abs(TARGET_START_POSE[0]-CAMERA_START_POSE[0])*math.tan(math.pi/(180/(FOV/2)))
IMAGE_HEIGHT        = 48
IMAGE_WIDTH         = 48
NUM_FRAMES          = 1
NUM_CHANNELS_BINARY = 1
NUM_CHANNELS_RGB    = 4
AMPLITUDE           = 4
NUM_STEPS           = 100
STEP                = 2*math.pi/NUM_STEPS
LEVEL_GOOD          = 0.6
LEVEL_OKAY          = 1.0
NUM_ACTIONS         = 5
DONE_THRESHOLD      = 4000
##############################

class Env:
    def __init__(self):
        self.fov = FOV
        self.aspect = 1.0
        self.near = 0.01
        self.far = 10
        self.camera_pos = CAMERA_START_POSE.copy()
        self.target_pos = TARGET_START_POSE.copy()
        self.up_vec     = [0, 0, 1]
        self.width      = IMAGE_WIDTH
        self.height     = IMAGE_HEIGHT
        self.num_frames = NUM_FRAMES
        self.num_channels = NUM_CHANNELS_RGB
        self.client     = p.connect(p.GUI)
        self.gravity    = p.setGravity(0, 0, -9.81)
        self.plane      = p.loadURDF("/home/lcc/me5406_part2/me5406-project-2/urdf/data/plane100.urdf")
        self.robot      = p.loadURDF("/home/lcc/me5406_part2/me5406-project-2/urdf/data/r2d2.urdf", TARGET_START_POSE.copy(), useFixedBase=False)
        self.redball    = p.loadURDF("/home/lcc/me5406_part2/me5406-project-2/urdf/data/sphere2red.urdf", [0,0,0.5])
        self.robot_start_pos = p.getBasePositionAndOrientation(self.robot)[0]
        self.robot_start_ori = p.getBasePositionAndOrientation(self.robot)[1]
        self.total_reward = 0
        self.robot_current_step = 0
        self.robot_current_pose = TARGET_START_POSE.copy()
        self.is_binary = True

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

    """
    The robot will move in a sinusodial trajectory in Y-axis, from AMPLITUDE to -AMPLITUDE
    The robot will actually move in a changing speed since each step in Y-axis is different
    """
    def move_robot(self):
        self.robot_current_step += STEP
        self.robot_current_pose[1] = AMPLITUDE * math.sin(self.robot_current_step)
        p.resetBasePositionAndOrientation(self.robot, self.robot_current_pose, self.robot_start_ori)
    
    """
    The red ball is the symbol of the camera as its location will sync with the camera which 
    can help us identify camera's realtime position. But the red ball has a deviation of -0.6
    in X-axis other wise the ball will block camera's view. Here we do not change the orientation
    """
    def move_camera_redball(self):
        redball_pos = [self.camera_pos[0]-0.6, self.camera_pos[1], 0.5]
        p.resetBasePositionAndOrientation(self.redball, redball_pos, self.robot_start_ori)

    """
    The reward function is actually depends on the absolute distance between robot and camera on 
    Y-axis. Currently we have four stages: Good, Okay, Half_view_range and Bad. The intuition of 
    the reward function is that the reward will be higher when distance is smaller. If the object
    is slip out of Okay level, there will be a big penalty, if the object is out of the view angle,
    there will be a huge penalty which can end the episode directly
    """
    def get_reward(self):
        current_robot_pos = p.getBasePositionAndOrientation(self.robot)[0]
        current_robot_y   = current_robot_pos[1]
        current_camera_y  = self.camera_pos[1]
        absolute_distance_y = abs(current_robot_y - current_camera_y)
        if absolute_distance_y <= LEVEL_GOOD:
            return 40
        elif LEVEL_GOOD < absolute_distance_y <= LEVEL_OKAY:
            return 5
        elif LEVEL_OKAY < absolute_distance_y <= HALF_VIEW_RANGE:
            return 0
        else:
            return -1000

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
        self.robot_current_pose = TARGET_START_POSE.copy()
        # Reset robot to start position
        p.resetBasePositionAndOrientation(self.robot, self.robot_start_pos, self.robot_start_ori)
        # Reset camera pose to start pose
        self.camera_pos = CAMERA_START_POSE.copy()
        # Reset target pose  to start pose
        self.target_pos = TARGET_START_POSE.copy()
        # Reset red ball model to start pose
        self.move_camera_redball()
        # Stack num_frames frames at the original position
        observation = self.get_observation()
        return np.array(observation)

    """
    The step function will have five actions, start from 0 to 4. 0 means no change in camera's
    position; 1 & 2 means 0.05 & -0.05 deviation in Y-axis; 3 & 4 means 0.1 & -0.1 deviation in 
    Y-axis; no deviation if other actions are input. Please know that you can change the 
    action_value if you want better performance in training.
    """
    def step(self, action):
        if action == 0:
            action_value = 0
        elif action == 1:
            action_value = 0.10
        elif action == 2:
            action_value = -0.10
        elif action == 3:
            action_value = 0.20
        elif action == 4:
            action_value = -0.20
        # elif action == 5:
        #     action_value = 0.2
        # elif action == 6:
        #     action_value = -0.2
        else:
            action_value = 0
        # Move virtual camera position with deviation selected by action value
        self.camera_pos[1] += action_value
        # Update virtual camera target position with changed camera position (only x and y)
        self.target_pos[0] = self.camera_pos[0] + TARGET_START_POSE.copy()[0]
        self.target_pos[1] = self.camera_pos[1] + TARGET_START_POSE.copy()[1]
        # Move the red ball according to camera's new position
        self.move_camera_redball()
        # Capture the observation states with new camera position
        observation = self.get_observation()
        # Get current reward from current state and add it to total_reward
        reward = self.get_reward()
        self.total_reward += reward
        # Set the threshold of terminal states with the total reward 
        if self.total_reward < 0:
            done = True
        elif self.total_reward > DONE_THRESHOLD:
            done = True
        else:
            done = False
        # Move the robot for each action
        self.move_robot()
        return np.array(observation), reward, done
