import pybullet as p
from math import cos, sin
import math
import cv2
import numpy as np
import time

from sensor_msgs.msg import Image
from std_msgs.msg import Header
# import rospy
from cv_bridge import CvBridge

##############################
FOV = 60
CAMERA_START_POSE = [0, 0, 1]
TARGET_START_POSE = [5, 0, 0.48]
HALF_VIEW_RANGE = abs(TARGET_START_POSE[0]-CAMERA_START_POSE[0])*math.tan(math.pi/(180/(FOV/2)))
IMAGE_WIDTH     = 84
IMAGE_HEIGHT    = 84
NUM_FRAMES      = 4
LEFT_SPEED      = 25
RIGHT_SPEED     = 40
AMPLITUDE       = 5
NUM_STEPS       = 200
STEP            = 2*math.pi/NUM_STEPS
LEVEL_GOOD      = 4.0
LEVEL_OKAY      = 5
##############################

p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
right_front_wheel = 2
right_back_wheel = 3
left_front_wheel = 6
left_back_wheel = 7

# set the wheel motors to turn in opposite directions to move forward
maxForce = 500 
max_speed = 100
left_speed = max_speed / 10
right_speed =  max_speed / 4

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
        self.change = None
        self.up_vec     = [0, 0, 1]
        self.width      = IMAGE_WIDTH
        self.height     = IMAGE_HEIGHT
        self.num_frames = NUM_FRAMES
        #self.client     = p.connect(p.GUI)
        self.client = p.connect(p.DIRECT)
        self.gravity    = p.setGravity(0, 0, -9.81)
        self.plane      = plane = p.loadURDF("/home/zheng/bullet3/data/plane_with_restitution.urdf")
        self.robot      = robot = p.loadURDF("/home/zheng/bullet3/data/r2d2.urdf", TARGET_START_POSE.copy(), useFixedBase=False)
        #self.redball    = p.loadURDF("/home/zheng/bullet3/data/sphere2red.urdf", [0,0,0.5])
        self.camera = p.computeViewMatrix(self.camera_pos, self.target_pos, self.up_vec)
        self.projection = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, 
                                                       self.far) # you need this for 
                                                        # the projection
        self.robot_start_pos = p.getBasePositionAndOrientation(self.robot)[0]
        self.robot_start_ori = p.getBasePositionAndOrientation(self.robot)[1]
        self.total_reward = 0
        self.robot_current_step = 0
        self.robot_current_pose = TARGET_START_POSE.copy()
        self.points = self.generate_points_in_circle(0, 0, 5, 10)
        p.setJointMotorControl2(self.robot, right_front_wheel,
                                p.VELOCITY_CONTROL,
                                targetVelocity= right_speed,
                                force=maxForce)
        p.setJointMotorControl2(self.robot, left_front_wheel,
                                p.VELOCITY_CONTROL,
                                targetVelocity= left_speed,
                                force=maxForce)
        p.setJointMotorControl2(self.robot, right_back_wheel,
                                p.VELOCITY_CONTROL,
                                targetVelocity= right_speed,
                                force=maxForce)
        p.setJointMotorControl2(self.robot, left_back_wheel,
                                p.VELOCITY_CONTROL,
                                targetVelocity= left_speed,
                                force=maxForce)

        self.delta_angle = np.pi/45

    """
    Get a raw image from the current camera matrix and process it to a binary image
    There are two images in src folder to show the difference between these two
    """            
    def get_binary_image(self):
        current_camera_matrix    = p.computeViewMatrix(self.camera_pos, self.target_pos, self.up_vec)
        current_projectio_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        raw_img = p.getCameraImage(self.width, self.height, current_camera_matrix, current_projectio_matrix, shadow=False, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
        gray_img        = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        ret, binary_img = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)
        return binary_img
        
    """
    The reward function is actually depends on the absolute distance between robot and camera on 
    Y-axis. Currently we have four stages: Good, Okay, Half_view_range and Bad. The intuition of 
    the reward function is that the reward will be higher when distance is smaller. If the object
    is slip out of Okay level, there will be a big penalty, if the object is out of the view angle,
    there will be a huge penalty which can end the episode directly
    """
    def get_reward(self):
        current_robot_pos = p.getBasePositionAndOrientation(self.robot)[0]
        print(self.is_point_in_fov(self.camera_pos, self.target_pos, current_robot_pos))
        current_robot_y   = current_robot_pos[1]
        current_camera_y  = self.camera_pos[1]
        absolute_distance_y = abs(current_robot_y - current_camera_y)
        frame = self.is_point_in_fov(self.camera_pos, self.target_pos, current_robot_pos)
        if frame == 0:
            return 5
        elif frame == 1:
            return 1
        else:
            return -10
        # if absolute_distance_y <= LEVEL_GOOD:
        #     return 10
        # elif LEVEL_GOOD < absolute_distance_y <= LEVEL_OKAY:
        #     return 5
        # elif LEVEL_OKAY < absolute_distance_y <= HALF_VIEW_RANGE:
        #     return -100
        # else:
        #     return -500
        
    def is_point_in_fov(self, camera_pos, camera_dir, point_pos, fov = FOV):
        # Calculate vector from camera position to point position
        # print(camera_pos, camera_dir, point_pos)
        # print("point: ", camera_pos)
        rel_pos = (point_pos[0] - camera_pos[0], point_pos[1] - camera_pos[1], point_pos[2] - camera_pos[2])
        
        # Calculate angle between camera direction and vector to point
        dot_prod = rel_pos[0] * camera_dir[0] + rel_pos[1] * camera_dir[1] + rel_pos[2] * camera_dir[2]
        mag_dir = math.sqrt(camera_dir[0]**2 + camera_dir[1]**2 + camera_dir[2]**2)
        mag_rel = math.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)
        angle = math.acos(dot_prod / (mag_dir * mag_rel))
        # print("angle: ", angle)
        # Check if angle is within FOV
        if np.rad2deg(angle) <= fov/4: # within the center
            return 0
        elif np.rad2deg(angle) <= fov/2: # within frame but not center
            return 1
        else:
            return 2 # out of frame
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
        self,p.resetBasePositionAndOrientation(self.robot, self.robot_start_pos, self.robot_start_ori)
        # Reset camera pose to start pose
        self.camera_pos = CAMERA_START_POSE.copy()
        # Reset target pose  to start pose
        self.target_pos = TARGET_START_POSE.copy()
        self.camera = p.computeViewMatrix(self.camera_pos, 
                                          self.target_pos, 
                                          self.up_vec)
        # Stack num_frames frames at the original position
        observation = np.zeros((self.width, self.height, self.num_frames), dtype=np.uint8)
        print("observation:", type(observation))
        for i in range(self.num_frames):
            observation[:, :, i] = np.array(self.get_binary_image())
        return np.array(observation)
    
    def generate_points_in_circle(self, center_x, center_y, radius, num_points):
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))
        return points

    
    """
    The robot will move in a sinusodial trajectory in Y-axis, from AMPLITUDE to -AMPLITUDE
    The robot will actually move in a changing speed since each step in Y-axis is different
    """
    def move_robot(self):
        self.robot_current_step += STEP
        self.robot_current_pose[0] =AMPLITUDE * math.cos(self.robot_current_step)
        self.robot_current_pose[1] = AMPLITUDE * math.sin(self.robot_current_step)
        p.resetBasePositionAndOrientation(self.robot, self.robot_current_pose, self.robot_start_ori)
        # p.stepSimulation()
        # Just robot, circular motion
        # get the wheel joints
        # time.sleep(0.01)

    def step(self, action):
        # print("taking step")
        x1 = self.target_pos[0]
        y1 = self.target_pos[1]
        if action == 0:     ## lower bound
            self.change = 0
        elif action == 1:
            self.change = 1
        elif action == 2:     ## upper bound
            self.change = -1
        theta = self.change * self.delta_angle
        # import pdb; pdb.set_trace();
        # from_x_axis = np.arccos(x1)
        # from_y_axis = np.arcsin(y1)
        x2 = cos(theta)* x1 - sin(theta)* y1 
        y2 = sin(theta)* x1 + cos(theta)* y1 
        print("Change ", self.change)
        self.target_pos = [x2, y2, 0.5]
        self.camera = p.computeViewMatrix(self.camera_pos, 
                                          self.target_pos, 
                                          self.up_vec)
        # Capture the observation states with new camera position
        observation = np.zeros((self.width, self.height, self.num_frames), dtype=np.uint8)
        for i in range(self.num_frames):
            observation[:, :, i] = np.array(self.get_binary_image())
        # Get current reward from current state and add it to total_reward
        reward = self.get_reward()
        self.total_reward += reward
        # Set the threshold of terminal states with the total reward 
        if self.total_reward < 0:
            done = True
        elif self.total_reward > 8000:
            done = True
        else:
            done = False
        # Move the robot for each action
        self.move_robot()
        return np.array(observation), reward, done



