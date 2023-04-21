"""
Class: ME5406
Author: Liu Chenchen
"""
import math
import time
AMPLITUDE = 4
NUM_STEPS       = 100
STEP            = 2*math.pi/NUM_STEPS
robot_current_step = 0
old_step_len = 0
for i in range (200):
    robot_current_step += STEP
    step_len = AMPLITUDE * math.sin(robot_current_step)
    print(abs(step_len-old_step_len))
    time.sleep(0.01)
    old_step_len = step_len
