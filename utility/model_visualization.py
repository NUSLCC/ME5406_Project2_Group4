"""
Class: ME5406
Author: Liu Chenchen
"""
from env import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS_RGB, NUM_CHANNELS_BINARY, NUM_FRAMES
from DQN_rgb import DQNAgent_rgb
from DQN_binary import DQNAgent_binary

TEST_BINARY = False
if TEST_BINARY:
    agent = DQNAgent_binary(state_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_FRAMES*NUM_CHANNELS_BINARY))
else:
    agent = DQNAgent_rgb(state_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_FRAMES*NUM_CHANNELS_RGB))

# This will print model structure images to the execution path of the script
agent.visualize_model(True)
