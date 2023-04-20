"""
Class: ME5406
Author: Liu Chenchen
"""
import time
from env import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS_RGB, NUM_CHANNELS_BINARY, NUM_FRAMES
from DQN_rgb import DQNAgent_rgb
from DQN_binary import DQNAgent_binary

TRAIN_BINARY = False
TRAIN_EPS = 500

if TRAIN_BINARY:
    # Create the agent that train DQN model with binary images
    agent = DQNAgent_binary(state_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_FRAMES*NUM_CHANNELS_BINARY))
else:
    # Create the agent that train DQN model with RGB images
    agent = DQNAgent_rgb(state_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_FRAMES*NUM_CHANNELS_RGB))

# Train the model inside the agent
start_time = time.time()
agent.train(num_episodes=TRAIN_EPS, max_steps_per_episode=10000)
end_time = time.time()
time_cost = end_time - start_time
print("Training Time taken:", time_cost, "seconds")

# Output the model
agent.output_model()
print("The model is generated!")