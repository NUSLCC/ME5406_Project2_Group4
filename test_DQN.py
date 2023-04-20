"""
Class: ME5406
Author: Liu Chenchen
"""
import tensorflow as tf
import numpy as np
import os
from env import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_FRAMES, NUM_CHANNELS_RGB, NUM_CHANNELS_BINARY
from DQN_rgb import DQNAgent_rgb
from DQN_binary import DQNAgent_binary

TEST_BINARY = True

if TEST_BINARY:
    # Create the agent that has DQN model for binary image
    agent = DQNAgent_binary(state_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_FRAMES*NUM_CHANNELS_BINARY))
    model_path  = "./DQN_weights/dqn_model_binary.h5"
    #model_path  = "./DQN_weights/dqn_binary_4000thresh_5actions_48x48_10&100interval.h5"
else:
    # Create the agent that has DQN model for RGB image
    agent = DQNAgent_rgb(state_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_FRAMES*NUM_CHANNELS_RGB))
    model_path  = "./DQN_weights/dqn_model_rgb.h5"

# Load the model by path
if not os.path.exists(model_path):
    raise ValueError("Model in this path '{}' does not exist.".format(model_path))

model = tf.keras.models.load_model(model_path)
print("Model is loaded successfully!")
# Start testing the model in the env
done = False
state = agent.env.reset()
state_add_none = np.expand_dims(state, axis=0)
total_reward = 0
step_count = 0
action_list = []

while not done:
    predicted_Q_values = model.predict(state_add_none)
    predicted_Q_values_list = predicted_Q_values[0]
    action = np.argmax(predicted_Q_values_list)
    next_state, reward, done = agent.env.step(action)
    next_state_add_none = np.expand_dims(next_state, axis=0)
    state_add_none = next_state_add_none
    total_reward += reward
    step_count += 1
    action_list.append(action)

print("Total reward in test:", total_reward)
print("Total steps in test:", step_count)
print("Action list:", action_list)
