"""
Class: ME5406
Author: Liu Chenchen
"""
import tensorflow as tf
import numpy as np
from collections import deque
import random
import visualkeras
from env import Env, NUM_ACTIONS

# log_path  = "/home/lcc/me5406_part2/me5406-project-2/src/me5406/src/log/"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0)

class DQNAgent_binary:
    """
    Initialize the class with some parameters and two models
    """
    def __init__(self, state_shape):
        self.env = Env()
        self.env.is_binary = True
        self.state_shape = state_shape
        self.num_actions = NUM_ACTIONS
        self.learning_rate = 0.025
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay_rate = 0.99
        self.replay_buffer_size = 10000
        self.minibatch_size = 32
        self.online_model = self.build_model()
        self.target_model = self.build_model()
        self.online_model_update_interval = 10
        self.target_model_update_interval = 100
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.save_path = "./DQN_weights/"

    """
    Whether you want to output the visualization of the model
    """
    def visualize_model(self, switch=False):
        if switch == True:
            visualkeras.layered_view(self.online_model, 
                                     to_file='graphic_model_binary.png', 
                                     legend=True)
            tf.keras.utils.plot_model(
                self.online_model,
                to_file="verbal_model_binary.png",
                show_shapes=True,
                show_dtype=False,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                dpi=96,
                layer_range=None,
                show_layer_activations=True)

    """
    Create a CNN model with 7 layers
    """
    def build_model(self):
        # Create a linear stack of layers
        model = tf.keras.models.Sequential()
        # Add the first convolutional layer with 32 filters with a kernel size of 3x3 and a stride of 1
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', input_shape=self.state_shape))
        # Add a max pooling layer with size 2
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        # Add a second convolutional layer with 64 filters with a kernel size of 3x3 and a stride of 1
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
        # Add another max pooling layer with size 2
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        # Flatten the output from the convolutional layers to a 1D tensor
        model.add(tf.keras.layers.Flatten())
        # Add a full connected dense layer with 128 neurons with RELU activation function 
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        # Add an output layer with num_actions neurons to learn possible Q-values that can be positive or negative
        model.add(tf.keras.layers.Dense(self.num_actions, activation='linear'))
        # Use the mean squared error (MSE) loss function with the Adam optimizer
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        #model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate, rho=0.95, epsilon=1e-07)),loss='mse')
        return model
    
    """
    Output the DQN model
    """
    def output_model(self):
        self.online_model.save(self.save_path + "dqn_model_binary.h5")
        self.online_model.summary()

    """
    Output the DQN model weights
    """
    def output_model_weights(self):
        self.online_model.save_weights(self.save_path + "dqn_weights_binary.h5")
        print("Output the weights")

    """
    Get action by using epsilon greedy policy
    """
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            # Add one dimension to the states as the input shape should be (none, width, height, num_frames)
            state_add_none = np.expand_dims(state, axis=0)
            predicted_Q_values = self.online_model.predict(state_add_none)
            Q_values_list = predicted_Q_values[0]
            return np.argmax(Q_values_list)

    """
    Sample a mini batch of transitions randomly from the replay buffer
    """
    def sample_minibatch(self):
        minibatch = random.sample(self.replay_buffer, self.minibatch_size)
        # The format of each transition is (state, action, reward, next_state, done)
        states      = np.array([transition[0] for transition in minibatch])
        actions     = np.array([transition[1] for transition in minibatch])
        rewards     = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones       = np.array([transition[4] for transition in minibatch])
        return states, actions, rewards, next_states, dones

    """
    Update the online model in some smaller step interval
    """
    def update_online_model(self):
        # Sample a random minibatch of trasitions from the replay buffer
        states, actions, rewards, next_states, dones = self.sample_minibatch()
        # Predict the Q-values for the current states-actions pair by online model
        Q_values         = self.online_model.predict(states)
        # Predict the Q-values for the next states-actions pair by target model
        Q_values_next    = self.target_model.predict(next_states)
        # Create the target Q-values for each state-action pair with target model
        Q_values_targets = np.copy(Q_values)
        for i in range(self.minibatch_size):
            if dones[i]:
                # For terminal states, update Q_values_targets with rewards
                Q_values_targets[i, actions[i]] = rewards[i]
            else:
                # For non-terminal states, update Q_values_targets with Q-learning rule
                # Here we use Q_values_next predicted by target model instead of online model
                Q_values_targets[i, actions[i]] = rewards[i] + self.gamma * np.max(Q_values_next[i])
        # Update the model using the current states from online model and target Q-values from target model
        self.online_model.fit(states, Q_values_targets, verbose=0)# callbacks=[tensorboard_callback])

    """
    Update the target model in some bigger step interval
    """
    def update_target_model(self):
        self.target_model.set_weights(self.online_model.get_weights())

    """
    Train the model with assigned number of episodes and max steps for each episode
    """
    def train(self, num_episodes, max_steps_per_episode):
        self.update_target_model()
        for episode in range(num_episodes):
            # Get original state by reseting the environment
            state = self.env.reset()
            total_reward = 0
            for step in range(max_steps_per_episode):
                # Get action by epsilon greedy policy
                action = self.get_action(state)
                # Take this action in the env
                next_state, reward, done = self.env.step(action)
                # Add this transition to the replay buffer 
                self.replay_buffer.append((state, action, reward, next_state, done))
                # Update the online model at some smaller interval
                # Need to make sure the replay buffer is larger than the size of minibatch
                if len(self.replay_buffer) >= self.minibatch_size and step % self.online_model_update_interval == 0:
                    self.update_online_model()
                # Update the target network at some larger interval
                if len(self.replay_buffer) >= self.minibatch_size and step % self.target_model_update_interval == 0:
                    self.update_target_model()
                # Update the state and total reward
                state = next_state
                total_reward += reward
                # End the episode if agent meets terminal state
                if done:
                    break
            # Decay the epsilon in each episode iteratively till epsilon_min
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)
            print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {self.epsilon:.4f}")

