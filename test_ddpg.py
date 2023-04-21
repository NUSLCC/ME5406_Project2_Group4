"""
Class: ME5406
Author: Zheng Jiezhi
Reference: https://keras.io/examples/rl
"""

import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

from env import Env

from DDPG import DDPG


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    env_name = "chaser"
    state_dim = (84, 84, 1)
    has_continuous_action_space = False
    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 10    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for DDPG
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    env = Env()

    # state space dimension
    state_dim = env.reset().shape[0]

    # action space dimension
    # if has_continuous_action_space:
    #     action_dim = env.action_space.shape[0]
    # else:
    #     # action_dim = NUM_ACTIONS
    #     pass
    # action_dim = env.action_space.shape[0]
    action_dim = 3
    # initialize a DDPG agent
    ddpg_agent = DDPG(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num
   
    directory = "DDPG_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "DDPG_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)
    #  DDPG_preTrained/chaser/DDPG_chaser_0_0.pth
    # ddpg_agent.load("/home/thebird/repos/me5406_group4_backup_2/DDPG_preTrained/chaser/DDPG_chaser_0_0.pth")
    ddpg_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            action = ddpg_agent.select_action(np.transpose(state, (2, 0, 1)))
            state, reward, done = env.step(action.item())
            ep_reward += reward

            if done:
                break

        # clear buffer
        ddpg_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    # env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':

    test()