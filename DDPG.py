"""
Class: ME5406
Author: Zheng Jiezhi
Reference: https://github.com/DLR-RM/stable-baselines3 
reference from PPO.py
"""

import torch
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import tensorflow as tf
from env_circle import Env

################################## set device ##################################
# set device to cpu or cuda
device = torch.device('cpu')
################################## set device ##################################

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(Actor, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space   ## ensure no discrete output
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # TODO
        self.actor = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=8, stride=4),
                nn.Tanh(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.Tanh(),
                nn.Conv2d(64, 1, kernel_size=3, stride=1),
                nn.Flatten(),
                # nn.Linear(49, 512),
                # nn.Tanh(),
                nn.Linear(49, action_dim),
                # nn.Flatten(),
                nn.Softmax(dim=-1)
        )

    def forward(self, state):
        state = state
        
        policy_dist = nn.relu(self.actor(state))   ## another method is using softmax
        # policy_dist = nn.softmax(self.actor_linear2(policy_dist), dim=1)

        return policy_dist
    
    def act(self, state):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        # print(state.view(state.size(0), -1))
        
        state_val = self.actor(state)
        # import pdb; pdb.set_trace();

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(torch.unsqueeze(state, dim=1))
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.actor(torch.unsqueeze(state, dim=1))
        
        return action_logprobs, state_values, dist_entropy

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        self.has_continuous_action_space = has_continuous_action_space   ## ensure no discrete output
        super(Critic, self).__init__()
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # TODO
        self.critic = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=8, stride=4),
                nn.Tanh(),
                nn.Conv2d(16, 5, kernel_size=2, stride=4),
                nn.Linear(5, 1),
                nn.Flatten(0),
                nn.Softmax(dim=-1)
            )
        
    def forward(self, state):
        state = state
        value = nn.relu(self.critic(state))
        return value
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        # import pdb; pdb.set_trace()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class DDPG:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):
        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_std = action_std_init
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()

        self.policy = Actor(state_dim, action_dim, has_continuous_action_space, action_std_init)
        self.value = Critic(state_dim, action_dim, has_continuous_action_space, action_std_init)
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.value.critic.parameters(), 'lr': lr_critic}
                    ])
        self.policy_old = Actor(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())   # replace the old policy
        self.MseLoss = nn.MSELoss()  # loss function

    def critic_target_net():  ## get target q
        state, _, _ = Env.step()
        

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")



    def select_action(self, state):
        
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        # print(action)
        # return action.item()
        return action[0]
    
    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        
        # calculate advantages
        print("check reward size:", rewards[0].shape)
        print("check state values size:", old_state_values[0][0].shape)
        advantages = rewards.detach()[0] - old_state_values.detach()[0][0]

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            print("surr1:", surr1.shape)
            print("surr2:", surr2.shape)
            print("state values:", state_values.shape)
            print("reward:", reward.shape)
            print("dist entropy:", dist_entropy.shape)
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy[0][0]
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
