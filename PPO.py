"""
Class: ME5406
Author: Ravi Girish
Reference: https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
"""
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from collections import deque
import random
import numpy as np

MINIBATCH_SIZE = 1000
################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
'''
Replay buffer is different from the original code. It now support for sampling state action parirs
from the buffer that can be used to train the model. 
'''
class RolloutBuffer:
    
    def __init__(self):
        self.replay_buffer_size = 10000
        self.actions = deque(maxlen=self.replay_buffer_size)
        self.states = deque(maxlen=self.replay_buffer_size)
        self.logprobs = deque(maxlen=self.replay_buffer_size)
        self.rewards = deque(maxlen=self.replay_buffer_size)
        self.state_values = deque(maxlen=self.replay_buffer_size)
        self.is_terminals = deque(maxlen=self.replay_buffer_size)
    
    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        '''
        This the actor network. It has been modified from the original code to take in the input of 
        (NUM_FRAMES, NUM_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT) and output (1, NUM_ACTIONS) tensor. From
        the output the best action is the index with the highest value. 
        '''
        self.actor = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=8, stride=4),
                nn.Tanh(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.Tanh(),
                nn.Conv2d(64, 1, kernel_size=3, stride=1),
                nn.Flatten(),
                nn.Linear(49, action_dim),
                nn.Softmax(dim=-1)
            )
        '''
        This is the critic network. It takes in the states and outputs the q_values of the states. 
        '''
        self.critic = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=8, stride=4),
                    nn.Tanh(),
                    nn.Conv2d(16, 5, kernel_size=2, stride=4),
                    nn.Conv2d(5, 1, kernel_size=5, stride=1),
                    nn.Flatten(0),
                    nn.Softmax(dim=-1)
                )
        ''' This is to ensure that the values are floating point type to allow the tensor to run the loss function. '''
        self.critic.float()
    
    '''
    This function returns the action for the input state.
    '''
    def act(self, state):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    '''
    This function evalutes the state-action pairs and returns the log-probabilities. 
    '''
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
        state_values = self.critic(torch.unsqueeze(state, dim=1))
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.minibatch_size = MINIBATCH_SIZE
        # Initialize the replay buffer. 
        self.buffer = RolloutBuffer()
        # Initialize the Actor-Critic PPO network
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss().float()

    '''
    Select action.
    '''
    def select_action(self, state):
        
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action[0]
    
    '''
    Class method to sample a batch of states/actions from the buffer for evaluation.
    '''
    def sample_minibatch(self):
        list_of_samples_with_index = random.sample(list(enumerate(self.buffer.states)), self.minibatch_size)
        states = []
        actions = []
        logprobs = []
        state_values = []
        rewards = []
        is_terminals = []
        for i, tensor_ in list_of_samples_with_index:
            states.append(tensor_)
            actions.append(self.buffer.actions[i])
            logprobs.append(self.buffer.logprobs[i])
            state_values.append(self.buffer.state_values[i])
            rewards.append(self.buffer.rewards[i])
            is_terminals.append(self.buffer.is_terminals[i])
        return states, actions, logprobs, state_values, rewards, is_terminals

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            # Standard Update function
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        states, actions, logprobs, state_values, rewards, is_terminals = self.sample_minibatch()        
        old_states = torch.squeeze(torch.stack(states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(state_values, dim=0)).detach().to(device)
        rewards = torch.tensor(rewards).detach().to(device)
        old_state_values = torch.tensor(old_state_values).detach().to(device)
        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

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
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.float(), rewards.float()) - 0.01 * dist_entropy
            
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