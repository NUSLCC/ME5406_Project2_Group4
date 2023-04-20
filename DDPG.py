"""
Class: ME5406
Author: Zheng Jiezhi
Reference: https://keras.io/examples/rl/ddpg_pendulum/
"""

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import tensorflow as tf
from env import Env, NUM_ACTIONS

################################## set device ##################################
# set device to cpu or cuda
device = torch.device('cpu')

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
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
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
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
    
    
    


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        self.has_continuous_action_space = has_continuous_action_space   ## ensure no discrete output
        super(Critic, self).__init__()
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # TODO
        self.critic = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.Tanh(),
                nn.Conv2d(32, 20, kernel_size=4, stride=4),
                nn.Linear(5, 10),
                nn.Flatten(0),
                # nn.Softmax(dim=-1)
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
        self.state, _, _ = Env.step(self, action=1)
        self.target_pos = Env.__init__.target_pose()
        self.state_ = state
        self.q = 0    # q value for learning
        if has_continuous_action_space:
            self.action_std = action_std_init
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()

        self.policy = Actor(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.target_q = self.critic_target_net(state, self.policy) 
        self.target_q = self._build_net_q(self.state_, self.policy, 'target_net', trainable=False)  # for target q
        self.eva_q = self._build_net_q(self.state_, self.policy, 'target_net', trainable=True)   # for eval q
        self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))   # TD error
        self.value = Critic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        


        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.value.critic.parameters(), 'lr': lr_critic}
                    ])
        self.policy_old = Actor(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())   # replace the old policy
        self.MseLoss = nn.MSELoss()  # loss function



        state, _, _ = Env.step()

    def _build_net_q(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def _build_net_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s, 30, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def critic_target_net(policy):  ## get target q
        R = Env.get_reward()
        target_q = R + GAMMA * self.q
        # nn.Conv2d(4, 32, kernel_size=8, stride=4),
        # nn.Tanh(),
        # nn.Conv2d(32, 64, kernel_size=4, stride=2),
        # nn.Tanh(),
        # nn.Conv2d(64, 1, kernel_size=3, stride=1),
        # nn.Flatten(),
        # nn.Linear(49, 512),
        # nn.Tanh(),
        # nn.Linear(49, action_dim=3),
        # nn.Flatten(),
        # nn.Softmax(dim=-1)
        return target_q
    

        

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
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values[0], rewards) - 0.01 * dist_entropy
            
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