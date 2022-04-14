import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from model import Model
from ReplayMemory import Replaymemory
from utils import *
from torch.autograd import Variable


class DQN(object):
    def __init__(self,cfg, action_num, state_num, env_a_shape):
        self.q_net, self.target_q_net = Model(action_num), Model(action_num)

        self.cfg = cfg
        self.action_num = action_num
        self.state_num = state_num
        self.env_a_shape = env_a_shape
        self.learn_step_counter = 0                                     # for target updating
        self.replaymemory = Replaymemory(self.cfg.memory_size)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.cfg.lr)
        self.loss_func = nn.MSELoss()
        self.action_num = action_num

    def choose_action(self, x):
        self.q_net.eval()
        # input only one sample
        if np.random.uniform() < self.cfg.epsilon:   # greedy
            actions_value = self.q_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if self.env_a_shape == 0 else action.reshape(self.env_a_shape)  # return the argmax index
        else:   # random
            action = np.random.randint(0, self.action_num)
            action = action if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
        return action

    def train(self):
        # target parameter update
        if self.learn_step_counter % self.cfg.update_target_frequency == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        batch_state, batch_action, batch_reward, batch_state_new, \
        batch_over = self.replaymemory.sample(self.cfg.batch_size)
        
        state = torch.from_numpy(batch_state).float()/255.0
        action = torch.from_numpy(batch_action).float()
        state_new = torch.from_numpy(batch_state_new).float()/255.0
        terminal = torch.from_numpy(batch_over).float()
        reward = torch.from_numpy(batch_reward).float()

        state = Variable(state)
        action = Variable(action)
        state_new = Variable(state_new)
        terminal = Variable(terminal)
        reward = Variable(reward)
        self.q_net.eval()
        self.target_q_net.eval()
        
        # use current network to evaluate action argmax_a' Q_current(s', a')_
        action_new = self.q_net.forward(state_new).max(dim=1)[1].cpu().data.view(-1, 1)
        action_new_onehot = torch.zeros(self.cfg.batch_size, self.action_num)
        action_new_onehot = Variable(action_new_onehot.scatter_(1, action_new, 1.0))
        
        # use target network to evaluate value y = r + discount_factor * Q_tar(s', a')
        y = (reward + torch.mul(((self.target_q_net.forward(state_new)*action_new_onehot).sum(dim=1)*terminal), self.cfg.discount_factor))
        
        # regression Q(s, a) -> y
        self.q_net.train()
        Q = (self.q_net.forward(state)*action.unsqueeze(1)).sum(dim=1)
        loss = F.mse_loss(input=Q, target=y.detach())
        
        # backward optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()