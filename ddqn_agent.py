import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from model import Model
from ReplayMemory import Replaymemory
from utils import *
from torch.autograd import Variable
import os 
import math
import glob
import random


class DDQN(object):
    def __init__(self,cfg, action_num):
        self.q_net, self.target_q_net = Model(action_num), Model(action_num)
        self.update_target_network()

        self.cfg = cfg
        self.action_num = action_num
        self.replaymemory = Replaymemory(self.cfg.memory_size)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.cfg.lr)
        self.loss_func = nn.MSELoss()

        self.prev_action = 0

    def choose_action(self, x, epsilon):
        self.q_net.to('cpu')
        # input only one sample
        if np.random.uniform() > epsilon:   # greedy
            x = torch.tensor(x, dtype=torch.float).unsqueeze(0)
            actions_value = self.q_net.forward(x)
            action = actions_value.max(1)[1].item()
        else:   # random
            action = np.random.randint(0, self.action_num)
        return action

    def update_target_network(self):
        # copy current_network to target network
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        assert set(self.q_net.state_dict())==set(self.target_q_net.state_dict())


    def train(self, fr):

        # sample batch transitions
        s0, a, r, s1, done = self.replaymemory.sample(self.cfg.batch_size)
        
        state = torch.from_numpy(s0).float()
        action = torch.from_numpy(a).float()
        state_new = torch.from_numpy(s1).float()
        terminal = torch.from_numpy(done).float()
        reward = torch.from_numpy(r).float()

        if self.cfg.cuda_num != None:
            state = state.to(self.cfg.cuda_num)
            action = action.to(self.cfg.cuda_num)
            state_new = state_new.to(self.cfg.cuda_num)
            terminal = terminal.to(self.cfg.cuda_num)
            reward = reward.to(self.cfg.cuda_num)
            self.q_net.to(self.cfg.cuda_num)
            self.target_q_net.to(self.cfg.cuda_num)

        state = torch.tensor(state)
        action = torch.tensor(action)
        state_new = torch.tensor(state_new)
        terminal = torch.tensor(terminal)
        reward = torch.tensor(reward)

        q_values = self.q_net.forward(state)
        next_q_values = self.q_net.forward(state_new)
        next_target_q_values = self.target_q_net.forward(state_new)

        # -- target -- #
        # torch.max(next_q_values, 1)[1] -> index of max q value [B]
        #  torch.max(next_q_values, 1)[1].unsqueeze(1) -> B, 1
        # target next q ????????? next q value??? max ?????? ???????????? ???????????? q ?????? [B] ?????? ?????????
        next_q_val = next_target_q_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        q_target = reward + self.cfg.discount_factor * next_q_val * (1 - terminal)

        # -- output -- #
        action = action.unsqueeze(1)
        q_val = q_values.gather(1, action.type(torch.int64)) # [B, 1]
        q_val = q_val.squeeze(1) # [B, 1]
        
        loss = self.loss_func(q_val, q_target.detach())

        # backward optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if fr % self.cfg.update_target_frequency == 0:
            self.update_target_network()

        return loss.data, q_val

    def save(self, step, logs_path, exp_time, fr):
        os.makedirs(logs_path, exist_ok=True)
        logs_path = os.path.join(logs_path, '{}.pth' .format(exp_time+'_'+fr))
        self.q_net.save(logs_path, step=step, optimizer=self.optimizer)
