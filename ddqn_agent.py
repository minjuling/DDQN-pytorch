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
    def __init__(self,cfg, action_num, state_shape, env_a_shape):
        self.q_net, self.target_q_net = Model(state_shape, action_num), Model(state_shape, action_num)
        self.update_target_network()

        self.cfg = cfg
        self.action_num = action_num
        self.env_a_shape = env_a_shape
        self.replaymemory = Replaymemory(self.cfg.memory_size)
        print("self.cfg.lr", self.cfg.lr)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.cfg.lr)
        self.loss_func = nn.MSELoss()

        epsilon_final = self.cfg.epsilon_min
        epsilon_start = self.cfg.epsilon
        epsilon_decay = self.cfg.eps_decay
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)
        
        self.epsilon = None
        
        
        self.prev_action = 0

    def choose_action(self, x, fr):
        self.q_net.to('cpu')
        # input only one sample
        self.epsilon = self.epsilon_by_frame(fr)
        # if np.random.uniform() > self.epsilon:   # greedy
        if np.random.uniform() > 0:   # greedy
            x = torch.tensor(x, dtype=torch.float).unsqueeze(0)
            actions_value = self.q_net.forward(x)
            action = actions_value.max(1)[1].item()
            # action = action[0] if self.env_a_shape == 0 else action.reshape(self.env_a_shape)  # return the argmax index
            # if self.prev_action != action:
            #     print(self.prev_action, action)
            #     self.prev_action = action
            # exit()
        else:   # random
            action = np.random.randint(0, self.action_num)
            # action = action if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
        return action

    def update_target_network(self):
        # copy current_network to target network
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        assert set(self.q_net.state_dict())==set(self.target_q_net.state_dict())
        # print("update target", self.q_net.parameters(), self.target_q_net.parameters())


    def train(self, fr):

        # sample batch transitions
        s0, a, r, s1, done = self.replaymemory.sample(self.cfg.batch_size)
        
        state = torch.from_numpy(s0).float()
        action = torch.from_numpy(a).float()
        state_new = torch.from_numpy(s1).float()
        terminal = torch.from_numpy(done).float()
        reward = torch.from_numpy(r).float()

        state = state.to('cuda:1')
        action = action.to('cuda:1')
        state_new = state_new.to('cuda:1')
        terminal = terminal.to('cuda:1')
        reward = reward.to('cuda:1')

        state = Variable(state)
        action = Variable(action)
        state_new = Variable(state_new)
        terminal = Variable(terminal)
        reward = Variable(reward)
        self.q_net.to('cuda:1')
        self.target_q_net.to('cuda:1')
        
        # print(action.shape, state_new.shape, terminal.shape, reward.shape)
        # torch.Size([32]) torch.Size([32, 4, 84, 84]) torch.Size([32]) torch.Size([32])

        q_values = self.q_net.forward(state)
        next_q_values = self.q_net.forward(state_new)
        next_target_q_values = self.target_q_net.forward(state_new)

        # -- target -- #
        # torch.max(next_q_values, 1)[1] -> index of max q value [B]
        #  torch.max(next_q_values, 1)[1].unsqueeze(1) -> B, 1
        # target next q 중에서 next q value의 max 값에 해당하는 인덱스의 q 값을 [B] 만큼 가져옴
        next_q_val = next_target_q_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        q_target = reward + self.cfg.discount_factor * next_q_val * (1 - terminal)
        # print(next_q_values.shape, next_q_values)
        # print(next_q_val.shape, next_q_val)
        # print(q_target.shape, q_target)
        # print(reward, self.cfg.discount_factor, next_q_val, terminal)
        # torch.Size([32, 6])
        # torch.Size([32])
        # torch.Size([32])
        # exit()

        # -- output -- #
        # self.q_net.train()
        action = action.unsqueeze(1)
        q_val = q_values.gather(1, action.type(torch.int64)) # [B, 1]
        q_val = q_val.squeeze(1) # [B, 1]
        
        loss = (q_val - q_target.detach().to('cuda:1')).pow(2).mean()
        if fr % 5000 == 0:
            print(q_val, q_target)

        # backward optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if fr % self.cfg.update_target_frequency == 0:
            self.update_target_network()

        return loss.data, q_val

    def save(self, step, logs_path, exp_time):
        os.makedirs(logs_path, exist_ok=True)
        model_list =  glob.glob(os.path.join(logs_path, '*.pth'))
        # if len(model_list) > self.cfg.maximum_model - 1 :
        #     min_step = min([int(li.split('/')[-1][6:-4]) for li in model_list]) 
        #     os.remove(os.path.join(logs_path, 'model-{}.pth' .format(min_step)))
        logs_path = os.path.join(logs_path, 'model-{}.pth' .format(exp_time))
        self.q_net.save(logs_path, step=step, optimizer=self.optimizer)
        # print('=> Save {}' .format(logs_path)) 