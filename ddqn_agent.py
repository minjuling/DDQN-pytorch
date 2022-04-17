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
import glob


class DDQN(object):
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
        self.q_net.eval().to('cpu')
        # input only one sample
        if np.random.uniform() < self.cfg.epsilon_start:   # greedy
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
        self.q_net.eval().to('cuda:1')
        self.target_q_net.eval().to('cuda:1')
        
        # print(action.shape, state_new.shape, terminal.shape, reward.shape)
        # torch.Size([32]) torch.Size([32, 4, 84, 84]) torch.Size([32]) torch.Size([32])

        # -- target -- #
        next_q_values = self.q_net.forward(state_new)
        next_target_q_values = self.q_net.forward(state_new)

        # torch.max(next_q_values, 1)[1] -> index of max q value [B]
        #  torch.max(next_q_values, 1)[1].unsqueeze(1) -> B, 1
        # target next q 중에서 next q value의 max 값에 해당하는 인덱스의 q 값을 [B] 만큼 가져옴
        next_q_val = next_target_q_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)

        q_target = reward + self.cfg.discount_factor * next_q_val * (1 - terminal)

        # -- output -- #
        self.q_net.train()
        q_values = self.q_net.forward(state)

        action = action.unsqueeze(1)
        q_val = q_values.gather(1, action.type(torch.int64)) # [B, 1]
        q_val = q_val.squeeze(1) # [B, 1]
        
        loss = self.loss_func(q_val, q_target.detach().to('cuda:1'))
        
        # backward optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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