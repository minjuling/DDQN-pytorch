"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from utils import *
from ddqn_agent import DDQN
from gym.wrappers import AtariPreprocessing, FrameStack
from config import Config as cfg
import os
from model import Model

from torch.utils.tensorboard import SummaryWriter

path = '/usr/src/DDQN-pytorch/logs/model-4_17_16_11.pth'

# logger = get_logger(cfg)
exp_time = get_tensorboard_name()
print("exp_time", exp_time)
# writer = SummaryWriter(cfg.tensorboard_path+exp_time)

env = FrameStack(AtariPreprocessing(gym.make('Breakout-v0'), frame_skip = 1), 4)
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

# msg = '[{time}]' 'starts experiments setting '\
#             '{exp_name}'.format(time = time.ctime(), 
#             exp_name = exp_time)
# logger.info(msg)
# logger.info("=> start test ...")

reward_logs = []
loss_logs = []


# print('\nCollecting experience...')
s = env.reset()
s = torch.unsqueeze(torch.FloatTensor(s), 0)
ep_r = 0
step = 0

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.05
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 750000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(4, 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #make sure input tensor is flattened
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)

q_net = Model(N_ACTIONS)
# model = torch.load(path)
# print(model)
# print(q_net.state_dict())
q_net.load_state_dict(torch.load(path)['state_dict'])

while True:
    
    q_net.eval()
    actions_value = q_net.forward(s)
    action = torch.max(actions_value, 1)[1].data.numpy()
    action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index

    # take action4esrdx
    s_, r, done, info = env.step(action)
    s_ = torch.unsqueeze(torch.FloatTensor(s_), 0)

    step += 1
    ep_r += r
        
    # save model
    if step % 10 == 0:
        # writer.add_scalar('QValue/Step', actions_value.mean(), step)
        # writer.add_scalar('r/Episode', ep_r, step)

        log = 'step: {}  Reward: {:.3f} ' .format(step, ep_r)
        # logger.info(log)
        print(log)
        
    if done:
        log = 'Total Step: {}  Total Reward: {:.3f} ' .format(step, ep_r)
        # logger.info(log)
        break
    s = s_