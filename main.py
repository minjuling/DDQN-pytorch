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
from model import Model
from ReplayMemory import Replaymemory
from utils import *
from dqn_agent import DQN
from gym.wrappers import AtariPreprocessing, FrameStack
from torch.autograd import Variable
from config import Config as cfg

env = FrameStack(AtariPreprocessing(gym.make('BreakoutNoFrameskip-v4')), 4)
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

dqn = DQN(cfg, N_ACTIONS, N_STATES, ENV_A_SHAPE)


print('\nCollecting experience...')
for i_episode in range(400):
    s = env.reset()
    s = torch.unsqueeze(torch.FloatTensor(s), 0)
    ep_r = 0
    while True:
        env.render()
        # s = convert(s, screen_height, screen_width)
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)
        s_ = torch.unsqueeze(torch.FloatTensor(s_), 0)

        dqn.replaymemory.append((s, a, r, s_, done))

        ep_r += r
        if len(dqn.replaymemory.buffer) > cfg.memory_capacity:
            dqn.train()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_