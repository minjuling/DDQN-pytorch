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

logger = get_logger(cfg)
exp_time = get_tensorboard_name()
print("exp_time", exp_time)
logger.info(cfg)
writer = SummaryWriter(cfg.tensorboard_path+exp_time)

env = FrameStack(AtariPreprocessing(gym.make('Breakout-v0'), frame_skip = 1), 4)
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

# msg = '[{time}]' 'starts experiments setting '\
#             '{exp_name}'.format(time = time.ctime(), 
#             exp_name = exp_time)
# logger.info(msg)
logger.info("=> start test ...")

reward_logs = []
loss_logs = []


# print('\nCollecting experience...')
s = env.reset()
s = torch.unsqueeze(torch.FloatTensor(s), 0)
ep_r = 0
step = 0
q_net = Model(N_ACTIONS)

while True:
    env.render()
    
    q_net.eval().to('cpu')
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
        writer.add_scalar('QValue/Step', actions_value, step)
        writer.add_scalar('r/Episode', ep_r, step)

        log = 'step: {}  Reward: {:.3f} ' .format(step, ep_r)
        logger.info(log)
        print(log)
        
    if done:
        log = 'Total Step: {}  Total Reward: {:.3f} ' .format(step, ep_r)
        logger.info(log)
        break
    s = s_