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
# from gym.wrappers import AtariPreprocessing, FrameStack
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from gym.wrappers import AtariPreprocessing, FrameStack
from config import Config as cfg
import os
import math

# from torch.utils.tensorboard import SummaryWriter

logger = get_logger(cfg)
exp_time = get_tensorboard_name()
print("exp_time", exp_time)
path = './logs/model-4_21_13_27best.pth'

# logger.info(cfg)
# writer = SummaryWriter(cfg.tensorboard_path+exp_time)

# handle the atari env
env = make_atari('PongNoFrameskip-v4')
env = wrap_deepmind(env, frame_stack=4)
env = wrap_pytorch(env)

N_ACTIONS = env.action_space.n

msg = '[{time}]' 'starts test setting '\
            '{exp_name}'.format(time = time.ctime(), 
            exp_name = exp_time)
logger.info(msg)
logger.info("=> creating model ...")

ddqn = DDQN(cfg, N_ACTIONS)
ddqn.q_net.load(path)
reward_logs = []
loss_logs = []

ep_r = 0
step = 0

# for fr in range(1, cfg.frames+1):
s = env.reset()

while True:
    env.render()
    a = ddqn.choose_action(s, 0)
    s_, r, done, info = env.step(a)

    step += 1
    ep_r += r
    s = s_
        
    # save model
    if step % 10 == 0:
        log = 'step: {}  Reward: {:.3f} ' .format(step, ep_r)
        logger.info(log)
        print(log)
        
    if done:
        log = 'Total Step: {}  Total Reward: {:.3f} ' .format(step, ep_r)
        logger.info(log)
        break