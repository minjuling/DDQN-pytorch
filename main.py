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

from torch.utils.tensorboard import SummaryWriter

logger = get_logger(cfg)
exp_time = get_tensorboard_name()
print("exp_time", exp_time)
# logger.info(cfg)
writer = SummaryWriter(cfg.tensorboard_path+exp_time)

# handle the atari env
env = FrameStack(AtariPreprocessing(gym.make(cfg.env)), 4)


N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

msg = '[{time}]' 'starts experiments setting '\
            '{exp_name}'.format(time = time.ctime(), 
            exp_name = exp_time)
logger.info(msg)
logger.info("=> creating model ...")

ddqn = DDQN(cfg, N_ACTIONS, N_STATES)
reward_logs = []
loss_logs = []

training_step = 0

print('\nCollecting experience...')
s = env.reset()
ep_r = 0
step = 0
ep_num = 0
losses = []
all_rewards = []
is_win = False


for fr in range(1, cfg.frames+1):
    
    # while True:
        # env.render()
    a = ddqn.choose_action(s, fr)

    # take action4esrdx
    s_, r, done, info = env.step(a)

    ddqn.replaymemory.append((torch.unsqueeze(torch.FloatTensor(s), 0), a, r, torch.unsqueeze(torch.FloatTensor(s_), 0), done))
    step += 1
    ep_r += r
    s = s_

    if len(ddqn.replaymemory.buffer) > cfg.batch_size:
        loss, q_val = ddqn.train(fr)

        # save model
        if training_step % cfg.save_logs_frequency == 0:
            ddqn.save(fr, cfg.logs_path, exp_time)
            writer.add_scalar('QValue/Step', q_val.mean(), fr-cfg.batch_size)
            writer.add_scalar('loss/Step', loss, fr-cfg.batch_size)
        
        # print reward and loss
        if (fr - cfg.batch_size) % cfg.show_loss_frequency == 0: 
            loss_logger = 'frames: {} epsilon: {} ep_num: {} Reward: {} Loss: {}' .format(fr-cfg.batch_size,  ddqn.epsilon, ep_num, np.mean(all_rewards[-10:]), loss)
            logger.info(loss_logger)

        if done:
            all_rewards.append(ep_r)
            writer.add_scalar('100 epi_reward/Episode', float(np.mean(all_rewards[-100:])), ep_num)
            writer.add_scalar('epi_reward/Episode', all_rewards[-1], ep_num)

            # -- reset -- #
            s = env.reset()
            ep_r = 0
            ep_num += 1

            if len(all_rewards) >= 100 and np.mean(all_rewards[-100:]) >= cfg.win_reward and all_rewards[-1] > cfg.win_reward:
                is_win = True
                ddqn.save(fr, cfg.logs_path, exp_time + 'best')
                msg = 'Ran {} episodes best 100-episodes average reward is {}. Solved after {} trials ✔'.format(ep_num, np.mean(all_rewards[-100:], ep_num - 100))
                logger.info(msg)
                break
        
            if not is_win:
                ddqn.save(fr, cfg.logs_path, exp_time + 'not_win')
            
        training_step += 1