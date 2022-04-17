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

from torch.utils.tensorboard import SummaryWriter

logger = get_logger(cfg)
exp_time = get_tensorboard_name()
print("exp_time", exp_time)
logger.info(cfg)
writer = SummaryWriter(cfg.tensorboard_path+exp_time)

env = FrameStack(AtariPreprocessing(gym.make('BreakoutNoFrameskip-v0'), frame_skip = 1), 4)
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

msg = '[{time}]' 'starts experiments setting '\
            '{exp_name}'.format(time = time.ctime(), 
            exp_name = exp_time)
logger.info(msg)
logger.info("=> creating model ...")

ddqn = DDQN(cfg, N_ACTIONS, N_STATES, ENV_A_SHAPE)
reward_logs = []
loss_logs = []

training_step = 0

print('\nCollecting experience...')
for i_episode in range(cfg.total_episode):
    s = env.reset()
    s = torch.unsqueeze(torch.FloatTensor(s), 0)
    ep_r = 0
    step = 0

    while True:
        # env.render()
        
        a = ddqn.choose_action(s)

        # take action4esrdx
        s_, r, done, info = env.step(a)
        s_ = torch.unsqueeze(torch.FloatTensor(s_), 0)

        ddqn.replaymemory.append((s, a, r, s_, done))
        step += 1
        ep_r += r
        if len(ddqn.replaymemory.buffer) > cfg.memory_capacity:
            
            
            loss, q_val = ddqn.train()

            

            # save model
            if training_step % cfg.save_logs_frequency == 0:
                ddqn.save(i_episode, cfg.logs_path, exp_time)
                writer.add_scalar('QValue/Step', q_val.mean(), training_step)
                writer.add_scalar('loss/Step', loss, training_step)
            
            

            if done:
                writer.add_scalar('total_reward/Episode', ep_r, i_episode)

                # print reward and loss
                if i_episode % cfg.show_loss_frequency == 0: 
                    loss_logger = 'Episode: {} step: {} training_step: {} Reward: {:.3f} Loss: {:.3f}' .format(i_episode, step, training_step, ep_r, loss)
                    logger.info(loss_logger)
                    print(loss_logger)
                
                if i_episode % cfg.update_target_frequency == 0:
                    ddqn.update_target_network()
                
            training_step += 1

        if done:
            break
        s = s_