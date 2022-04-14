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

env = FrameStack(AtariPreprocessing(gym.make('BreakoutNoFrameskip-v4')), 4)
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

ddqn = DDQN(cfg, N_ACTIONS, N_STATES, ENV_A_SHAPE)
reward_logs = []
loss_logs = []

print('\nCollecting experience...')
for i_episode in range(400):
    s = env.reset()
    s = torch.unsqueeze(torch.FloatTensor(s), 0)
    ep_r = 0
    total_reward = 0
    step = 0

    while True:
        env.render()
        # s = convert(s, screen_height, screen_width)
        a = ddqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)
        s_ = torch.unsqueeze(torch.FloatTensor(s_), 0)
        total_reward += r

        ddqn.replaymemory.append((s, a, r, s_, done))
        step += 1
        ep_r += r
        if len(ddqn.replaymemory.buffer) > cfg.memory_capacity:
            # save model
            if i_episode % cfg.save_logs_frequency == 0:
                ddqn.save(i_episode, cfg.logs_path)
                np.save(os.path.join(cfg.logs_path, 'loss.npy'), np.array(loss_logs))
                np.save(os.path.join(cfg.logs_path, 'reward.npy'), np.array(reward_logs))
            
            loss = ddqn.train()

            loss_logs.extend([[i_episode, loss]]) 
            reward_logs.extend([[i_episode, total_reward]]) 

            if done:
                # print reward and loss
                if i_episode % cfg.show_loss_frequency == 0: 
                    print('Episode: {} step: {} Reward: {:.3f} Loss: {:.3f}' .format(i_episode, step, total_reward, loss))
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_