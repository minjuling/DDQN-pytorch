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
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from config import Config as cfg
import math

from torch.utils.tensorboard import SummaryWriter

logger = get_logger(cfg)
exp_time = get_tensorboard_name()
writer = SummaryWriter(cfg.tensorboard_path+exp_time)
print("exp_time", exp_time)

# handle the atari env
env = make_atari('PongNoFrameskip-v4')
env = wrap_deepmind(env, frame_stack=4)
env = wrap_pytorch(env)

N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape

msg = '[{time}]' 'starts experiments setting '\
            '{exp_name}'.format(time = time.ctime(), 
            exp_name = exp_time)
logger.info(msg)
logger.info("=> creating model ...")

ddqn = DDQN(cfg, N_ACTIONS)

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

epsilon_final = cfg.epsilon_min
epsilon_start = cfg.epsilon
epsilon_decay = cfg.eps_decay
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
    -1. * frame_idx / epsilon_decay)


for fr in range(1, cfg.frames+1):
    
    # while True:
        # env.render()
    epsilon = epsilon_by_frame(fr)
    a = ddqn.choose_action(s, epsilon)

    # take action4esrdx
    s_, r, done, info = env.step(a)

    ddqn.replaymemory.append((torch.unsqueeze(torch.FloatTensor(s), 0), a, r, torch.unsqueeze(torch.FloatTensor(s_), 0), done))
    step += 1
    ep_r += r
    s = s_

    if len(ddqn.replaymemory.buffer) > cfg.batch_size:
        loss, q_val = ddqn.train(fr)

        # save model & tensorboard writer
        if training_step % cfg.save_logs_frequency == 0:
            ddqn.save(fr, cfg.logs_path, exp_time, str(fr))
            writer.add_scalar('QValue/Step', q_val.mean(), fr-cfg.batch_size)
            writer.add_scalar('loss/Step', loss, fr-cfg.batch_size)
        
        # print reward and loss
        if (fr - cfg.batch_size) % cfg.show_loss_frequency == 0: 
            loss_logger = 'frames: {} epsilon: {} ep_num: {} Reward: {} Loss: {}' .format(fr-cfg.batch_size,  epsilon, ep_num, np.mean(all_rewards[-10:]), loss)
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
                ddqn.save(fr, cfg.logs_path, exp_time + 'best', str(fr))
                break
            
        training_step += 1