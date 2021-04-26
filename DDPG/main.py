# coding=utf-8

import sys, os
# 添加环境变量到系统中去
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

import torch
import gym
import numpy as np
from common.utils import save_results, mk_dir, del_empty_dir
from common.plot import plot_rewards, plot_losses
from env import NormalizedActions, OUNoise
from agent import DDPG

# 创建相关存储文件夹

SAVED_MODEL_PATH = curr_path + '/saved_model/'
RESULT_PATH = curr_path + '/results/'
mk_dir(SAVED_MODEL_PATH, RESULT_PATH)

class DDPGConfig:
    def __init__(self):
        self.env = 'Pendulum-v0'
        self.algo = 'DDPG'
        self.gamma = 0.99
        self.critic_lr = 1e-3  
        self.actor_lr = 1e-4 
        self.memory_capacity = 10000
        self.batch_size = 128
        self.train_eps =300
        self.eval_eps = 200
        self.eval_steps = 200
        self.hidden_dim = 30
        self.soft_tau=1e-2 # 更新目标网络进行软更新
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def train(cfg, env, agent):
    print("Start to train !")
    ou_noise = OUNoise(env.action_space) # 动作噪声
    rewards = []
    ma_rewards = []
    ep_steps = []
    for i_episode in range(cfg.train_eps):
        state = env.reset()
        ou_noise.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)
            action = ou_noise.get_action(action, i_step)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
        print('Episode:{}/{}, Reward:{}'.format(i_episode+1,cfg.train_eps,ep_reward))
        ep_steps.append(i_step)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Complete training！')
    return rewards,ma_rewards

if __name__ == '__main__':
    cfg = DDPGConfig()
    env = NormalizedActions(gym.make('Pendulum-v0'))
    env.seed(1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPG(state_dim, action_dim, cfg)
    rewards,ma_rewards = train(cfg, env, agent)
    agent.save(path=SAVED_MODEL_PATH)
    save_results(rewards, ma_rewards, tag='train', path=RESULT_PATH)
    plot_rewards(rewards, ma_rewards, tag="train", algo=cfg.algo, path=RESULT_PATH)