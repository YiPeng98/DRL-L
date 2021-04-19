# coding=utf-8

import sys, os
current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path) # 将当前终端所在的路径添加到系统路径中

import gym
import torch
import numpy as np
import datetime
from common.plot import plot_rewards
from common.utils import save_results
from PPO.agent import PPO

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
SAVED_MODEL_PATH = current_path + "/saved_model/"
if not os.path.exists(SAVED_MODEL_PATH):
    os.mkdir(SAVED_MODEL_PATH)
RESULT_PATH = current_path + "/results/"
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)

class PPOConfig:
    def __init__(self) -> None:
        self.env = "CartPole-v0"
        self.algo = 'PPO'
        self.n_epochs = 4 # 具体代表同一批数据的使用次数
        self.train_eps = 300
        self.update_fre = 20
        self.batch_size = 5 # 用minibatch方法时会定义batch_size，即把数据分几份，分批次训练
        self.gamma = 0.99
        self.actor_lr =0.0003
        self.critic_lr = 0.0003
        self.gae_lambda = 0.95 # 在计算优势函数时需要采用generalized advantage estimation技巧来平衡偏差和方差
        self.policy_clip = 0.2
        self.hidden_dim = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(cfg, env, agent):
    best_reward = env.reward_range[0] # 0：代表-inf，1：代表inf
    rewards= [] # 存储每一个episode的reward
    ma_rewards = [] # moving average rewards
    avg_reward = 0
    running_steps = 0 
    for i_episode in range(cfg.train_eps): # 玩300回合来产生数据并训练，并不代表训练的次数
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, ln_prob, val = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            running_steps += 1
            ep_reward += reward
            agent.memory.push(state, reward, action, done, ln_prob, val)
            if running_steps % cfg.update_fre == 0: # 不管此回合结束与否到达update_fre时就进行训练
                agent.update()
            state = state_
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        avg_reward = np.mean(rewards[-100:])
        if avg_reward > best_reward: # 当有更优的奖励值出现时，保存模型
            best_reward = avg_reward
            agent.save(SAVED_MODEL_PATH)
        print('Episode:{}/{}, Reward:{:.1f}, avg reward:{:.1f}, Done:{}'.format(i_episode+1,cfg.train_eps,ep_reward,avg_reward,done))
    return rewards, ma_rewards

if __name__ == '__main__':
    cfg = PPOConfig()
    env = gym.make(cfg.env)
    env.seed(1)
    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.n
    agent = PPO(state_dim, action_dim, cfg)
    rewards, ma_rewards = train(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='train', path=RESULT_PATH)
    plot_rewards(rewards, ma_rewards, tag="train", algo = cfg.algo, path=RESULT_PATH)