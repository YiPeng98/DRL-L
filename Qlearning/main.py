#!/usr/bin/env python
# conding=utf-8

import sys
import os
import numpy as np
curr_path = os.path.dirname(__file__)# __file__就是当前脚本运行的路径 e:/Work/Python/DRL-L/Qlearning/main.py ，dirname获取当前目录的父目录
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path) # add current terminal path to sys.path 是为了能够正确导入同级目录下的包

#########以上是将终端路径添加到系统路径中内，以便于引用自定义模块，注意必须在setting.json中设置才能生效！！！##########

import gym
import datetime
from envs.gridworld_env import CliffWalkingWapper, FrozenLakeWapper
from QLearning.agent import QLearning
from common.plot import plot_rewards
from common.utils import save_results

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间
SAVED_MODEL_PATH = curr_path + "/saved_model/" + SEQUENCE + "/" # 模型的存放位置
if not os.path.exists(curr_path + "/saved_model/"):
    os.mkdir(curr_path + "/saved_model/")
if not os.path.exists(SAVED_MODEL_PATH):
    os.mkdir(SAVED_MODEL_PATH)

RESULT_PATH = curr_path + "/results/" + SEQUENCE + "/" # 图像和奖励数据存放位置
if not os.path.exists(curr_path + "/results/"):
    os.mkdir(curr_path + "/results/")
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)

class QlearningConfig:
    def __init__(self):
        self.train_eps = 200 # 训练的回合数
        self.gamma = 0.9 # reward的衰减率
        self.epsilon_start = 0.99 # e-greedy策略中的初始epsilon
        self.epsilon_end = 0.01 # e-greedy策略中的终止epsilon
        self.epsilon_decay = 200 # e-greedy策略中epsilon的衰减率
        self.lr = 0.1 # 学习

def train(cfg, env, agent, render): # 训练Q表
    rewards = []
    ma_rewards = []
    steps = []
    for i_episode in range(cfg.train_eps):
        ep_reward = 0
        ep_steps = 0
        state = env.reset()
        while True:
            if render:
                env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            ep_steps += 1
            if done:
                break
        rewards.append(ep_reward)
        steps.append(ep_steps)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9 + ep_reward*0.1) # 移动平均奖励，上一回合和这一回合的奖励和
        else:
            ma_rewards.append(ep_reward)
        print("Episode:{}/{}; reward:{}".format(i_episode+1, cfg.train_eps, ep_reward))
    return rewards, ma_rewards

def eval(cfg, env, agent, render): # 测试Q表
    rewards = []
    ma_rewards = [] # 滑动平均的reward
    steps = []
    for i_episode in range(cfg.train_eps):
        ep_reward = 0
        ep_steps = 0
        state = env.reset()
        while True:
            if render:
                env.render()
            action = np.argmax(agent.Q_table[str(state)])
            next_state, reward, done, _ = env.step(action)
            # agent.update(state, action, reward, next_state, done) 在评估时不需要更新模型
            state = next_state
            ep_reward += reward
            ep_steps += 1
            if done:
                break
        rewards.append(ep_reward)
        steps.append(ep_steps)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9 + ep_reward*0.1) # 移动平均奖励，上一回合和这一回合的奖励和
        else:
            ma_rewards.append(ep_reward)
        print("Episode:{}/{}; reward:{}".format(i_episode+1, cfg.train_eps, ep_reward))
    return rewards, ma_rewards

if __name__ == "__main__":
    cfg = QlearningConfig()
    env = gym.make("CliffWalking-v0") # 0 up, 1 right, 2 down, 3 left
    env = CliffWalkingWapper(env)
    action_dim = env.env.action_space.n # 因为env是CliffWalkingWapper类打包后的环境，所以要调用属性值的话必须用env.env
    agent = QLearning(action_dim, cfg)
    rewards, ma_rewards = train(cfg, env, agent, False)
    eval(cfg, env, agent, True)
    agent.save(path=SAVED_MODEL_PATH)
    save_results(rewards, ma_rewards, tag='train', path=RESULT_PATH)
    plot_rewards(rewards, ma_rewards, tag='train', algo='Off-Policy First-Visit QLearning', path=RESULT_PATH)