# coding=utf-8

import os, sys
sys.path.append(os.getcwd()) # os.getcwd()获取当前路径

from itertools import count
import datetime
import gym
import torch
from PolicyGradient.agent import PolicyGradient

from common.plot import plot_rewards
from common.utils import save_results

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间

SAVED_MODEL_PATH = os.path.split(os.path.abspath(__file__))[0] + "/saved_model/" + '/' # 保存模型的路径
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0] + "/saved_model/"):
    os.mkdir(os.path.split(os.path.abspath(__file__))[0] + "/saved_model/" )
if not os.path.exists(SAVED_MODEL_PATH):
    os.mkdir(SAVED_MODEL_PATH)

RESULT_PATH = os.path.split(os.path.abspath(__file__))[0] + "/results/" + SEQUENCE + '/' # path to save reward
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0] + "/results/"):
    os.mkdir(os.path.split(os.path.abspath(__file__))[0] + "/results/")
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PGConfig:
    def __init__(self):
        self.gamma = 0.99
        self.train_eps = 300
        self.lr = 0.01
        self.batch_size = 8
        self.hidden_dim = 36 # 涉及到了神经网路就要加上关于神经元的参数

def train(cfg, env, agent):
    '''pool存放轨迹序列，用于梯度下降'''
    state_pool = [] # 存放batch_size个episode的state序列（下同）
    action_pool = [] # 是列表属于numpy类型，所以后边要进行tensor的转换
    reward_pool = []
    '''存储每个episode的reward用于绘图'''
    rewards = [] 
    ma_rewards = []
    for i_episode in range(cfg.train_eps): # 外循环：游戏的回合数
        state = env.reset()
        state = torch.from_numpy(state).to(device)
        ep_reward = 0
        while True: # 内循环：一个回合内游戏的进行
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                reward = 0 # 为什么要将reward设置为0呢
            state_pool.append(state)
            action_pool.append(action)
            reward_pool.append(reward)
            state = torch.from_numpy(next_state).to(device)
            if done:
                print('Episode:', i_episode, 'Reward:', ep_reward)
                break
        if i_episode > 0 and i_episode % cfg.batch_size == 0:
            agent.update(reward_pool, state_pool, action_pool)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Compele Training!')
    return rewards, ma_rewards

def eval(cfg, env, agent):
    agent.load_model(SAVED_MODEL_PATH) # 测试的时候只需要加载训练好的模型就可以
    '''存储每个episode的reward用于绘图'''
    rewards = [] 
    ma_rewards = []
    for i_episode in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        while True: # 内循环：一个回合内游戏的进行
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                reward = 0 # 为什么要将reward设置为0呢
            # state_pool.append(state)
            # action_pool.append(action)
            # reward_pool.append(reward)
            state = next_state
            if done:
                print('Episode:', i_episode, 'Reward:', ep_reward)
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print("Complete Evaluating!")
    return rewards, ma_rewards

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(1) # seed()用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed()值，则每次生成的随机数相同，设置仅一次有效
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    cfg = PGConfig()
    agent = PolicyGradient(state_dim, cfg, device)
    # rewards, ma_rewards = eval(cfg, env, agent)
    rewards, ma_rewards = train(cfg, env, agent)
    agent.save_model(SAVED_MODEL_PATH)
    save_results(rewards, ma_rewards, tag='train', path=RESULT_PATH)
    plot_rewards(rewards, ma_rewards, tag='train', algo="Policy Gradient", path=RESULT_PATH)