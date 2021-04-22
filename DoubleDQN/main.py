# coding=utf-8

import os, sys

# 在引用自定义包之前，添加系统路径
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

import torch
import gym
from agent import DoubleDQN
from common.utils import mk_dir, del_empty_dir, save_results
from common.plot import plot_rewards

# 创建相关文件夹
SAVED_MODEL_PATH = curr_path + '/save_model/'
RESULT_PATH = curr_path + '/results/'
mk_dir(SAVED_MODEL_PATH, RESULT_PATH)
del_empty_dir(SAVED_MODEL_PATH, RESULT_PATH)

class DoubleDQNConfig:
    def __init__(self):
        self.train_eps = 300
        self.train_steps = 200
        self.lr = 0.001
        self.gamma = 0.99
        self.target_update = 20
        self.epsilon_start = 1
        self.epsilon_end = 0.1
        self.epsilon_decay = 200
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = 128
        self.env = 'CartPole-v0'
        self.memory_capacity = 10000
        self.batch_size = 64
        self.algo = 'DoubleDQN'

def train(env, agent, cfg):
    print('Start to train !')
    rewards = []
    ma_rewards = []
    ep_steps = []
    for i_eps in range(cfg.train_eps):
        ep_reward = 0
        state = env.reset()
        for i_step in range(cfg.train_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            agent.update()
            if done:
                break
        if i_eps % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print('Episode:{}/{}, Reward:{}, Steps:{}, Done:{}'.format(i_eps+1, cfg.train_eps, ep_reward, i_step, done))
        ep_steps.append(i_step+1)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Complete Training !')
    return rewards, ma_rewards

if __name__ == "__main__":
    cfg = DoubleDQNConfig()
    env = gym.make(cfg.env)
    env.seed(1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DoubleDQN(state_dim, action_dim, cfg)
    rewards, ma_rewards = train(env, agent, cfg)
    agent.save(path=SAVED_MODEL_PATH)
    save_results(rewards,ma_rewards,tag='train',path=RESULT_PATH)
    plot_rewards(rewards,ma_rewards,tag="train",algo = cfg.algo,path=RESULT_PATH)