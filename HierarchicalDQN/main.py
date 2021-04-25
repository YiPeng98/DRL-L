# coding=utf-8

# 添加系统路径
import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

# 创建相关文件夹
from common.utils import mk_dir,del_empty_dir, save_results
SAVED_MODEL_PATH = curr_path + '/saved_model/'
RESULT_PATH = curr_path + '/results/'
mk_dir(SAVED_MODEL_PATH, RESULT_PATH)
# del_empty_dir(SAVED_MODEL_PATH, RESULT_PATH)

import numpy as np
import torch
import gym
from common.plot import plot_rewards, plot_losses
from agent import HierarchicalDQN

class HierarchicalDQNConfig:
    def __init__(self):
        self.algo = 'H-DQN'
        self.gamma = 0.99
        self.epsilon_start = 1
        self.epsilon_end = 0.01
        self.epsilon_decay = 200
        self.lr = 0.0001
        self.memory_capacity = 10000
        self.batch_size = 32
        self.train_eps = 300
        self.train_update = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = 256

def train(cfg, env, agent):
    print('Start to train!')
    rewards = []
    ma_rewards = []
    for i_episode in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        # 高层循环
        while not done:
            '''负责高层策略的制定'''
            goal = agent.set_goal(state) # 输入状态后产生高层的一个目标
            onehot_goal = agent.to_onehot(goal)
            meta_state = state # 原始状态
            extrinsic_reward = 0 # 外部奖励是指在高层下的奖励
            # 低层循环
            '''负责低层策略的执行 '''
            while not done and goal != np.argmax(state): # 此处限定条件为目标达到state值最大的索引,条件满足后重新产生高层策略进行迭代
                goal_state = np.concatenate([state, onehot_goal]) # 将高层的知道策略作为低层的考虑内容
                action = agent.choose_action(goal_state) # 此处是底层agent结合高层给出的指示输出相关动作
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                extrinsic_reward += reward
                intrinsic_reward = 1.0 if goal == np.argmax(next_state) else 0.0 # 内部奖励是指在低层策略下的奖励，是我们自己设定的，当我们的下一个状态达到高层指定的策略goal时就给出奖励
                # 此经验池中存储的是低层每一步的数据
                agent.memory.push(goal_state, action, intrinsic_reward, np.concatenate([next_state, onehot_goal]), done)
                state = next_state
                agent.update()
            # 此经验池中每达到一次高层限定的目标时就存储一次数据
            agent.meta_memory.push(meta_state, goal, extrinsic_reward, state, done)
        print('Episode:{}/{}, Reward:{}, Loss:{:.2f}, Meta_Loss:{:.2f}'.format(i_episode+1, cfg.train_eps, ep_reward, agent.loss_numpy, agent.meta_loss_numpy))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Complete training !')
    return rewards, ma_rewards

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(1)
    cfg = HierarchicalDQNConfig()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = HierarchicalDQN(state_dim, action_dim, cfg)
    rewards, ma_rewards = train(cfg, env, agent)
    save_results(rewards, ma_rewards, 'train', RESULT_PATH)
    plot_rewards(rewards, ma_rewards, 'train', RESULT_PATH)
    plot_losses(agent.losses, cfg.algo, RESULT_PATH)




